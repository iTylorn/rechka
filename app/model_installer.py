"""Offline model inventory and explicitly requested, verified model updates."""

from dataclasses import replace
import json
from pathlib import Path
import threading

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from model_updates import (
    DownloadCancelled, MARKER_NAME, MODEL_CATALOG, ModelSpec, WESPEAKER_XML,
    check_update, installed_model_complete, install_model, validate_model,
)
from settings import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_FOLDER, DEFAULT_MODELS_FOLDER,
    MODEL_STORE_FOLDER,
)


def _directory(spec, path):
    path = Path(path)
    return path.parent if spec.kind == "wespeaker" and path.suffix.lower() == ".xml" else path


def _read_json(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _stored_revision(spec, path):
    if not path:
        return None
    marker = _read_json(_directory(spec, path) / MARKER_NAME)
    revision = marker.get("revision")
    if marker.get("repository") == spec.repo and isinstance(revision, str):
        return revision
    return None


def _legacy_key(path, kind):
    """Identify a legacy active path without scanning revision-history folders."""
    path = Path(path)
    if kind == "wespeaker":
        return "wespeaker"
    marker = _read_json(path / MARKER_NAME)
    for candidate in MODEL_CATALOG:
        if candidate.kind == kind and marker.get("repository") == candidate.repo:
            return candidate.key
    label = str(path).lower()
    for candidate in MODEL_CATALOG:
        if candidate.kind == kind and candidate.folder.lower() in label:
            return candidate.key
    config = _read_json(path / "config.json")
    if kind == "llm":
        identity = str(config.get("_name_or_path", "")).lower()
        if "qwen3-14b" in identity or (config.get("model_type") == "qwen3"
                                       and config.get("hidden_size") == 5120):
            return "qwen14"
        if "qwen3-8b" in identity or (config.get("model_type") == "qwen3"
                                      and config.get("hidden_size") == 4096):
            return "qwen8"
    return None


def find_installed_model(settings, spec):
    """Return one usable exact directory (or WeSpeaker XML), with no network I/O.

    The current user selection wins over a saved catalog path when it identifies
    this preset. Sibling revision directories are never searched.
    This is only a presence check, not a checksum or GPU validation.
    """
    paths = settings.get("model_paths", {}) or {}
    managed = paths.get(spec.key) if isinstance(paths, dict) else None
    candidates = []
    if spec.kind == "wespeaker":
        candidates.extend((settings.get("embed_model_path"), managed, DEFAULT_EMBEDDING_MODEL))
    else:
        active_key = "last_model" if spec.kind == "whisper" else "llm_model_path"
        folder_key = "models_folder" if spec.kind == "whisper" else "llm_models_folder"
        active = settings.get(active_key)
        if active and _legacy_key(active, spec.kind) == spec.key:
            candidates.append(active)
        candidates.append(managed)
        legacy = settings.get(folder_key)
        if legacy:
            candidates.append(Path(legacy) / spec.folder)
        default = DEFAULT_MODELS_FOLDER if spec.kind == "whisper" else DEFAULT_LLM_FOLDER
        candidates.append(default / spec.folder)
    base = Path(settings.get("model_store_folder") or MODEL_STORE_FOLDER)
    candidates.extend((base / spec.relative_dir, MODEL_STORE_FOLDER / spec.relative_dir))
    if spec.kind == "wespeaker":
        candidates.append(base / "diarization" / WESPEAKER_XML)
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if str(path) in seen:
            continue
        seen.add(str(path))
        try:
            if installed_model_complete(spec, path):
                return path
        except OSError:
            continue
    return None


def model_path_for(settings, key):
    spec = next(spec for spec in MODEL_CATALOG if spec.key == key)
    return find_installed_model(settings, spec)


def model_looks_complete(path, spec):
    """Compatibility alias for the old offline presence check."""
    return installed_model_complete(spec, path)


class ModelUpdateWorker(QThread):
    progress = Signal(int, str)
    checked = Signal(str, object)
    model_ready = Signal(str, str)
    model_error = Signal(str, str)
    completed = Signal(int, int)
    cancelled = Signal()

    def __init__(self, mode, specs, current_paths, base_folder=None,
                 plans=None, previous_paths=None, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.specs = list(specs)
        self.current_paths = dict(current_paths)
        self.base_folder = Path(base_folder) if base_folder else None
        self.plans = dict(plans or {})
        self.previous_paths = dict(previous_paths or {})
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise DownloadCancelled()

    def run(self):
        succeeded = failed = 0
        try:
            for index, spec in enumerate(self.specs):
                self._check_cancel()
                try:
                    current = self.current_paths.get(spec.key)
                    if self.mode == "check":
                        self.progress.emit(int(100 * index / len(self.specs)),
                                           f"Проверка: {spec.title}")
                        result = check_update(spec, current, cancel=self._cancel.is_set)
                        self._check_cancel()
                        self.checked.emit(spec.key, result)
                    elif self.mode == "rollback":
                        path = self.previous_paths[spec.key]
                        self.progress.emit(0, f"Проверка предыдущей версии на GPU: {spec.title}")
                        validate_model(spec, path, cancel=self._cancel.is_set)
                        self._check_cancel()
                        self.model_ready.emit(spec.key, str(path))
                    else:
                        plan = self.plans.get(spec.key)
                        if plan is None:
                            result = check_update(spec, current, cancel=self._cancel.is_set)
                            self.checked.emit(spec.key, result)
                            plan = result.plan
                        self._check_cancel()
                        if plan is None:
                            raise RuntimeError("Не удалось получить проверенный список файлов. Повторите проверку обновлений.")
                        path = install_model(
                            spec, self.base_folder, plan=plan, current_path=current,
                            progress=self.progress.emit, cancel=self._cancel.is_set,
                        )
                        self._check_cancel()
                        self.model_ready.emit(spec.key, str(path))
                    succeeded += 1
                except DownloadCancelled:
                    raise
                except Exception as exc:
                    self._check_cancel()
                    failed += 1
                    self.model_error.emit(spec.key, str(exc))
            self.completed.emit(succeeded, failed)
        except DownloadCancelled:
            self.cancelled.emit()


class ModelInstallerDialog(QDialog):
    def __init__(self, settings, parent=None, initial_selection=("whisper",)):
        super().__init__(parent)
        self.settings = settings
        self.worker = None
        self._closing = False
        self._changed = False
        self._save_errors = 0
        self._mode = None
        self._results = {}
        self._errors = {}
        self._current_paths = {}
        self.setWindowTitle("Модели: установка и обновления")
        self.resize(1120, 650)
        self.setMinimumSize(880, 570)
        layout = QVBoxLayout(self)
        introduction = QLabel(
            "Проверка обновлений обращается к Hugging Face только по вашей команде. "
            "Новая версия включается после проверки файлов и загрузки на GPU; предыдущая сохраняется для отката."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(str(settings.get("model_store_folder") or MODEL_STORE_FOLDER))
        self.choose_button = QPushButton("Выбрать папку…")
        self.choose_button.clicked.connect(self.choose_folder)
        folder_row.addWidget(QLabel("Хранить новые версии в:"))
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(self.choose_button)
        layout.addLayout(folder_row)

        self.table = QTableWidget(len(MODEL_CATALOG), 5)
        self.table.setHorizontalHeaderLabels(("Выбор", "Модель", "Локальная ревизия", "Hugging Face", "Состояние"))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 55)
        self.table.setColumnWidth(2, 155)
        self.table.setColumnWidth(3, 125)
        self.checkboxes = {}
        self._rows = {}
        for row, spec in enumerate(MODEL_CATALOG):
            self._rows[spec.key] = row
            checkbox = QCheckBox()
            checkbox.setAccessibleName(f"Выбрать {spec.title}")
            checkbox.setChecked(spec.key in initial_selection)
            self.checkboxes[spec.key] = checkbox
            self.table.setCellWidget(row, 0, checkbox)
            for column in range(1, 5):
                self.table.setItem(row, column, QTableWidgetItem())
            self.table.item(row, 1).setText(spec.title)
            self.table.item(row, 1).setToolTip(f"https://huggingface.co/{spec.repo}")
            self.table.setRowHeight(row, 66)
        self.table.currentCellChanged.connect(self._show_details)
        layout.addWidget(self.table, 1)
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.details_label)
        unknown_note = QLabel(
            "У моделей, установленных вручную, ревизия может быть неизвестна. "
            "Для них доступна установка проверенной версии в отдельную папку. "
            "Для отката выделите строку нужной модели."
        )
        unknown_note.setWordWrap(True)
        layout.addWidget(unknown_note)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Локальные модели найдены без подключения к интернету. Проверку можно запустить кнопкой ниже.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.check_button = QPushButton("Проверить обновления")
        self.check_button.clicked.connect(self.check_updates)
        self.install_button = QPushButton("Установить / обновить выбранные")
        self.install_button.clicked.connect(self.start_download)
        self.rollback_button = QPushButton("Откатить версию")
        self.rollback_button.clicked.connect(self.start_rollback)
        self.cancel_button = QPushButton("Отменить операцию")
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.cancel_button.setEnabled(False)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.reject)
        for button in (self.check_button, self.install_button, self.rollback_button,
                       self.cancel_button, self.close_button):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._refresh_inventory()
        self.table.setCurrentCell(0, 1)

    def _refresh_inventory(self):
        for spec in MODEL_CATALOG:
            self._current_paths[spec.key] = find_installed_model(self.settings, spec)
            self._refresh_row(spec)
        self._show_details()

    def _refresh_row(self, spec):
        row = self._rows[spec.key]
        path = self._current_paths.get(spec.key)
        result = self._results.get(spec.key)
        revision = _stored_revision(spec, path)
        if result is not None:
            revision = result.current_revision
        local_text = (revision[:12] if revision else "Неизвестна") if path else "Не установлена"
        self.table.item(row, 2).setText(local_text)
        self.table.item(row, 2).setToolTip(f"{revision or 'Ревизия неизвестна'}\n{path or 'Модель не найдена'}")
        remote = result.latest_revision if result else None
        self.table.item(row, 3).setText(remote[:12] if remote else "Не проверялась")
        self.table.item(row, 3).setToolTip(remote or "Проверка Hugging Face ещё не выполнялась.")
        status = "Найдена; не проверялась" if path else "Не установлена"
        detail = "Проверка обновлений также проверит локальные файлы по контрольным суммам."
        if result is not None:
            status = {
                "not_installed": "Доступна установка",
                "up_to_date": "Актуальна",
                "update_available": "Есть обновление",
                "unknown": "Ревизия неизвестна; доступна установка",
            }.get(result.status, "Не удалось определить")
            detail = result.detail or status
        if spec.key in self._errors:
            status = "Ошибка — см. подробности"
            detail = self._errors[spec.key]
        self.table.item(row, 4).setText(status)
        self.table.item(row, 4).setToolTip(detail)

    def _selected_spec(self):
        row = self.table.currentRow()
        return MODEL_CATALOG[row] if 0 <= row < len(MODEL_CATALOG) else None

    def _show_details(self, *_):
        spec = self._selected_spec()
        if spec is None:
            self.rollback_button.setEnabled(False)
            return
        path = self._current_paths.get(spec.key)
        previous = (self.settings.get("model_previous_paths", {}) or {}).get(spec.key)
        detail = self._errors.get(spec.key)
        if not detail and spec.key in self._results:
            detail = self._results[spec.key].detail
        self.details_label.setText(
            f"Текущая папка: {path or 'не установлена'}\n"
            f"Предыдущая версия: {previous or 'нет'}"
            + (f"\n{detail}" if detail else "")
        )
        self.rollback_button.setEnabled(bool(previous) and self.worker is None)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка для моделей", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    def _start_worker(self, mode, specs, base=None):
        if self.worker is not None:
            return
        self._mode = mode
        self._save_errors = 0
        for spec in specs:
            self._errors.pop(spec.key, None)
            if mode == "check":
                self._results.pop(spec.key, None)
            self._refresh_row(spec)
        self.progress_bar.setValue(0)
        self.status_label.setText("Выполняется проверка моделей…" if mode == "check" else "Подготовка моделей…")
        self.worker = ModelUpdateWorker(
            mode, specs, self._current_paths, base,
            plans={key: result.plan for key, result in self._results.items() if result.plan},
            previous_paths=self.settings.get("model_previous_paths", {}) or {}, parent=self,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.checked.connect(self.on_checked)
        self.worker.model_ready.connect(self.on_model_ready)
        self.worker.model_error.connect(self.on_model_error)
        self.worker.completed.connect(self.on_completed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self.on_worker_finished)
        self._set_busy(True)
        self.worker.start()

    def _set_busy(self, busy):
        for widget in (self.folder_edit, self.choose_button, self.check_button, self.install_button):
            widget.setEnabled(not busy)
        for checkbox in self.checkboxes.values():
            checkbox.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self._show_details()

    def check_updates(self):
        self._start_worker("check", list(MODEL_CATALOG))

    def start_download(self):
        selected = [spec for spec in MODEL_CATALOG if self.checkboxes[spec.key].isChecked()]
        if not selected:
            QMessageBox.information(self, "Модели", "Отметьте хотя бы одну модель в столбце «Выбор».")
            return
        base = Path(self.folder_edit.text().strip()).expanduser()
        if not base.is_absolute():
            QMessageBox.warning(self, "Папка моделей", "Укажите абсолютный путь к папке для новых моделей.")
            return
        self._start_worker("install", selected, base)

    def start_rollback(self):
        spec = self._selected_spec()
        if spec and (self.settings.get("model_previous_paths", {}) or {}).get(spec.key):
            self._start_worker("rollback", [spec])

    @Slot(int, str)
    def on_progress(self, percent, text):
        self.progress_bar.setValue(percent)
        if not self._closing and not (self.worker and self.worker._cancel.is_set()):
            self.status_label.setText(text)

    @Slot(str, object)
    def on_checked(self, key, result):
        self._results[key] = result
        self._errors.pop(key, None)
        self._refresh_row(next(spec for spec in MODEL_CATALOG if spec.key == key))
        self._show_details()

    @Slot(str, str)
    def on_model_ready(self, key, path):
        """The worker emits this only after validation; save all pointers atomically."""
        if self.worker is not None and self.worker._cancel.is_set():
            return
        spec = next(spec for spec in MODEL_CATALOG if spec.key == key)
        selected_path = Path(path)
        model_dir = _directory(spec, path)
        active = dict(self.settings.get("model_paths", {}) or {})
        previous = dict(self.settings.get("model_previous_paths", {}) or {})
        old = self._current_paths.get(key)
        if old and _directory(spec, old) != model_dir:
            previous[key] = str(old)
        active[key] = str(selected_path)
        values = {"model_paths": active, "model_previous_paths": previous}
        if spec.kind == "whisper":
            values["last_model"] = str(model_dir)
        elif spec.kind == "llm":
            values["llm_model_path"] = str(model_dir)
        else:
            values["embed_model_path"] = str(selected_path if selected_path.suffix.lower() == ".xml"
                                             else model_dir / WESPEAKER_XML)
        if self._mode == "install":
            values["model_store_folder"] = self.folder_edit.text().strip()
        try:
            self.settings.update(values)
        except Exception as exc:
            self._save_errors += 1
            self.on_model_error(key, f"Модель проверена, но не удалось сохранить выбор: {exc}. Прежняя версия остаётся выбранной.")
            return
        self._changed = True
        self._current_paths[key] = selected_path
        if key in self._results:
            revision = _stored_revision(spec, selected_path)
            if revision is None and self._mode == "install":
                revision = self._results[key].latest_revision
            self._results[key] = replace(
                self._results[key], current_revision=revision,
                status="up_to_date" if self._mode == "install" else "unknown",
                detail="Выбранная версия проверена на GPU.",
            )
        self._errors.pop(key, None)
        self._refresh_row(spec)
        self.table.item(self._rows[key], 4).setText("Откат выполнен; GPU проверен" if self._mode == "rollback" else "Готова; GPU проверен")
        self._show_details()

    @Slot(str, str)
    def on_model_error(self, key, message):
        self._errors[key] = message
        self._refresh_row(next(spec for spec in MODEL_CATALOG if spec.key == key))
        self._show_details()
        self.status_label.setText(f"Ошибка: {message}")

    @Slot(int, int)
    def on_completed(self, succeeded, failed):
        if self.worker is not None and self.worker._cancel.is_set():
            self.on_cancelled()
            return
        failed += self._save_errors
        succeeded = max(0, succeeded - self._save_errors)
        self.progress_bar.setValue(100)
        action = {"check": "Проверка завершена", "install": "Установка завершена", "rollback": "Откат завершён"}[self._mode]
        self.status_label.setText(
            f"{action}. Успешно: {succeeded}; ошибок: {failed}. "
            + ("Подробности ошибки — в строке модели. Прежние рабочие версии сохранены." if failed else
               "Можно выбрать другую модель или закрыть окно.")
        )

    @Slot()
    def on_cancelled(self):
        self.status_label.setText("Операция отменена. Проверенные модели сохранены, скачанные части можно докачать позднее.")

    @Slot()
    def on_worker_finished(self):
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)
        if self._closing:
            self._finish_close()

    def cancel_operation(self):
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Останавливаю операцию. Ожидаю завершения текущего запроса или проверки GPU…")

    def _finish_close(self):
        super().done(QDialog.Accepted if self._changed else QDialog.Rejected)

    def _request_close(self):
        # Keep the dialog and QThread alive until all queued result slots finish.
        # isRunning() alone misses the interval before the finished slot arrives.
        if self.worker is not None:
            self._closing = True
            self.cancel_operation()
            self.close_button.setEnabled(False)
            return False
        self._finish_close()
        return True

    def reject(self):
        self._request_close()

    def accept(self):
        self._request_close()

    def done(self, _result):
        self._request_close()

    def stop_or_close(self):
        self._request_close()

    def closeEvent(self, event):
        if self._request_close():
            event.accept()
        else:
            event.ignore()
