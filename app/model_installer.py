"""Install official OpenVINO models from Hugging Face with resumable downloads."""

import hashlib
import json
import os
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QVBoxLayout,
)

from settings import MODEL_STORE_FOLDER


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    repo: str
    folder: str
    files: tuple[str, ...]
    approximate_size: int

    @property
    def relative_dir(self):
        category = "turbo-gpu" if self.key == "whisper" else "llm-models"
        return Path(category) / self.folder


WHISPER_FILES = (
    "added_tokens.json", "config.json", "generation_config.json", "merges.txt",
    "normalizer.json", "openvino_config.json", "openvino_decoder_model.bin",
    "openvino_decoder_model.xml", "openvino_detokenizer.bin",
    "openvino_detokenizer.xml", "openvino_encoder_model.bin",
    "openvino_encoder_model.xml", "openvino_tokenizer.bin",
    "openvino_tokenizer.xml", "preprocessor_config.json",
    "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
    "vocab.json",
)
QWEN_FILES = (
    "added_tokens.json", "chat_template.jinja", "config.json",
    "generation_config.json", "merges.txt", "openvino_config.json",
    "openvino_detokenizer.bin", "openvino_detokenizer.xml",
    "openvino_model.bin", "openvino_model.xml", "openvino_tokenizer.bin",
    "openvino_tokenizer.xml", "special_tokens_map.json",
    "tokenizer_config.json", "tokenizer.json", "vocab.json",
)
QWEN14_FILES = tuple(
    name for name in QWEN_FILES if name not in ("chat_template.jinja", "openvino_config.json")
)

MODEL_CATALOG = (
    ModelSpec(
        "whisper", "Whisper large-v3-turbo INT8 — распознавание, ~0,9 ГБ",
        "OpenVINO/whisper-large-v3-turbo-int8-ov",
        "whisper-large-v3-turbo-int8-ov", WHISPER_FILES, 900_000_000,
    ),
    ModelSpec(
        "qwen8", "Qwen3-8B INT4 — обработка текста, ~4,9 ГБ",
        "OpenVINO/Qwen3-8B-int4-ov", "Qwen3-8B-int4-ov",
        QWEN_FILES, 4_900_000_000,
    ),
    ModelSpec(
        "qwen14", "Qwen3-14B INT4 — обработка текста, ~9,8 ГБ",
        "OpenVINO/Qwen3-14B-int4-ov", "Qwen3-14B-int4-ov",
        QWEN14_FILES, 9_800_000_000,
    ),
)


def _request_json(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "WhisperGUI/1.1 model installer"}
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)


def repository_manifest(spec):
    """Pin a repository revision and retrieve sizes and available LFS hashes."""
    encoded_repo = urllib.parse.quote(spec.repo, safe="/")
    info = _request_json(f"https://huggingface.co/api/models/{encoded_repo}")
    revision = info.get("sha")
    if not revision or len(revision) != 40:
        raise RuntimeError(f"Hugging Face не сообщил ревизию модели {spec.repo}.")
    tree = _request_json(
        f"https://huggingface.co/api/models/{encoded_repo}/tree/{revision}"
        "?recursive=true&expand=true"
    )
    by_path = {item.get("path"): item for item in tree if item.get("type") == "file"}
    missing = sorted(set(spec.files) - set(by_path))
    if missing:
        raise RuntimeError(
            f"В репозитории {spec.repo} отсутствуют нужные файлы: {', '.join(missing)}"
        )
    files = []
    for name in spec.files:
        entry = by_path[name]
        size = entry.get("size")
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"Не удалось проверить размер файла {name} в {spec.repo}.")
        lfs = entry.get("lfs") or {}
        digest = lfs.get("oid") if isinstance(lfs, dict) else None
        if digest and digest.startswith("sha256:"):
            digest = digest.split(":", 1)[1]
        if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())):
            digest = None
        git_oid = entry.get("oid") if not digest else None
        if git_oid and (
            len(git_oid) != 40
            or any(c not in "0123456789abcdef" for c in git_oid.lower())
        ):
            git_oid = None
        files.append({"name": name, "size": size, "sha256": digest,
                      "git_sha1": git_oid})
    return revision, files


def _valid_file(path, expected_size, sha256=None, git_sha1=None):
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    if not sha256 and not git_sha1:
        return True
    digest = hashlib.sha256() if sha256 else hashlib.sha1()
    if git_sha1:
        digest.update(f"blob {expected_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower() == (sha256 or git_sha1).lower()


def model_looks_complete(path, spec):
    """Fast startup check; the installer does full remote-manifest validation."""
    folder = Path(path)
    return all((folder / name).is_file() and (folder / name).stat().st_size > 0
               for name in spec.files)


class DownloadCancelled(Exception):
    pass


class ModelDownloadWorker(QThread):
    progress = Signal(int, str)
    model_ready = Signal(str, str)
    succeeded = Signal()
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, base_folder, specs, parent=None):
        super().__init__(parent)
        self.base_folder = Path(base_folder)
        self.specs = list(specs)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise DownloadCancelled()

    def run(self):
        try:
            for spec in self.specs:
                self._check_cancel()
                path = self._install(spec)
                self.model_ready.emit(spec.key, str(path))
            self.succeeded.emit()
        except DownloadCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _download_file(self, spec, revision, entry, stage_path, report):
        name = entry["name"]
        size = entry["size"]
        digest = entry["sha256"]
        git_oid = entry.get("git_sha1")
        final_staged = stage_path / name
        if _valid_file(final_staged, size, digest, git_oid):
            report(0, name)
            return
        partial = stage_path / (name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            partial.unlink()
            offset = 0
        encoded_name = urllib.parse.quote(name, safe="/")
        url = f"https://huggingface.co/{spec.repo}/resolve/{revision}/{encoded_name}"
        headers = {"User-Agent": "WhisperGUI/1.1 model installer"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code != 416 or offset != size:
                raise
            response = None
        if response is not None:
            with response:
                if offset and response.status != 206:
                    report(-offset, name)
                    offset = 0
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Сервер вернул неверный диапазон для {name}.")
                mode = "ab" if offset else "wb"
                with partial.open(mode) as output:
                    while True:
                        self._check_cancel()
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        report(len(block), name)
        self._check_cancel()
        if not _valid_file(partial, size, digest, git_oid):
            actual = partial.stat().st_size if partial.exists() else 0
            if actual == size:
                partial.unlink()
            raise RuntimeError(
                f"Файл {name} не прошёл проверку (получено {actual:,} из {size:,} байт). "
                "Повторите установку для докачки."
            )
        os.replace(partial, final_staged)

    def _install(self, spec):
        self.progress.emit(0, f"Получение списка файлов: {spec.title}")
        revision, files = repository_manifest(spec)
        target = self.base_folder / spec.relative_dir
        stage = self.base_folder / "_downloads" / spec.folder
        stage.mkdir(parents=True, exist_ok=True)
        total = sum(entry["size"] for entry in files)
        completed = 0
        for entry in files:
            if (
                _valid_file(target / entry["name"], entry["size"], entry["sha256"], entry.get("git_sha1"))
                or _valid_file(stage / entry["name"], entry["size"], entry["sha256"], entry.get("git_sha1"))
            ):
                completed += entry["size"]
            else:
                partial = stage / (entry["name"] + ".part")
                if partial.is_file():
                    completed += min(partial.stat().st_size, entry["size"])

        def report(count, filename):
            nonlocal completed
            completed = max(0, completed + count)
            percent = min(100, int(100 * completed / total))
            self.progress.emit(
                percent, f"{spec.folder}: {filename} — {completed / 1e9:.2f} / {total / 1e9:.2f} ГБ"
            )

        for entry in files:
            self._check_cancel()
            if _valid_file(target / entry["name"], entry["size"], entry["sha256"], entry.get("git_sha1")):
                continue
            self._download_file(spec, revision, entry, stage, report)

        self._check_cancel()
        target.mkdir(parents=True, exist_ok=True)
        for entry in files:
            destination = target / entry["name"]
            if not _valid_file(destination, entry["size"], entry["sha256"], entry.get("git_sha1")):
                os.replace(stage / entry["name"], destination)
        marker = target / "_whispergui_manifest.json"
        marker_tmp = target / "_whispergui_manifest.json.tmp"
        marker_tmp.write_text(
            json.dumps({"repository": spec.repo, "revision": revision, "files": files}, indent=2),
            encoding="utf-8",
        )
        os.replace(marker_tmp, marker)
        self.progress.emit(100, f"Установлена модель: {spec.folder}")
        return target


class ModelInstallerDialog(QDialog):
    def __init__(self, settings, parent=None, initial_selection=("whisper",)):
        super().__init__(parent)
        self.settings = settings
        self.worker = None
        self._closing = False
        self._successful = False
        self.setWindowTitle("Установка моделей")
        self.setMinimumWidth(690)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Модели будут скачаны из официальных репозиториев OpenVINO на Hugging Face. "
            "Доступ к интернету нужен только при установке; затем приложение работает локально."
        ))
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(
            settings.get("model_store_folder", str(MODEL_STORE_FOLDER))
        )
        choose_button = QPushButton("Выбрать папку...")
        choose_button.clicked.connect(self.choose_folder)
        folder_row.addWidget(QLabel("Хранить модели в:"))
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(choose_button)
        layout.addLayout(folder_row)
        self.checkboxes = {}
        for spec in MODEL_CATALOG:
            existing = self._existing_model(spec)
            checkbox = QCheckBox(spec.title + (" (уже найдена)" if existing else ""))
            checkbox.setChecked(spec.key in initial_selection)
            checkbox.setToolTip(f"https://huggingface.co/{spec.repo}")
            self.checkboxes[spec.key] = checkbox
            layout.addWidget(checkbox)
        embed = Path(settings.get("embed_model_path", ""))
        if embed.is_file() and embed.with_suffix(".bin").is_file():
            embed_text = "Модель голосов WeSpeaker включена в комплект и готова к работе."
        else:
            embed_text = "Модель голосов WeSpeaker не найдена. Укажите её XML в настройках."
        layout.addWidget(QLabel(embed_text))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Выберите модели и нажмите «Скачать и установить».")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        buttons = QHBoxLayout()
        self.install_button = QPushButton("Скачать и установить")
        self.install_button.clicked.connect(self.start_download)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.stop_or_close)
        buttons.addStretch()
        buttons.addWidget(self.install_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def _existing_model(self, spec):
        candidates = [Path(self.settings.get("model_store_folder") or MODEL_STORE_FOLDER) / spec.relative_dir]
        if spec.key == "whisper":
            candidates.append(Path(self.settings.get("models_folder", "")) / spec.folder)
        else:
            candidates.append(Path(self.settings.get("llm_models_folder", "")) / spec.folder)
        return any(model_looks_complete(folder, spec) for folder in candidates)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Папка для моделей", self.folder_edit.text()
        )
        if folder:
            self.folder_edit.setText(folder)

    def start_download(self):
        selected = [spec for spec in MODEL_CATALOG if self.checkboxes[spec.key].isChecked()]
        if not selected:
            QMessageBox.information(self, "Модели", "Выберите хотя бы одну модель для установки.")
            return
        base = Path(self.folder_edit.text().strip()).expanduser()
        if not base.is_absolute():
            QMessageBox.warning(self, "Путь", "Выберите абсолютный путь к папке моделей.")
            return
        try:
            base.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(base).free
        except OSError as exc:
            QMessageBox.critical(self, "Папка моделей", str(exc))
            return
        needed = sum(spec.approximate_size for spec in selected)
        if free < needed + 500_000_000:
            QMessageBox.warning(
                self, "Недостаточно места",
                f"Для выбранных моделей потребуется около {needed / 1e9:.1f} ГБ "
                "и дополнительное место для установки.",
            )
            return
        self.settings.set("model_store_folder", str(base))
        self.install_button.setEnabled(False)
        self.folder_edit.setEnabled(False)
        for checkbox in self.checkboxes.values():
            checkbox.setEnabled(False)
        self.close_button.setText("Остановить")
        self.status_label.setText("Подключение к Hugging Face...")
        self.worker = ModelDownloadWorker(base, selected, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.model_ready.connect(self.on_model_ready)
        self.worker.succeeded.connect(self.on_success)
        self.worker.failed.connect(self.on_error)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_progress(self, percent, status):
        self.progress_bar.setValue(percent)
        self.status_label.setText(status)

    def on_model_ready(self, key, path):
        model_dir = Path(path)
        if key == "whisper":
            self.settings.set("models_folder", str(model_dir.parent))
            self.settings.set("last_model", str(model_dir))
        else:
            self.settings.set("llm_models_folder", str(model_dir.parent))
            self.settings.set("llm_model_path", str(model_dir))
        self.status_label.setText(f"Установлена модель: {model_dir.name}")

    def on_success(self):
        self.progress_bar.setValue(100)
        self.status_label.setText("Все выбранные модели установлены и проверены.")
        self._successful = True

    def on_error(self, message):
        self.status_label.setText(
            f"Ошибка: {message}\nУже скачанные части сохранены; повторный запуск продолжит загрузку."
        )
        QMessageBox.warning(self, "Не удалось установить модель", self.status_label.text())

    def on_cancelled(self):
        self.status_label.setText("Загрузка остановлена. Скачанные части сохранены.")

    def on_worker_finished(self):
        self.worker = None
        if self._successful:
            QMessageBox.information(self, "Готово", "Модели установлены. Можно начать работу.")
            self.accept()
            return
        self.install_button.setEnabled(True)
        self.folder_edit.setEnabled(True)
        for checkbox in self.checkboxes.values():
            checkbox.setEnabled(True)
        self.close_button.setText("Закрыть")
        if self._closing:
            self.reject()

    def stop_or_close(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.close_button.setEnabled(False)
            self.status_label.setText("Останавливаю загрузку после текущего блока...")
            self._closing = True
        else:
            self.reject()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.stop_or_close()
            event.ignore()
            return
        super().closeEvent(event)
