import os
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QRadioButton, QButtonGroup, QTextEdit, QProgressBar,
    QMessageBox, QGroupBox, QCheckBox, QFormLayout, QDialog,
    QDialogButtonBox, QListWidget, QAbstractItemView, QDoubleSpinBox
)
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from transcriber import Transcriber
from settings import Settings
from model_manager import ModelManager
from model_installer import ModelInstallerDialog
from media_loader import MEDIA_EXTENSIONS, MediaLoader


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(600)

        layout = QVBoxLayout(self)

        # --- Группа моделей Whisper ---
        models_group = QGroupBox("Модели Whisper")
        models_form = QFormLayout(models_group)
        self.models_folder_edit = QLineEdit(self.settings.get("models_folder"))
        btn_browse_models = QPushButton("Обзор...")
        btn_browse_models.clicked.connect(self.browse_models_folder)
        models_folder_layout = QHBoxLayout()
        models_folder_layout.addWidget(self.models_folder_edit)
        models_folder_layout.addWidget(btn_browse_models)
        models_form.addRow("Папка с моделями:", models_folder_layout)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["GPU"])
        current_device = self.settings.get("device", "GPU")
        if current_device == "GPU":
            self.device_combo.setCurrentText(current_device)
        models_form.addRow("Устройство:", self.device_combo)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 60)
        self.chunk_spin.setValue(int(self.settings.get("chunk_minutes", 5)))
        models_form.addRow("Размер чанка (мин):", self.chunk_spin)

        layout.addWidget(models_group)

        # --- Группа сохранения ---
        save_group = QGroupBox("Сохранение")
        save_layout = QVBoxLayout(save_group)

        self.save_mode_group = QButtonGroup(self)
        self.rb_next_to_source = QRadioButton("Рядом с исходным файлом")
        self.rb_custom_folder = QRadioButton("В выбранную папку")
        self.rb_ask = QRadioButton("Спрашивать при сохранении")
        self.save_mode_group.addButton(self.rb_next_to_source)
        self.save_mode_group.addButton(self.rb_custom_folder)
        self.save_mode_group.addButton(self.rb_ask)

        save_mode = self.settings.get("save_mode", "next_to_source")
        if save_mode == "custom_folder":
            self.rb_custom_folder.setChecked(True)
        elif save_mode == "ask":
            self.rb_ask.setChecked(True)
        else:
            self.rb_next_to_source.setChecked(True)

        save_layout.addWidget(self.rb_next_to_source)
        save_layout.addWidget(self.rb_custom_folder)
        save_layout.addWidget(self.rb_ask)

        self.custom_folder_layout = QHBoxLayout()
        self.custom_folder_edit = QLineEdit(self.settings.get("save_folder", ""))
        btn_browse_save = QPushButton("Обзор...")
        btn_browse_save.clicked.connect(self.browse_save_folder)
        self.custom_folder_layout.addWidget(self.custom_folder_edit)
        self.custom_folder_layout.addWidget(btn_browse_save)
        save_layout.addLayout(self.custom_folder_layout)

        self.filename_template_edit = QLineEdit(self.settings.get("filename_template", "{original_name}_transcript.txt"))
        save_layout.addWidget(QLabel("Шаблон имени файла (для отдельных файлов):"))
        save_layout.addWidget(self.filename_template_edit)
        save_layout.addWidget(QLabel("Доступно: {original_name} — имя исходного файла без расширения"))

        self.multi_mode_checkbox = QCheckBox("Сохранять все результаты в один файл (суффикс _multi)")
        self.multi_mode_checkbox.setChecked(bool(self.settings.get("multi_mode", False)))
        save_layout.addWidget(self.multi_mode_checkbox)

        self.keep_extracted_audio_checkbox = QCheckBox(
            "Сохранять извлечённый из видео WAV рядом с исходным файлом"
        )
        self.keep_extracted_audio_checkbox.setChecked(
            bool(self.settings.get("keep_extracted_audio", False))
        )
        save_layout.addWidget(self.keep_extracted_audio_checkbox)

        layout.addWidget(save_group)

        # --- Группа диаризации ---
        diar_group = QGroupBox("Диаризация спикеров (WeSpeaker / OpenVINO GPU)")
        diar_layout = QVBoxLayout(diar_group)

        self.diar_checkbox = QCheckBox("Определять спикеров")
        self.diar_checkbox.setChecked(bool(self.settings.get("enable_diarization", False)))
        diar_layout.addWidget(self.diar_checkbox)

        self.rename_speakers_checkbox = QCheckBox(
            "После распознавания предложить указать имена спикеров"
        )
        self.rename_speakers_checkbox.setChecked(
            bool(self.settings.get("prompt_speaker_names", True))
        )
        diar_layout.addWidget(self.rename_speakers_checkbox)

        diar_form = QFormLayout()
        self.embed_model_edit = QLineEdit(self.settings.get("embed_model_path", ""))
        btn_browse_embed = QPushButton("Обзор...")
        btn_browse_embed.clicked.connect(lambda: self.browse_file(self.embed_model_edit, "Embedding модель (*.xml *.onnx);;Все файлы (*.*)"))
        embed_layout = QHBoxLayout()
        embed_layout.addWidget(self.embed_model_edit)
        embed_layout.addWidget(btn_browse_embed)
        diar_form.addRow("Embedding модель:", embed_layout)

        self.diar_threshold_spin = QDoubleSpinBox()
        self.diar_threshold_spin.setRange(0.30, 1.0)
        self.diar_threshold_spin.setSingleStep(0.01)
        self.diar_threshold_spin.setValue(float(self.settings.get("diar_threshold", 0.74)))
        self.diar_threshold_spin.setToolTip(
            "Чем выше значение, тем охотнее похожие голоса объединяются и тем меньше спикеров."
        )
        diar_form.addRow("Порог объединения (выше = меньше):", self.diar_threshold_spin)

        self.num_speakers_spin = QSpinBox()
        self.num_speakers_spin.setRange(0, 20)
        self.num_speakers_spin.setSpecialValueText("Автоматически")
        self.num_speakers_spin.setValue(int(self.settings.get("num_speakers", 0)))
        self.num_speakers_spin.setToolTip(
            "Если число участников известно, укажите его — это самый надёжный режим."
        )
        diar_form.addRow("Точное количество спикеров:", self.num_speakers_spin)

        self.max_auto_speakers_spin = QSpinBox()
        self.max_auto_speakers_spin.setRange(2, 30)
        self.max_auto_speakers_spin.setValue(
            int(self.settings.get("max_auto_speakers", 8))
        )
        self.max_auto_speakers_spin.setToolTip(
            "Авторежим никогда не создаст больше указанного количества голосов."
        )
        diar_form.addRow("Максимум в авторежиме:", self.max_auto_speakers_spin)

        self.speaker_names_edit = QLineEdit(
            "; ".join(self.settings.get("speaker_names", []))
        )
        self.speaker_names_edit.setPlaceholderText("Например: Андрей; Филипп; Екатерина")
        self.speaker_names_edit.setToolTip(
            "Эти имена появятся в выпадающем списке после распознавания."
        )
        diar_form.addRow("Известные участники:", self.speaker_names_edit)

        diar_layout.addLayout(diar_form)
        layout.addWidget(diar_group)

        # --- Кнопки OK/Cancel ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_models_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбор папки с моделями", self.models_folder_edit.text())
        if folder:
            self.models_folder_edit.setText(folder)

    def browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбор папки для сохранения", self.custom_folder_edit.text())
        if folder:
            self.custom_folder_edit.setText(folder)

    def browse_file(self, line_edit, filter_str):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбор файла модели", "", filter_str)
        if file_path:
            line_edit.setText(file_path)

    def accept(self):
        self.settings.set("models_folder", self.models_folder_edit.text())
        self.settings.set("device", self.device_combo.currentText())
        self.settings.set("chunk_minutes", self.chunk_spin.value())

        if self.rb_custom_folder.isChecked():
            save_mode = "custom_folder"
        elif self.rb_ask.isChecked():
            save_mode = "ask"
        else:
            save_mode = "next_to_source"
        self.settings.set("save_mode", save_mode)
        self.settings.set("save_folder", self.custom_folder_edit.text())
        self.settings.set("filename_template", self.filename_template_edit.text())
        self.settings.set("multi_mode", self.multi_mode_checkbox.isChecked())
        self.settings.set(
            "keep_extracted_audio", self.keep_extracted_audio_checkbox.isChecked()
        )
        self.settings.set("enable_diarization", self.diar_checkbox.isChecked())
        self.settings.set("embed_model_path", self.embed_model_edit.text())
        self.settings.set("diar_threshold", self.diar_threshold_spin.value())
        self.settings.set("num_speakers", self.num_speakers_spin.value())
        self.settings.set("max_auto_speakers", self.max_auto_speakers_spin.value())
        speaker_names = [
            name.strip()
            for name in self.speaker_names_edit.text().replace(",", ";").split(";")
            if name.strip()
        ]
        self.settings.set("speaker_names", list(dict.fromkeys(speaker_names)))
        self.settings.set(
            "prompt_speaker_names", self.rename_speakers_checkbox.isChecked()
        )
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.transcriber = None
        self.current_files = []
        self.last_saved_path = None
        self.pending_ai_path = None
        self.last_media_path = None
        self.run_files = []
        self.completed_transcripts = []
        self._offered_model_install = False

        from app_version import APP_VERSION
        self.setWindowTitle(f"WhisperGUI {APP_VERSION} — аудио и видео")
        self.setMinimumSize(850, 750)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Выбор аудиофайлов ---
        audio_group = QGroupBox("Аудио- и видеофайлы")
        audio_layout = QVBoxLayout(audio_group)

        btn_add_files = QPushButton("Добавить файлы...")
        btn_add_files.clicked.connect(self.add_files)
        btn_add_folder = QPushButton("Добавить папку...")
        btn_add_folder.clicked.connect(self.add_folder)
        btn_remove = QPushButton("Удалить выбранное")
        btn_remove.clicked.connect(self.remove_selected)
        btn_clear = QPushButton("Очистить список")
        btn_clear.clicked.connect(self.clear_list)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_add_files)
        btn_layout.addWidget(btn_add_folder)
        btn_layout.addWidget(btn_remove)
        btn_layout.addWidget(btn_clear)
        btn_layout.addStretch()

        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list_widget.setDragDropMode(QAbstractItemView.DropOnly)
        self.file_list_widget.setAcceptDrops(True)

        audio_layout.addLayout(btn_layout)
        audio_layout.addWidget(self.file_list_widget)
        main_layout.addWidget(audio_group)

        # --- Выбор модели Whisper ---
        model_group = QGroupBox("Модель Whisper")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        btn_refresh_models = QPushButton("Обновить")
        btn_refresh_models.clicked.connect(self.scan_models)
        btn_manual_model = QPushButton("Указать вручную")
        btn_manual_model.clicked.connect(self.manual_select_model)
        btn_install_models = QPushButton("Скачать модели...")
        btn_install_models.clicked.connect(self.open_model_installer)

        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(btn_refresh_models)
        model_row.addWidget(btn_manual_model)
        model_row.addWidget(btn_install_models)

        self.model_path_label = QLabel("Путь: не выбрана")
        model_layout.addLayout(model_row)
        model_layout.addWidget(self.model_path_label)
        main_layout.addWidget(model_group)

        # --- Параметры (устройство, чанк) ---
        params_group = QGroupBox("Параметры")
        params_layout = QHBoxLayout(params_group)

        params_layout.addWidget(QLabel("Устройство:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["GPU"])
        current_device = self.settings.get("device", "GPU")
        if current_device == "GPU":
            self.device_combo.setCurrentText(current_device)
        params_layout.addWidget(self.device_combo)

        params_layout.addSpacing(20)

        params_layout.addWidget(QLabel("Язык:"))
        self.language_combo = QComboBox()
        self.language_combo.addItem("Русский", "ru")
        self.language_combo.addItem("Авто", "auto")
        if self.settings.get("language", "ru") == "auto":
            self.language_combo.setCurrentIndex(1)
        params_layout.addWidget(self.language_combo)

        params_layout.addSpacing(20)

        params_layout.addWidget(QLabel("Размер чанка (мин):"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1, 60)
        self.chunk_spin.setValue(int(self.settings.get("chunk_minutes", 5)))
        params_layout.addWidget(self.chunk_spin)

        params_layout.addStretch()
        main_layout.addWidget(params_group)

        # --- Кнопки управления ---
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Старт")
        self.start_btn.clicked.connect(self.start_transcription)
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_transcription)
        self.open_folder_btn = QPushButton("Открыть папку с результатом")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self.open_result_folder)
        self.settings_btn = QPushButton("Настройки")
        self.settings_btn.clicked.connect(self.open_settings)
        self.ai_btn = QPushButton("Обработать текст ИИ")
        self.ai_btn.clicked.connect(self.open_ai_processing)
        self.rename_speakers_btn = QPushButton("Имена спикеров...")
        self.rename_speakers_btn.clicked.connect(self.rename_last_transcript)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.open_folder_btn)
        control_layout.addWidget(self.rename_speakers_btn)
        control_layout.addWidget(self.ai_btn)
        control_layout.addWidget(self.settings_btn)
        main_layout.addLayout(control_layout)

        # --- Прогресс ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # --- Лог ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text, 1)

        self.setAcceptDrops(True)
        self.scan_models()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._offered_model_install:
            self._offered_model_install = True
            QTimer.singleShot(350, self.offer_missing_model_install)

    def offer_missing_model_install(self):
        if self.model_combo.count() or not self.isVisible():
            return
        answer = QMessageBox.question(
            self,
            "Нужна модель Whisper",
            "Модель распознавания не найдена. Скачать и установить Whisper INT8? "
            "Понадобится около 0,9 ГБ свободного места и доступ к интернету.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self.open_model_installer(initial_selection=("whisper",))

    def open_model_installer(self, initial_selection=("whisper",)):
        # clicked(bool) from QPushButton is not a model-selection argument.
        if isinstance(initial_selection, bool):
            initial_selection = ("whisper",)
        dialog = ModelInstallerDialog(self.settings, self, initial_selection)
        dialog.exec()
        self.scan_models()

    # ---------- Обработчики событий ----------

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выбор аудио- и видеофайлов", "",
            MediaLoader.dialog_filter()
        )
        added = 0
        for f in files:
            if MediaLoader.is_supported(f) and f not in self.current_files:
                self.current_files.append(f)
                self.file_list_widget.addItem(f)
                added += 1
        self.log(f"Добавлено файлов: {added}")

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбор папки с аудио и видео")
        if not folder:
            return
        added = 0
        for root, dirs, files in os.walk(folder):
            dirs[:] = []  # не рекурсивно; чтобы включить рекурсию, удалите эту строку
            for file in files:
                if Path(file).suffix.lower() in MEDIA_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    if full_path not in self.current_files:
                        self.current_files.append(full_path)
                        self.file_list_widget.addItem(full_path)
                        added += 1
        self.log(f"Добавлено файлов из папки: {added}")

    def remove_selected(self):
        for item in self.file_list_widget.selectedItems():
            row = self.file_list_widget.row(item)
            self.file_list_widget.takeItem(row)
            if row < len(self.current_files):
                del self.current_files[row]

    def clear_list(self):
        self.file_list_widget.clear()
        self.current_files.clear()
        self.log("Список файлов очищен.")

    def scan_models(self):
        models_folder = self.settings.get("models_folder", "")
        if not models_folder or not os.path.isdir(models_folder):
            self.model_combo.clear()
            self.model_path_label.setText("Путь: папка моделей не найдена")
            self.log("Папка моделей не найдена. Укажите путь в настройках.")
            return

        models = ModelManager.find_models(models_folder)
        self.model_combo.clear()
        for model_path in models:
            model_name = ModelManager.get_model_name(model_path)
            self.model_combo.addItem(model_name, model_path)
        if models:
            self.model_combo.setCurrentIndex(0)
            last_model = self.settings.get("last_model", "")
            if last_model:
                for idx in range(self.model_combo.count()):
                    if self.model_combo.itemData(idx) == last_model:
                        self.model_combo.setCurrentIndex(idx)
                        break
            self.update_model_path_label()
            self.log(f"Найдено моделей: {len(models)}")
        else:
            self.model_path_label.setText("Путь: модели не найдены")
            self.log("Модели не найдены в указанной папке.")

    def update_model_path_label(self):
        if self.model_combo.count() > 0:
            path = self.model_combo.currentData()
            self.model_path_label.setText(f"Путь: {path}")
        else:
            self.model_path_label.setText("Путь: не выбрана")

    def manual_select_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбор папки с моделью", self.settings.get("models_folder", ""))
        if folder:
            if any(f.endswith('.xml') for f in os.listdir(folder)):
                model_name = os.path.basename(folder.rstrip(os.sep))
                self.model_combo.addItem(model_name, folder)
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
                self.update_model_path_label()
                self.log(f"Модель добавлена вручную: {folder}")
            else:
                QMessageBox.warning(self, "Ошибка", "В выбранной папке не найдены файлы модели (.xml)")

    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            # Настройки сохранены, обновляем поля
            self.device_combo.setCurrentText(self.settings.get("device", "GPU"))
            self.chunk_spin.setValue(int(self.settings.get("chunk_minutes", 5)))
            self.scan_models()

    def open_ai_processing(self, auto_start=False):
        from ai_dialog import AIProcessingDialog

        transcript_path = ""
        if self.last_saved_path and Path(self.last_saved_path).suffix.lower() in (".txt", ".md"):
            transcript_path = self.last_saved_path
        dialog = AIProcessingDialog(
            self.settings, transcript_path, self, auto_start=bool(auto_start)
        )
        dialog.exec()

    def start_transcription(self):
        if len(self.current_files) == 0:
            QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы один аудиофайл.")
            return
        if self.model_combo.count() == 0:
            self.open_model_installer(initial_selection=("whisper",))
            if self.model_combo.count() == 0:
                return

        model_path = self.model_combo.currentData()
        device = self.device_combo.currentText()
        language = self.language_combo.currentData()
        chunk_minutes = self.chunk_spin.value()
        save_mode = self.settings.get("save_mode", "next_to_source")
        save_folder = self.settings.get("save_folder", "")
        filename_template = self.settings.get("filename_template", "{original_name}_transcript.txt")
        multi_mode = self.settings.get("multi_mode", False)
        keep_extracted_audio = self.settings.get("keep_extracted_audio", False)
        enable_diarization = self.settings.get("enable_diarization", False)
        embed_model_path = self.settings.get("embed_model_path", "")
        diar_threshold = self.settings.get("diar_threshold", 0.74)
        num_speakers = self.settings.get("num_speakers", 0)
        max_auto_speakers = self.settings.get("max_auto_speakers", 8)
        self.settings.set("language", language)
        self.settings.set("last_model", model_path)

        self.set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.run_files = self.current_files.copy()
        self.completed_transcripts = []

        self.transcriber = Transcriber(
            file_list=self.current_files.copy(),
            model_path=model_path,
            device=device,
            chunk_minutes=chunk_minutes,
            save_mode=save_mode,
            save_folder=save_folder,
            filename_template=filename_template,
            multi_mode=multi_mode,
            keep_extracted_audio=keep_extracted_audio,
            language=language,
            enable_diarization=enable_diarization,
            embed_model_path=embed_model_path,
            diar_threshold=diar_threshold,
            num_speakers=num_speakers,
            max_auto_speakers=max_auto_speakers
        )
        self.transcriber.log.connect(self.log)
        self.transcriber.progress.connect(self.update_progress)
        self.transcriber.file_done.connect(self.on_file_done)
        self.transcriber.finished_ok.connect(self.on_finished_ok)
        self.transcriber.error.connect(self.on_transcription_error)
        self.transcriber.finished.connect(self.on_thread_finished)

        self.transcriber.start()

    def stop_transcription(self):
        if self.transcriber and self.transcriber.isRunning():
            self.transcriber.cancel()
            self.log("Запрос на остановку отправлен...")

    def set_controls_enabled(self, enabled):
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        self.file_list_widget.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.device_combo.setEnabled(enabled)
        self.language_combo.setEnabled(enabled)
        self.chunk_spin.setEnabled(enabled)
        self.settings_btn.setEnabled(enabled)
        self.ai_btn.setEnabled(enabled)
        self.rename_speakers_btn.setEnabled(enabled)

    @Slot(str)
    def log(self, message):
        self.log_text.append(message)

    @Slot(int, int, str)
    def update_progress(self, current, total, message):
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setFormat(f"{current}/{total}")
        self.log_text.append(message)

    @Slot(str, int, int)
    def on_file_done(self, save_path, idx, total):
        self.last_saved_path = save_path
        if not self.settings.get("multi_mode", False) and 0 <= idx < len(self.run_files):
            source_path = self.run_files[idx]
            self.completed_transcripts.append((save_path, source_path))
            self.last_media_path = source_path
        self.log(f"Сохранён файл: {save_path}")
        self.open_folder_btn.setEnabled(True)

    @Slot(str)
    def on_finished_ok(self, save_path):
        self.last_saved_path = save_path
        self.pending_ai_path = save_path
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("100%")
        self.open_folder_btn.setEnabled(True)
        if self.settings.get("multi_mode", False) and len(self.run_files) == 1:
            self.completed_transcripts = [(save_path, self.run_files[0])]
            self.last_media_path = self.run_files[0]
        self.log("Обработка завершена.")

    @Slot(str)
    def on_transcription_error(self, error_message):
        self.log_text.append(f"\nОШИБКА: {error_message}")
        QMessageBox.critical(self, "Ошибка", error_message)

    @Slot()
    def on_thread_finished(self):
        self.set_controls_enabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("100%")
        if (
            self.settings.get("enable_diarization", False)
            and self.settings.get("prompt_speaker_names", True)
        ):
            if self.settings.get("multi_mode", False) and len(self.run_files) > 1:
                self.log(
                    "Переименование пропущено для общего файла из нескольких записей: "
                    "номера спикеров в разных файлах могут относиться к разным людям."
                )
            else:
                for transcript_path, media_path in self.completed_transcripts:
                    self.show_speaker_rename_dialog(transcript_path, media_path)
        if self.pending_ai_path and self.settings.get("auto_ai_processing", False):
            self.last_saved_path = self.pending_ai_path
            self.pending_ai_path = None
            QTimer.singleShot(200, lambda: self.open_ai_processing(True))
        else:
            self.pending_ai_path = None

    def show_speaker_rename_dialog(self, transcript_path, media_path=""):
        from speaker_renamer import SpeakerRenameDialog, collect_speaker_samples

        try:
            text = Path(transcript_path).read_text(encoding="utf-8")
            if not collect_speaker_samples(text):
                self.log("В расшифровке не найдены метки SPEAKER_* — переименование не требуется.")
                return False
            dialog = SpeakerRenameDialog(
                transcript_path, media_path, self, settings=self.settings
            )
            dialog.exec()
            if dialog.saved:
                self.log(f"Имена спикеров сохранены в файле: {transcript_path}")
                return True
        except Exception as exc:
            QMessageBox.warning(self, "Имена спикеров", str(exc))
        return False

    def rename_last_transcript(self):
        initial_path = self.last_saved_path or ""
        transcript_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите расшифровку со спикерами",
            initial_path,
            "Расшифровки (*.txt *.md);;Все файлы (*.*)",
        )
        if not transcript_path:
            return
        media_path = ""
        if (
            transcript_path == self.last_saved_path
            and self.last_media_path
            and os.path.isfile(self.last_media_path)
        ):
            media_path = self.last_media_path
        else:
            media_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выберите исходное аудио или видео для прослушивания",
                os.path.dirname(transcript_path),
                MediaLoader.dialog_filter(),
            )
        self.last_saved_path = transcript_path
        self.last_media_path = media_path or None
        self.show_speaker_rename_dialog(transcript_path, media_path)

    def open_result_folder(self):
        if self.last_saved_path:
            folder = os.path.dirname(self.last_saved_path)
            if os.path.isdir(folder):
                if sys.platform == "win32":
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    os.system(f'open "{folder}"')
                else:
                    os.system(f'xdg-open "{folder}"')

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        added = 0
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path) and MediaLoader.is_supported(file_path):
                if file_path not in self.current_files:
                    self.current_files.append(file_path)
                    self.file_list_widget.addItem(file_path)
                    added += 1
        self.log(f"Файлы добавлены через drag-and-drop: {added}")

    @staticmethod
    def _format_time(seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
