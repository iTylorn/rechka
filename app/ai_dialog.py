import os
from pathlib import Path

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from llm_processor import LLMProcessingWorker
from prompt_library import PromptLibrary
from model_installer import model_path_for


class AIProcessingDialog(QDialog):
    def __init__(self, settings, transcript_path="", parent=None, auto_start=False):
        super().__init__(parent)
        self.settings = settings
        self.library = PromptLibrary(settings.config_dir)
        self.worker = None
        self._closing = False
        self._close_result = QDialog.Rejected
        self._model_installer = None
        self._starting = False
        self._auto_start_pending = bool(auto_start)
        self._auto_start_timer = QTimer(self)
        self._auto_start_timer.setSingleShot(True)
        self._auto_start_timer.timeout.connect(self._run_auto_start)
        self.current_prompt = None
        self.last_output_path = ""
        self.setWindowTitle("Обработка расшифровки локальной нейросетью")
        self.resize(1050, 820)

        layout = QVBoxLayout(self)
        source_form = QFormLayout()

        models_root = Path(settings.get("llm_models_folder", ""))
        model_8b = str(model_path_for(settings, "qwen8") or models_root / "Qwen3-8B-int4-ov")
        model_14b = str(model_path_for(settings, "qwen14") or models_root / "Qwen3-14B-int4-ov")
        current_model = settings.get("llm_model_path", model_8b)
        self.model_preset_combo = QComboBox()
        self.model_preset_combo.addItem("Быстрая — Qwen3-8B INT4", model_8b)
        self.model_preset_combo.addItem("Качественная — Qwen3-14B INT4", model_14b)
        self.model_preset_combo.setToolTip(
            "8B работает быстрее; 14B точнее извлекает поручения и факты. Обе модели используют GPU."
        )
        preset_index = self.model_preset_combo.findData(current_model)
        if preset_index >= 0:
            self.model_preset_combo.setCurrentIndex(preset_index)
        self.model_edit = QLineEdit(current_model)
        self.model_preset_combo.currentIndexChanged.connect(
            lambda: self.model_edit.setText(self.model_preset_combo.currentData())
        )
        model_button = QPushButton("Обзор...")
        model_button.clicked.connect(self.browse_model)
        self.install_model_button = QPushButton("Модели и обновления...")
        self.install_model_button.clicked.connect(self.install_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_preset_combo)
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(model_button)
        model_row.addWidget(self.install_model_button)
        source_form.addRow("Локальная LLM:", model_row)

        self.transcript_edit = QLineEdit(transcript_path or settings.get("last_transcript_path", ""))
        transcript_button = QPushButton("Открыть TXT...")
        transcript_button.clicked.connect(self.browse_transcript)
        transcript_row = QHBoxLayout()
        transcript_row.addWidget(self.transcript_edit, 1)
        transcript_row.addWidget(transcript_button)
        source_form.addRow("Расшифровка:", transcript_row)
        layout.addLayout(source_form)

        prompt_row = QHBoxLayout()
        self.prompt_combo = QComboBox()
        self.prompt_combo.currentIndexChanged.connect(self.load_selected_prompt)
        prompt_row.addWidget(QLabel("Промпт:"))
        prompt_row.addWidget(self.prompt_combo, 1)
        self.new_button = QPushButton("Новый")
        self.copy_button = QPushButton("Копировать")
        self.save_prompt_button = QPushButton("Сохранить")
        self.delete_prompt_button = QPushButton("Удалить")
        self.new_button.clicked.connect(self.new_prompt)
        self.copy_button.clicked.connect(self.copy_prompt)
        self.save_prompt_button.clicked.connect(self.save_prompt)
        self.delete_prompt_button.clicked.connect(self.delete_prompt)
        for button in (self.new_button, self.copy_button, self.save_prompt_button, self.delete_prompt_button):
            prompt_row.addWidget(button)
        layout.addLayout(prompt_row)

        splitter = QSplitter()
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        prompt_form = QFormLayout()
        self.name_edit = QLineEdit()
        self.category_edit = QLineEdit()
        self.description_edit = QLineEdit()
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(200, 5000)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 1.5)
        self.temperature_spin.setSingleStep(0.1)
        prompt_form.addRow("Название:", self.name_edit)
        prompt_form.addRow("Категория:", self.category_edit)
        prompt_form.addRow("Описание:", self.description_edit)
        prompt_form.addRow("Макс. новых токенов:", self.tokens_spin)
        prompt_form.addRow("Температура:", self.temperature_spin)
        editor_layout.addLayout(prompt_form)
        editor_layout.addWidget(QLabel("Системная инструкция:"))
        self.system_edit = QTextEdit()
        editor_layout.addWidget(self.system_edit, 1)
        editor_layout.addWidget(QLabel("Пользовательский промпт (вставка текста: {transcript}):"))
        self.user_edit = QTextEdit()
        editor_layout.addWidget(self.user_edit, 2)
        splitter.addWidget(editor)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.addWidget(QLabel("Результат / журнал генерации:"))
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        preview_layout.addWidget(self.output_edit)
        splitter.addWidget(preview)
        splitter.setSizes([510, 510])
        layout.addWidget(splitter, 1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Формат:"))
        self.format_combo = QComboBox()
        self.format_combo.addItem("Markdown (*.md)", "md")
        self.format_combo.addItem("Текст (*.txt)", "txt")
        self.format_combo.addItem("HTML (*.html)", "html")
        current_format = settings.get("ai_output_format", "md")
        index = self.format_combo.findData(current_format)
        if index >= 0:
            self.format_combo.setCurrentIndex(index)
        controls.addWidget(self.format_combo)
        self.auto_checkbox = QCheckBox("Автоматически после распознавания")
        self.auto_checkbox.setChecked(bool(settings.get("auto_ai_processing", False)))
        self.auto_checkbox.toggled.connect(
            lambda checked: self.settings.set("auto_ai_processing", checked)
        )
        controls.addWidget(self.auto_checkbox)
        controls.addStretch()
        self.start_button = QPushButton("Обработать")
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setEnabled(False)
        self.open_folder_button = QPushButton("Открыть результат")
        self.open_folder_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        self.open_folder_button.clicked.connect(self.open_result_folder)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.open_folder_button)
        layout.addLayout(controls)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.refresh_prompts(settings.get("last_prompt_id", "meeting_protocol"))
        if auto_start:
            self._auto_start_timer.start(200)

    def _run_auto_start(self):
        if self._closing or not self.isVisible() or not self._auto_start_pending:
            return
        if self.worker is not None:
            self._auto_start_pending = False
            return
        modal = QApplication.activeModalWidget()
        if self._model_installer is not None or self._starting or (modal is not None and modal is not self):
            self._auto_start_timer.start(200)
            return
        self._auto_start_pending = False
        self.start_processing()

    def refresh_prompts(self, selected_id=None):
        self.prompt_combo.blockSignals(True)
        self.prompt_combo.clear()
        for prompt in self.library.all():
            label = f"{prompt.get('category', 'Общее')} — {prompt.get('name', 'Без названия')}"
            self.prompt_combo.addItem(label, prompt)
        self.prompt_combo.blockSignals(False)
        if selected_id:
            for index in range(self.prompt_combo.count()):
                if self.prompt_combo.itemData(index).get("id") == selected_id:
                    self.prompt_combo.setCurrentIndex(index)
                    break
        self.load_selected_prompt()

    @Slot()
    def load_selected_prompt(self):
        prompt = self.prompt_combo.currentData()
        if not prompt:
            return
        self.current_prompt = dict(prompt)
        self.name_edit.setText(prompt.get("name", ""))
        self.category_edit.setText(prompt.get("category", ""))
        self.description_edit.setText(prompt.get("description", ""))
        self.system_edit.setPlainText(prompt.get("system_prompt", ""))
        self.user_edit.setPlainText(prompt.get("user_prompt", ""))
        self.tokens_spin.setValue(int(prompt.get("max_new_tokens", 1400)))
        self.temperature_spin.setValue(float(prompt.get("temperature", 0.1)))
        builtin = bool(prompt.get("builtin", False))
        self.delete_prompt_button.setEnabled(not builtin)
        self.save_prompt_button.setText("Сохранить копию" if builtin else "Сохранить")

    def prompt_from_editor(self):
        prompt = dict(self.current_prompt or {})
        prompt.update({
            "name": self.name_edit.text().strip() or "Новый промпт",
            "category": self.category_edit.text().strip() or "Пользовательские",
            "description": self.description_edit.text().strip(),
            "system_prompt": self.system_edit.toPlainText().strip(),
            "user_prompt": self.user_edit.toPlainText().strip(),
            "max_new_tokens": self.tokens_spin.value(),
            "temperature": self.temperature_spin.value(),
        })
        return prompt

    def new_prompt(self):
        self.current_prompt = {"id": "", "builtin": False}
        self.name_edit.setText("Новый промпт")
        self.category_edit.setText("Пользовательские")
        self.description_edit.clear()
        self.system_edit.setPlainText("Работай только по предоставленной расшифровке. Не выдумывай факты.")
        self.user_edit.setPlainText("Обработай расшифровку по следующей задаче:\n\n{transcript}")
        self.tokens_spin.setValue(1200)
        self.temperature_spin.setValue(0.1)
        self.delete_prompt_button.setEnabled(False)

    def copy_prompt(self):
        self.current_prompt = self.library.duplicate(self.prompt_from_editor())
        self.name_edit.setText(self.current_prompt["name"])
        self.delete_prompt_button.setEnabled(False)

    def save_prompt(self):
        prompt = self.prompt_from_editor()
        if prompt.get("builtin"):
            prompt = self.library.duplicate(prompt)
        saved = self.library.upsert_user_prompt(prompt)
        self.current_prompt = saved
        self.refresh_prompts(saved["id"])

    def delete_prompt(self):
        if not self.current_prompt or self.current_prompt.get("builtin"):
            return
        if self.library.delete(self.current_prompt.get("id")):
            self.refresh_prompts("meeting_protocol")

    def browse_model(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбор папки OpenVINO LLM", self.model_edit.text())
        if folder:
            self.model_edit.setText(folder)

    def install_model(self):
        from model_installer import ModelInstallerDialog

        if self._closing or self.worker is not None or self._model_installer is not None:
            return False
        model_key = "qwen14" if self.model_preset_combo.currentIndex() == 1 else "qwen8"
        dialog = ModelInstallerDialog(self.settings, self, initial_selection=(model_key,))
        self._model_installer = dialog
        try:
            dialog.exec()
        finally:
            self._model_installer = None
            dialog.deleteLater()
        if self._closing:
            self._request_close(self._close_result)
            return False
        models_root = Path(self.settings.get("llm_models_folder", ""))
        self.model_preset_combo.setItemData(0, str(model_path_for(self.settings, "qwen8") or models_root / "Qwen3-8B-int4-ov"))
        self.model_preset_combo.setItemData(1, str(model_path_for(self.settings, "qwen14") or models_root / "Qwen3-14B-int4-ov"))
        installed = self.settings.get("llm_model_path", "")
        index = self.model_preset_combo.findData(installed)
        if index >= 0:
            self.model_preset_combo.setCurrentIndex(index)
        if all((Path(installed) / name).is_file() for name in (
            "openvino_model.xml", "openvino_model.bin", "tokenizer.json"
        )):
            self.model_edit.setText(installed)
            return True
        return False

    def browse_transcript(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбор расшифровки", "", "Текст (*.txt *.md);;Все файлы (*.*)")
        if path:
            self.transcript_edit.setText(path)

    def start_processing(self):
        if (self._closing or self.worker is not None
                or self._model_installer is not None or self._starting):
            return
        self._auto_start_pending = False
        self._auto_start_timer.stop()
        self._starting = True
        try:
            self._start_processing()
        finally:
            self._starting = False

    def _start_processing(self):
        model_path = self.model_edit.text().strip()
        transcript_path = self.transcript_edit.text().strip()
        model_dir = Path(model_path)
        if not all((model_dir / name).is_file() for name in (
            "openvino_model.xml", "openvino_model.bin", "tokenizer.json"
        )):
            if not self.install_model():
                return
            model_path = self.model_edit.text().strip()
        if not os.path.isfile(transcript_path):
            QMessageBox.warning(self, "Расшифровка не найдена", "Выберите существующий TXT или MD файл.")
            return
        prompt = self.prompt_from_editor()
        if not prompt.get("user_prompt"):
            QMessageBox.warning(self, "Пустой промпт", "Введите пользовательский промпт.")
            return
        self.settings.set("llm_model_path", model_path)
        self.settings.set("last_transcript_path", transcript_path)
        self.settings.set("last_prompt_id", prompt.get("id", ""))
        self.settings.set("ai_output_format", self.format_combo.currentData())
        self.output_edit.clear()
        self.progress_bar.setValue(0)
        self.set_running(True)
        self.worker = LLMProcessingWorker(
            model_path,
            transcript_path,
            prompt,
            self.format_combo.currentData(),
            self,
        )
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.update_progress)
        self.worker.partial.connect(self.output_edit.setPlainText)
        self.worker.completed.connect(self.processing_completed)
        self.worker.error.connect(self.processing_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    @Slot()
    def on_worker_finished(self):
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        if self._closing:
            self._request_close(self._close_result)
        else:
            self.set_running(False)

    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.append_log("Остановка после текущего шага...")

    def set_running(self, running):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.prompt_combo.setEnabled(not running)
        self.install_model_button.setEnabled(not running)

    @Slot(str)
    def append_log(self, message):
        self.output_edit.append(f"\n[Система] {message}")

    @Slot(int, str)
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% — {message}")

    @Slot(str, str)
    def processing_completed(self, path, result):
        self.last_output_path = path
        self.output_edit.setPlainText(result)
        self.open_folder_button.setEnabled(True)
        self.progress_bar.setValue(100)
        if not self._closing:
            QMessageBox.information(self, "Готово", f"Результат сохранён:\n{path}")

    @Slot(str)
    def processing_error(self, message):
        self.output_edit.append(f"\nОШИБКА: {message}")
        if not self._closing:
            QMessageBox.critical(self, "Ошибка LLM", message)

    def open_result_folder(self):
        if self.last_output_path and os.path.isfile(self.last_output_path):
            os.startfile(os.path.dirname(self.last_output_path))

    def _request_close(self, result=QDialog.Rejected):
        first_request = not self._closing
        self._closing = True
        if first_request:
            self._close_result = result
        self._auto_start_pending = False
        self._auto_start_timer.stop()
        if self.worker is not None:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            if first_request:
                self.append_log("Останавливаю обработку перед закрытием окна…")
            return False
        if self._model_installer is not None:
            self._model_installer.close()
            return False
        super().done(self._close_result)
        return True

    def reject(self):
        self._request_close(QDialog.Rejected)

    def accept(self):
        self._request_close(QDialog.Accepted)

    def done(self, result):
        self._request_close(result)

    def closeEvent(self, event):
        if self._request_close():
            event.accept()
        else:
            event.ignore()
