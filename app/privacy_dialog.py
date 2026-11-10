"""Review and save a separate anonymized audio transcript."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from model_installer import model_path_for
from privacy_worker import PrivacyWorker
from privacy_session import PrivacySession, SourceSnapshot
from transcript_privacy import (
    Finding, approved_mapping, ensure_removed, key, matches, replace_text, save_result,
)


class PrivacyDialog(QDialog):
    def __init__(self, settings, transcript_path='', parent=None, transcripts=(),
                 continue_mode=False, source_text=None, known_aliases=(), seed_findings=()):
        super().__init__(parent)
        self.settings = settings
        self.worker = None
        self._model_installer = None
        self._closing = False
        self._close_result = QDialog.Rejected
        self._discard_result = False
        self._verify = False
        self._ready = False
        self.source = ''
        self.source_path = ''
        self.findings = []
        self.session = PrivacySession(known_aliases=known_aliases)
        self._memory_text = source_text
        self._seed_findings = list(seed_findings)
        self._preview_outputs = []
        self.approved_texts = []
        self.approved_aliases = []
        self.approved_findings = []
        self.continue_mode = continue_mode
        self.setWindowTitle('Обезличивание текстов и протоколов')
        self.resize(1160, 850)
        self.setMinimumSize(860, 650)
        layout = QVBoxLayout(self)
        intro = QLabel('1. Добавьте файлы → 2. Проверьте общие замены и вхождения → 3. Проверьте результат')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.inputs = QWidget()
        inputs = QVBoxLayout(self.inputs)
        inputs.setContentsMargins(0, 0, 0, 0)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel('Расшифровка:'))
        self.source_combo = QComboBox()
        self.source_combo.setMinimumContentsLength(30)
        self.source_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        paths = [transcript_path] + list(transcripts)
        if not transcript_path:
            paths.append(settings.get('last_transcript_path', ''))
        for path in dict.fromkeys(paths):
            if path and Path(path).suffix.lower() in ('.txt', '.md'):
                path = str(Path(path).resolve())
                if self.source_combo.findData(path) >= 0:
                    continue
                self.source_combo.addItem(str(path), str(path))
        self.browse_source_button = QPushButton('Добавить TXT / MD…')
        self.browse_source_button.clicked.connect(self.browse_source)
        source_row.addWidget(self.source_combo, 1)
        source_row.addWidget(self.browse_source_button)
        self.remove_source_button = QPushButton('Убрать файл')
        self.remove_source_button.clicked.connect(self.remove_source)
        source_row.addWidget(self.remove_source_button)
        if source_text is not None:
            self.source_combo.clear()
            self.source_combo.addItem('Черновик протокола', '')
            self.browse_source_button.setEnabled(False)
            self.remove_source_button.setEnabled(False)
        inputs.addLayout(source_row)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('Qwen · Intel GPU:'))
        self.model_combo = QComboBox()
        models_root = Path(settings.get('llm_models_folder', ''))
        for model_key, size in (('qwen8', '8B'), ('qwen14', '14B')):
            path = model_path_for(settings, model_key) or models_root / f'Qwen3-{size}-int4-ov'
            self.model_combo.addItem('Qwen3-' + size, str(path))
        current = settings.get('llm_model_path', '') or self.model_combo.itemData(1)
        index = self.model_combo.findData(current)
        self.model_combo.setCurrentIndex(max(index, 0))
        self.model_edit = QLineEdit(current)
        self.model_edit.setToolTip('Папка локальной модели Qwen OpenVINO. Для более точного поиска рекомендуется 14B.')
        self.model_combo.currentIndexChanged.connect(
            lambda: self.model_edit.setText(self.model_combo.currentData()))
        model_browse = QPushButton('Папка…')
        model_browse.clicked.connect(self.browse_model)
        self.install_button = QPushButton('Модели…')
        self.install_button.clicked.connect(self.install_model)
        for widget in (self.model_combo, self.model_edit, model_browse, self.install_button):
            model_row.addWidget(widget, 1 if widget is self.model_edit else 0)
        inputs.addLayout(model_row)
        layout.addWidget(self.inputs)

        splitter = QSplitter(Qt.Vertical)
        review = QWidget()
        review_layout = QVBoxLayout(review)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_label = QLabel('Общий словарь выбранной пачки. «Вхождения…» показывает контекст и позволяет '
                              'задать разные решения для одинаковых имён. Снятая галочка разрешает оставить значение.')
        review_label.setWordWrap(True)
        review_layout.addWidget(review_label)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Заменить', 'Тип', 'Исходное значение', 'Замена', 'Вхождений'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setWordWrap(False)
        for column in (0, 1, 4):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in (2, 3):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        self.table.itemChanged.connect(self.item_changed)
        review_layout.addWidget(self.table, 1)
        self.edit_controls = QWidget()
        edits = QGridLayout(self.edit_controls)
        edits.setContentsMargins(0, 0, 0, 0)
        for index, (label, callback) in enumerate((('Добавить строку', self.add_row),
                                ('Связать выделенные строки', self.link_rows),
                                ('Разъединить', self.unlink_rows),
                                ('Вхождения…', self.review_occurrences),
                                ('Скрыть выделенный фрагмент', self.hide_selection),
                                ('Выбрать всё', lambda: self.select_all(True)),
                                ('Снять все галочки', lambda: self.select_all(False)))):
            button = QPushButton(label)
            button.clicked.connect(callback)
            edits.addWidget(button, index // 4, index % 4)
        review_layout.addWidget(self.edit_controls)
        splitter.addWidget(review)
        self.tabs = QTabWidget()
        self.original_view = QTextEdit()
        self.original_view.setReadOnly(True)
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)
        self.tabs.addTab(self.original_view, 'Исходная расшифровка')
        self.tabs.addTab(self.result_view, 'Результат замен')
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 260])
        layout.addWidget(splitter, 1)
        note = QLabel('Поиск и замены выполняются локально. Перед передачей прочитайте весь результат: '
                      'косвенные признаки и редкие технические сведения могут остаться. '
                      'Словарь замен хранится только до закрытия окна.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.review_checkbox = QCheckBox('Я просмотрел(а) результат и разрешённые исключения перед сохранением / продолжением')
        self.review_checkbox.toggled.connect(self.update_controls)
        layout.addWidget(self.review_checkbox)
        buttons = QGridLayout()
        self.find_button = QPushButton('1. Найти сведения')
        self.verify_button = QPushButton('2. Применить и проверить повторно')
        self.save_button = QPushButton('Сохранить проверенные файлы…')
        self.save_button.setVisible(not continue_mode)
        self.stop_button = QPushButton('Остановить')
        self.use_button = QPushButton('Подтвердить и продолжить')
        self.use_button.setVisible(continue_mode)
        self.use_button.clicked.connect(self.use_result)
        self.find_button.clicked.connect(lambda: self.start_scan(False))
        self.verify_button.clicked.connect(lambda: self.start_scan(True))
        self.save_button.clicked.connect(self.save)
        self.stop_button.clicked.connect(self.stop)
        for index, button in enumerate((self.find_button, self.verify_button, self.save_button, self.use_button, self.stop_button)):
            buttons.addWidget(button, index // 3, index % 3)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.source_combo.currentIndexChanged.connect(self.load_source)
        self.model_edit.textChanged.connect(self.invalidate)
        self.load_source()

    def load_source(self):
        if self.worker is not None or self._closing:
            return
        paths = [self.source_combo.itemData(i) for i in range(self.source_combo.count())
                 if self.source_combo.itemData(i)]
        self.source_path = self.source_combo.currentData() or ''
        if self.session.sources and paths == [s.path for s in self.session.sources] and self._memory_text is None:
            self.show_document()
            return
        self.source = ''
        self.findings = []
        message = 'Добавьте TXT или Markdown в UTF-8.'
        try:
            sources = ([SourceSnapshot('Черновик протокола', self._memory_text)]
                       if self._memory_text is not None else [SourceSnapshot.read(p) for p in paths])
            self.session = PrivacySession(sources, self.session.known_aliases)
            self.source = self.session.text
            from transcript_privacy import merge_findings
            self.findings = merge_findings([], self._seed_findings, self.source)
            message = f'Файлов: {len(sources)}. Найдите сведения и проверьте общие замены.'
        except ValueError as exc:
            self.session = PrivacySession()
            message = str(exc)
        self.show_document()
        self.render_table()
        self.invalidate()
        self.status.setText(message)

    def browse_source(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Добавить тексты', '', 'Текст (*.txt *.md)')
        self.source_combo.blockSignals(True)
        for path in paths:
            path = str(Path(path).resolve())
            if self.source_combo.findData(path) < 0:
                self.source_combo.addItem(path, path)
        self.source_combo.blockSignals(False)
        if paths:
            self.load_source()

    def remove_source(self):
        self.source_combo.blockSignals(True)
        self.source_combo.removeItem(self.source_combo.currentIndex())
        self.source_combo.blockSignals(False)
        self.load_source()

    def show_document(self):
        index = max(0, self.source_combo.currentIndex())
        if index < len(self.session.sources):
            self.original_view.setPlainText(self.session.sources[index].text)
            self.result_view.setPlainText(self._preview_outputs[index] if index < len(self._preview_outputs) else '')
        else:
            self.original_view.clear()

    def hide_selection(self):
        value = self.original_view.textCursor().selectedText().replace('\u2029', '\n').strip()
        if value:
            self.findings.append(Finding(value, replacement='[Фрагмент скрыт]'))
            self.render_table()
            self.invalidate()

    def unlink_rows(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        groups = {self.findings[row].entity for row in rows}
        for group in groups:
            remaining = [f for i, f in enumerate(self.findings) if i not in rows and f.entity == group]
            if remaining:
                canonical = key(remaining[0].original)
                for finding in remaining:
                    finding.entity = canonical
        for row in rows:
            finding = self.findings[row]
            finding.entity = key(finding.original)
            finding.replacement = f'[Объект {len(self.findings) + row + 1}]'
        self.render_table()
        self.invalidate()

    def review_occurrences(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.findings):
            return
        finding = self.findings[row]
        occurrences = self.session.occurrences(finding.original)
        dialog = QDialog(self)
        dialog.setWindowTitle('Вхождения: проверьте, кто или что упомянуто')
        dialog.resize(1000, 500)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('Галочка — заменить это вхождение. Можно ввести разные метки для разных людей.'))
        table = QTableWidget(len(occurrences), 4)
        table.setHorizontalHeaderLabels(['Заменить', 'Файл', 'Контекст', 'Замена'])
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.setWordWrap(True)
        for index, occurrence in enumerate(occurrences):
            span = (occurrence.start, occurrence.end)
            alias = self.session.overrides.get(span, finding.replacement if finding.selected else None)
            check = QTableWidgetItem()
            check.setCheckState(Qt.Checked if alias is not None else Qt.Unchecked)
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            table.setItem(index, 0, check)
            for column, value in ((1, self.session.sources[occurrence.source_index].label),
                                  (2, occurrence.context), (3, alias or finding.replacement)):
                item = QTableWidgetItem(value)
                if column != 3:
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(index, column, item)
        layout.addWidget(table)
        button = QPushButton('Применить решения по вхождениям')
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)
        if dialog.exec() == QDialog.Accepted:
            for index, occurrence in enumerate(occurrences):
                self.session.overrides[(occurrence.start, occurrence.end)] = (
                    table.item(index, 3).text().strip() if table.item(index, 0).checkState() == Qt.Checked else None)
            self.invalidate()
        dialog.deleteLater()

    def browse_model(self):
        path = QFileDialog.getExistingDirectory(self, 'Папка Qwen OpenVINO', self.model_edit.text())
        if path:
            self.model_edit.setText(path)

    def install_model(self):
        if self.worker is not None or self._closing or self._model_installer is not None:
            return
        from model_installer import ModelInstallerDialog
        model_key = 'qwen14' if self.model_combo.currentIndex() == 1 else 'qwen8'
        dialog = ModelInstallerDialog(self.settings, self, initial_selection=(model_key,))
        self._model_installer = dialog
        try:
            dialog.exec()
        finally:
            self._model_installer = None
            dialog.deleteLater()
        if self._closing:
            self._request_close(self._close_result)
            return
        for index, key_name in enumerate(('qwen8', 'qwen14')):
            path = model_path_for(self.settings, key_name)
            if path:
                self.model_combo.setItemData(index, str(path))
        self.model_edit.setText(self.model_combo.currentData())

    def render_table(self):
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.findings))
            for row, finding in enumerate(self.findings):
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                check.setCheckState(Qt.Checked if finding.selected else Qt.Unchecked)
                self.table.setItem(row, 0, check)
                for col, value in ((1, finding.category), (2, finding.original),
                                   (3, finding.replacement), (4, str(len(self.session.occurrences(finding.original))))):
                    item = QTableWidgetItem(value)
                    if col in (1, 4):
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setToolTip(value if col != 3 else value + '\nИзменение применяется ко всем связанным строкам.')
                    self.table.setItem(row, col, item)
                decisions = [o for o in self.session.occurrences(finding.original)
                             if (o.start, o.end) in self.session.overrides]
                self.table.item(row, 4).setToolTip(
                    f'Индивидуальных решений: {len(decisions)}. Нажмите «Вхождения…» для просмотра исключений.')
        finally:
            self.table.blockSignals(False)

    def item_changed(self, item):
        finding = self.findings[item.row()]
        if item.column() == 0:
            finding.selected = item.checkState() == Qt.Checked
        elif item.column() == 2:
            finding.original = item.text()
            finding.entity = key(item.text())
        elif item.column() == 3:
            for related in self.findings:
                if related is finding or (finding.entity and related.entity == finding.entity):
                    related.replacement = item.text()
        self.render_table()
        self.invalidate()

    def add_row(self):
        self.findings.append(Finding('', replacement='[Скрыто]'))
        self.render_table()
        self.invalidate()
        self.table.setCurrentCell(len(self.findings) - 1, 2)
        self.table.editItem(self.table.item(len(self.findings) - 1, 2))

    def link_rows(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if len(rows) < 2:
            self.status.setText('Выделите минимум две строки с Ctrl или Shift, затем свяжите их.')
            return
        first = self.findings[rows[0]]
        entity = first.entity or key(first.original)
        groups = {self.findings[row].entity for row in rows if self.findings[row].entity}
        for row, finding in enumerate(self.findings):
            if row in rows or finding.entity in groups:
                finding.entity, finding.replacement = entity, first.replacement
        self.render_table()
        self.invalidate()

    def select_all(self, selected):
        for finding in self.findings:
            finding.selected = selected
        self.render_table()
        self.invalidate()

    def invalidate(self):
        self._ready = False
        self._preview_outputs = []
        self.review_checkbox.setChecked(False)
        self.result_view.clear()
        self.progress.setValue(0)
        self.status.setText('Изменения требуют повторной проверки результата.')
        self.update_controls()

    def update_controls(self):
        idle = self.worker is None and not self._closing
        self.inputs.setEnabled(idle)
        self.table.setEnabled(idle)
        self.edit_controls.setEnabled(idle and bool(self.source))
        self.find_button.setEnabled(idle and bool(self.source))
        self.verify_button.setEnabled(idle and bool(self.source))
        self.save_button.setEnabled(idle and self._ready and self.review_checkbox.isChecked())
        self.use_button.setEnabled(idle and self._ready and self.review_checkbox.isChecked())
        self.stop_button.setEnabled(not idle and not self._closing and not self._discard_result)

    def start_scan(self, verify=False):
        if self._closing or self.worker is not None or self._model_installer is not None or not self.source:
            return
        model_path = self.model_edit.text().strip()
        if not all((Path(model_path) / name).is_file() for name in
                   ('openvino_model.xml', 'openvino_model.bin', 'tokenizer.json')):
            QMessageBox.warning(self, 'Нужна локальная Qwen',
                                'Выберите установленную модель Qwen OpenVINO или откройте «Модели…».')
            return
        try:
            self.session.assert_current()
            if verify:
                outputs, _ = self.session.render(self.findings)
                preview = '\n\n'.join(outputs)
        except ValueError as exc:
            QMessageBox.warning(self, 'Проверьте замены', str(exc))
            return
        self.invalidate()
        if verify:
            self.result_view.setPlainText(preview)
            self.tabs.setCurrentWidget(self.result_view)
        self._verify, self._discard_result = verify, False
        self._previous_count = len(self.findings)
        self.worker = PrivacyWorker(model_path, self.source, self.findings, verify, self, session=self.session)
        self.worker.progress.connect(self.scan_progress)
        self.worker.completed.connect(self.scan_completed)
        self.worker.error.connect(self.scan_error)
        self.worker.cancelled.connect(self.scan_cancelled)
        self.worker.finished.connect(self.on_worker_finished)
        self.update_controls()
        self.worker.start()

    def scan_progress(self, value, message):
        if not self._closing and not self._discard_result:
            self.progress.setValue(value)
            self.status.setText(message)

    def scan_completed(self, findings, preview):
        if self._closing or self._discard_result:
            return
        added = len(findings) - self._previous_count
        self.findings = findings
        self.render_table()
        self.progress.setValue(100)
        if self._verify:
            try:
                self._preview_outputs, _ = self.session.render(findings)
                self.session.assert_current()
            except ValueError as exc:
                self.scan_error(str(exc))
                return
        self.show_document()
        self._ready = self._verify and added == 0
        if self._ready:
            self.status.setText('Повторная проверка не нашла новых сведений. '
                                'Прочитайте результат целиком перед сохранением.')
        elif self._verify:
            self.status.setText(f'Дополнительно найдено строк: {added}. Проверьте новые строки в конце таблицы '
                                'и снова примените замены. Текущий результат ещё не окончательный.')
            self.table.scrollToBottom()
        else:
            self.status.setText(f'Строк для проверки: {len(findings)}. Отредактируйте замены, '
                                'свяжите варианты одной сущности и нажмите «Применить и проверить повторно».')

    def scan_error(self, message):
        self._ready = False
        if not self._closing and not self._discard_result:
            self.status.setText(message + ' Результат не проверен; сохранение недоступно.')

    def scan_cancelled(self):
        self._ready = False
        if not self._closing:
            self.status.setText('Проверка остановлена. Сохранение доступно после полной повторной проверки.')

    def stop(self):
        if self.worker is not None:
            self._discard_result = True
            self._ready = False
            self.worker.cancel()
            self.status.setText('Остановка проверки…')
            self.update_controls()

    def on_worker_finished(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()
        if self._closing:
            self._request_close(self._close_result)
        else:
            if self._discard_result:
                self.scan_cancelled()
            self.update_controls()

    def save(self):
        if not self._ready or not self.review_checkbox.isChecked() or self.worker is not None or self._closing:
            return
        try:
            self.session.assert_current()
        except ValueError as exc:
            self.invalidate()
            QMessageBox.warning(self, 'Повторите проверку', str(exc))
            return
        if len(self.session.sources) > 1:
            folder = QFileDialog.getExistingDirectory(self, 'Папка для обезличенной пачки')
            if folder:
                try:
                    self.session.save_batch(folder, self._preview_outputs)
                    QMessageBox.information(self, 'Готово', 'Пачка сохранена под нейтральными именами.')
                except (OSError, ValueError):
                    QMessageBox.warning(self, 'Пачка не сохранена полностью',
                                        'Проверьте папку: часть проверенных файлов могла быть сохранена.')
            return
        suffix = Path(self.source_path).suffix.lower() or '.txt'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Сохранить обезличенную расшифровку',
            str(Path(self.source_path).parent / ('Документ_обезличено' + suffix)), 'Текст (*.txt *.md)')
        if not path:
            return
        if not Path(path).suffix:
            path += '.txt'
        try:
            self.session.assert_current()
            outputs, _ = self.session.render(self.findings)
            result = outputs[0]
            if result != self.result_view.toPlainText():
                raise ValueError('Замены изменились. Повторите проверку.')
            save_result(path, result, self.source_path, [])
        except (OSError, ValueError) as exc:
            message = str(exc) if isinstance(exc, ValueError) else 'Не удалось записать файл. Выберите новое имя и доступную папку.'
            QMessageBox.warning(self, 'Результат не сохранён', message)
            return
        self.status.setText('Отдельная обезличенная расшифровка сохранена. Исходник сохранён без изменений.')
        QMessageBox.information(self, 'Готово', 'Сохранено:\n' + path)

    def use_result(self):
        if not self._ready or not self.review_checkbox.isChecked() or self.worker is not None:
            return
        try:
            self.session.assert_current()
            outputs, _ = self.session.render(self.findings)
            if outputs != self._preview_outputs:
                raise ValueError('Замены изменились. Повторите проверку.')
        except ValueError as exc:
            self.invalidate()
            QMessageBox.warning(self, 'Повторите проверку', str(exc))
            return
        self.approved_texts = outputs
        self.approved_aliases = list(dict.fromkeys(
            [f.replacement for f in self.findings if f.selected]
            + [alias for alias in self.session.overrides.values() if alias is not None]))
        self.approved_findings = list(self.findings)
        self.accept()

    def _request_close(self, result=QDialog.Rejected):
        if not self._closing:
            self._close_result = result
        self._closing = True
        self._discard_result = True
        if self.worker is not None:
            self.worker.cancel()
            self.update_controls()
            self.status.setText('Остановка проверки перед закрытием…')
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
