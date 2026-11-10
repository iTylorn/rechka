import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QDialog

from privacy_dialog import PrivacyDialog
from transcript_privacy import Finding
from test_worker_lifecycle import FakeWorker, MemorySettings


class PrivacyDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'transcript.txt'
        self.source.write_text('Иван, Ивану, Валентин', encoding='utf-8')
        self.settings = MemorySettings(self.temp.name)
        self.dialog = PrivacyDialog(self.settings, str(self.source))
        self.dialog.show()
        self.addCleanup(self.close_dialog)

    def close_dialog(self):
        self.dialog.worker = None
        self.dialog._model_installer = None
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def test_linked_variants_change_together_and_invalidate_preview(self):
        self.dialog.findings = [Finding('Иван', replacement='[А]', entity='иван'),
                                Finding('Ивану', replacement='[А]', entity='иван')]
        self.dialog.render_table()
        self.dialog._ready = True
        self.dialog.table.item(0, 3).setText('[Автор]')
        self.assertEqual([f.replacement for f in self.dialog.findings], ['[Автор]', '[Автор]'])
        self.assertFalse(self.dialog._ready)
        self.assertFalse(self.dialog.save_button.isEnabled())

    def test_residual_findings_require_another_review(self):
        self.dialog._verify = True
        self.dialog._previous_count = 1
        found = [Finding('Иван', replacement='[Автор]'), Finding('Валентин', replacement='[Участник]')]
        self.dialog.scan_completed(found, '[Автор], Ивану, Валентин')
        self.dialog.on_worker_finished()
        self.assertFalse(self.dialog.save_button.isEnabled())
        self.dialog._previous_count = 2
        self.dialog.scan_completed(found, '[Автор], Ивану, [Участник]')
        self.dialog.on_worker_finished()
        self.assertFalse(self.dialog.save_button.isEnabled())
        self.dialog.review_checkbox.setChecked(True)
        self.assertTrue(self.dialog.save_button.isEnabled())
        self.dialog.model_edit.setText('another model')
        self.assertFalse(self.dialog.save_button.isEnabled())

    def test_save_outputs_only_checked_preview_and_keeps_source(self):
        self.dialog.findings = [Finding('Иван', replacement='[Автор]')]
        self.dialog._verify, self.dialog._previous_count = True, 1
        self.dialog.scan_completed(self.dialog.findings, '[Автор], Ивану, Валентин')
        self.dialog.on_worker_finished()
        self.dialog.review_checkbox.setChecked(True)
        output = Path(self.temp.name) / 'public.txt'
        with patch('privacy_dialog.QFileDialog.getSaveFileName', return_value=(str(output), '')), \
                patch('privacy_dialog.QMessageBox.information'):
            self.dialog.save()
        self.assertEqual(output.read_text(encoding='utf-8'), '[Автор], Ивану, Валентин')
        self.assertEqual(self.source.read_text(encoding='utf-8'), 'Иван, Ивану, Валентин')
        self.assertNotIn('privacy_mapping', self.settings.data)

    def test_all_close_routes_wait_for_finished_signal(self):
        for action in ('reject', 'accept', 'done', 'close'):
            for running in (True, False):
                with self.subTest(action=action, running=running):
                    dialog = PrivacyDialog(self.settings, str(self.source))
                    dialog.show()
                    worker = FakeWorker(running)
                    dialog.worker = worker
                    worker.finished.connect(dialog.on_worker_finished)
                    if action == 'done':
                        dialog.done(QDialog.Accepted)
                    else:
                        getattr(dialog, action)()
                    self.assertTrue(dialog.isVisible())
                    self.assertTrue(worker.cancelled)
                    worker.finish()
                    self.assertFalse(dialog.isVisible())
                    self.assertTrue(worker.deleted)
                    dialog.deleteLater()

    def test_cancel_discards_a_queued_success(self):
        self.dialog.worker = FakeWorker(False)
        self.dialog._verify, self.dialog._previous_count = True, 0
        self.dialog.stop()
        self.dialog.scan_completed([], 'should not be accepted')
        self.dialog.on_worker_finished()
        self.assertFalse(self.dialog._ready)
        self.assertFalse(self.dialog.save_button.isEnabled())
        self.assertNotIn('should not', self.dialog.result_view.toPlainText())

    def test_installer_and_new_scan_wait_for_finished_signal(self):
        self.dialog.worker = FakeWorker(False)
        with patch('model_installer.ModelInstallerDialog') as install, \
                patch('privacy_dialog.PrivacyWorker') as worker:
            self.dialog.install_model()
            self.dialog.start_scan()
        install.assert_not_called()
        worker.assert_not_called()

    def test_adding_source_rebuilds_batch_and_invalidates_preview(self):
        self.dialog.findings = [Finding('Иван', replacement='[Автор]')]
        self.dialog._ready = True
        other = Path(self.temp.name) / 'other.txt'
        other.write_text('Новый текст', encoding='utf-8')
        self.dialog.source_combo.addItem(str(other), str(other))
        self.dialog.source_combo.setCurrentIndex(1)
        self.assertEqual(self.dialog.source, 'Иван, Ивану, Валентин\n\nНовый текст')
        self.assertEqual(self.dialog.original_view.toPlainText(), 'Новый текст')
        self.assertEqual(self.dialog.findings, [])
        self.assertFalse(self.dialog.save_button.isEnabled())
