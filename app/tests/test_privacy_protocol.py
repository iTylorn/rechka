"""Verify the privacy gate and the absence of premature protocol publication."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QDialog

from ai_dialog import AIProcessingDialog
from llm_processor import LLMProcessingWorker
from privacy_session import PrivacySession, SourceSnapshot
from transcript_privacy import Finding
from test_worker_lifecycle import MemorySettings
from test_worker_lifecycle import FakeWorker


class PrivacyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / 'private-name.txt'
        self.source.write_text('Иван сказал: привет.', encoding='utf-8')
        self.model = self.folder / 'model'
        self.model.mkdir()
        for name in ('openvino_model.xml', 'openvino_model.bin', 'tokenizer.json'):
            (self.model / name).touch()
        self.settings = MemorySettings(self.folder)
        self.settings.data.update(anonymize_before_ai=True, llm_model_path=str(self.model))
        self.dialog = AIProcessingDialog(self.settings, str(self.source))
        self.addCleanup(self.close_dialog)

    def close_dialog(self):
        self.dialog.worker = None
        self.dialog._privacy_dialog = None
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def approval(self, accepted=True, text='[Участник 1] сказал: привет.'):
        dialog = Mock()
        dialog.exec.return_value = QDialog.Accepted if accepted else QDialog.Rejected
        dialog.approved_texts = [text]
        dialog.approved_aliases = ['[Участник 1]']
        dialog.approved_findings = [Finding('Иван', 'ФИО', '[Участник 1]')]
        dialog.session = PrivacySession([SourceSnapshot.read(self.source)])
        return dialog

    def test_pre_review_cancellation_does_not_start_generation(self):
        with patch('privacy_dialog.PrivacyDialog', return_value=self.approval(False)), \
                patch('ai_dialog.LLMProcessingWorker') as factory:
            self.dialog.start_processing()
        factory.assert_not_called()
        self.assertEqual(list(self.folder.glob('*обезличено*')), [])

    def test_only_sanitized_snapshot_enters_model_and_draft_is_not_written(self):
        with patch('privacy_dialog.PrivacyDialog', return_value=self.approval()), \
                patch.object(LLMProcessingWorker, 'start'):
            self.dialog.start_processing()
        worker = self.dialog.worker
        self.assertFalse(worker.publish)
        self.assertNotIn('Иван', worker.source_text)
        processor = Mock()
        processor.process.return_value = 'Черновик: [Участник 1] подтвердил.'
        with patch('llm_processor.LocalLLMProcessor', return_value=processor), \
                patch('llm_processor.Path.read_text', side_effect=AssertionError('must not reread source')):
            worker.run()
        self.assertNotIn('Иван', processor.process.call_args.args[0])
        self.assertEqual(list(self.folder.glob('*.md')), [])
        self.assertEqual(self.source.read_text(encoding='utf-8'), 'Иван сказал: привет.')
        self.assertEqual(len(list(self.folder.glob('*обезличено.txt'))), 1)
        self.assertIsNotNone(self.dialog._draft_pending)

    def test_post_review_cancellation_never_saves_protocol(self):
        self.dialog._draft_pending = 'Иван вернулся в черновик'
        with patch('privacy_dialog.PrivacyDialog', return_value=self.approval(False)), \
                patch('ai_dialog.QFileDialog.getSaveFileName') as choose:
            self.dialog._review_protocol()
        choose.assert_not_called()
        self.assertFalse(self.dialog.last_output_path)

    def test_post_review_saves_reviewed_html_and_keeps_original(self):
        self.dialog._draft_pending = 'Иван <script>bad</script>'
        self.dialog._privacy_output_format = 'html'
        output = self.folder / 'Протокол_обезличено.html'
        with patch('privacy_dialog.PrivacyDialog', return_value=self.approval(text='[Участник 1] <script>bad</script>')), \
                patch('ai_dialog.QFileDialog.getSaveFileName', return_value=(str(output), '')):
            self.dialog._review_protocol()
        content = output.read_text(encoding='utf-8')
        self.assertNotIn('Иван', content)
        self.assertIn('&lt;script&gt;', content)
        self.assertNotIn('<script>', content)

    def test_changed_source_blocks_protocol_export(self):
        self.dialog._draft_pending = 'Черновик'
        self.dialog._approved_source_session = PrivacySession([SourceSnapshot.read(self.source)])
        self.source.write_text('Внешнее изменение', encoding='utf-8')
        with patch('privacy_dialog.PrivacyDialog', return_value=self.approval()), \
                patch('ai_dialog.QFileDialog.getSaveFileName') as choose, \
                patch('ai_dialog.QMessageBox.warning'):
            self.dialog._review_protocol()
        choose.assert_not_called()

    def test_private_native_exception_does_not_expose_source(self):
        worker = LLMProcessingWorker(str(self.model), str(self.source), {}, source_text='SECRET', publish=False)
        errors = []
        worker.error.connect(errors.append)
        with patch('llm_processor.LocalLLMProcessor', side_effect=RuntimeError('SECRET')):
            worker.run()
        self.assertEqual(len(errors), 1)
        self.assertNotIn('SECRET', errors[0])

    def test_stop_discards_queued_draft_even_after_native_thread_stopped(self):
        self.dialog._privacy_active = True
        self.dialog.worker = FakeWorker(False)
        self.dialog.stop_processing()
        self.dialog.processing_completed('', 'Must not survive cancellation')
        self.assertIsNone(self.dialog._draft_pending)

    def test_new_run_cannot_replace_pending_draft(self):
        self.dialog._draft_pending = 'Pending review'
        with patch.object(self.dialog, '_start_processing') as start:
            self.dialog.start_processing()
        start.assert_not_called()
