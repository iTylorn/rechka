"""Exercise Qt shutdown and modal-operation exclusion without running inference."""

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QDialog

from ai_dialog import AIProcessingDialog
from main_window import MainWindow
from settings import DEFAULT_SETTINGS


class MemorySettings:
    def __init__(self, folder):
        self.config_dir = Path(folder)
        self.data = copy.deepcopy(DEFAULT_SETTINGS)
        self.data.update({"enable_diarization": False, "auto_ai_processing": False})

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class FakeWorker(QObject):
    finished = Signal()

    def __init__(self, running=True):
        super().__init__()
        self.running = running
        self.cancelled = False
        self.deleted = False

    def isRunning(self):
        return self.running

    def cancel(self):
        self.cancelled = True

    def start(self):
        self.running = True

    def deleteLater(self):
        self.deleted = True

    def finish(self):
        self.running = False
        self.finished.emit()


class FakeTranscriber(FakeWorker):
    log = Signal(str)
    progress = Signal(int, int, str)
    file_done = Signal(str, int, int)
    finished_ok = Signal(str)
    error = Signal(str)


class WorkerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = MemorySettings(self.temporary.name)
        self.widgets = []
        self.addCleanup(self.close_widgets)

    def close_widgets(self):
        for widget in self.widgets:
            if isinstance(widget, MainWindow):
                widget.transcriber = None
                widget._model_installer = None
                widget._ai_dialog = None
            else:
                widget.worker = None
                widget._model_installer = None
            widget.close()
            widget.deleteLater()
        self.app.processEvents()

    def main_window(self):
        with patch("main_window.Settings", return_value=self.settings), patch(
            "main_window.ModelManager.find_models", return_value=[]
        ), patch("main_window.model_path_for", return_value=None):
            window = MainWindow()
        self.widgets.append(window)
        window.show()
        window._model_offer_timer.stop()
        return window

    def ai_dialog(self, auto_start=False):
        with patch("ai_dialog.model_path_for", return_value=None):
            dialog = AIProcessingDialog(self.settings, auto_start=auto_start)
        self.widgets.append(dialog)
        dialog.show()
        return dialog

    def test_main_close_waits_for_finished_even_when_thread_has_stopped(self):
        for running in (True, False):
            with self.subTest(running=running):
                window = self.main_window()
                worker = FakeWorker(running)
                window.transcriber = worker
                worker.finished.connect(window.on_thread_finished)
                window.pending_ai_path = "transcript.txt"
                window._auto_ai_path = "queued.txt"
                window._auto_ai_timer.start(200)
                self.settings.set("auto_ai_processing", True)
                with patch.object(window, "open_ai_processing") as open_ai:
                    window.close()
                    self.assertTrue(window.isVisible())
                    self.assertTrue(worker.cancelled)
                    self.assertIs(window.transcriber, worker)
                    self.assertFalse(window.start_btn.isEnabled())
                    self.assertFalse(window._auto_ai_timer.isActive())
                    worker.finish()
                    self.assertFalse(window.isVisible())
                    self.assertIsNone(window.transcriber)
                    self.assertTrue(worker.deleted)
                    self.assertIsNone(window.pending_ai_path)
                    open_ai.assert_not_called()

    def test_ai_all_exit_routes_wait_for_worker_finished(self):
        for action in ("reject", "accept", "done", "close"):
            for running in (True, False):
                with self.subTest(action=action, running=running):
                    dialog = self.ai_dialog()
                    worker = FakeWorker(running)
                    dialog.worker = worker
                    worker.finished.connect(dialog.on_worker_finished)
                    dialog.set_running(True)
                    if action == "done":
                        dialog.done(QDialog.Accepted)
                    else:
                        getattr(dialog, action)()
                    self.assertTrue(dialog.isVisible())
                    self.assertTrue(worker.cancelled)
                    self.assertIs(dialog.worker, worker)
                    worker.finish()
                    self.assertFalse(dialog.isVisible())
                    self.assertIsNone(dialog.worker)
                    self.assertTrue(worker.deleted)
                    expected = QDialog.Accepted if action in ("accept", "done") else QDialog.Rejected
                    self.assertEqual(dialog.result(), expected)

    def test_ai_installer_waits_for_queued_worker_finished_signal(self):
        dialog = self.ai_dialog()
        dialog.worker = FakeWorker(running=False)
        with patch("model_installer.ModelInstallerDialog") as installer:
            self.assertFalse(dialog.install_model())
            installer.assert_not_called()

    def test_ai_auto_start_defers_until_model_manager_closes(self):
        dialog = self.ai_dialog(auto_start=True)
        dialog._model_installer = Mock()
        with patch.object(dialog, "start_processing") as start:
            dialog._run_auto_start()
            start.assert_not_called()
            self.assertTrue(dialog._auto_start_pending)
            self.assertTrue(dialog._auto_start_timer.isActive())
            dialog._model_installer = None
            dialog._run_auto_start()
            start.assert_called_once_with()
            self.assertFalse(dialog._auto_start_pending)

    def test_ai_manual_start_is_blocked_while_manager_open(self):
        dialog = self.ai_dialog()
        dialog._model_installer = Mock()
        with patch.object(dialog, "_start_processing") as start:
            dialog.start_processing()
            start.assert_not_called()

    def test_closed_ai_dialog_cannot_auto_start_later(self):
        dialog = self.ai_dialog(auto_start=True)
        dialog.reject()
        with patch.object(dialog, "start_processing") as start:
            dialog._run_auto_start()
            start.assert_not_called()
        self.assertFalse(dialog._auto_start_timer.isActive())

    def test_main_auto_ai_defers_during_model_manager(self):
        window = self.main_window()
        window._auto_ai_path = "captured-transcript.txt"
        window._model_installer = Mock()
        with patch.object(window, "open_ai_processing") as open_ai:
            window._run_auto_ai()
            open_ai.assert_not_called()
            self.assertTrue(window._auto_ai_timer.isActive())
            self.assertEqual(window._auto_ai_path, "captured-transcript.txt")
            window._model_installer = None
            window._run_auto_ai()
            open_ai.assert_called_once_with(True)
            self.assertEqual(window.last_saved_path, "captured-transcript.txt")
            self.assertIsNone(window._auto_ai_path)

    def test_first_run_offer_does_not_nest_inside_manager(self):
        window = self.main_window()
        window._model_installer = Mock()
        with patch("main_window.QMessageBox.question") as question:
            window.offer_missing_model_install()
            question.assert_not_called()
        self.assertTrue(window._model_offer_timer.isActive())

    def test_main_manager_blocks_transcription_and_auto_ai_reentrancy(self):
        window = self.main_window()
        window._auto_ai_path = "transcript.txt"
        installer = Mock()

        def nested_events():
            window.start_transcription()
            window._run_auto_ai()

        installer.exec.side_effect = nested_events
        with patch("main_window.ModelInstallerDialog", return_value=installer), patch(
            "main_window.Transcriber"
        ) as transcriber, patch.object(window, "open_ai_processing") as open_ai:
            window.open_model_installer()
            transcriber.assert_not_called()
            open_ai.assert_not_called()
        self.assertIsNone(window._model_installer)
        installer.deleteLater.assert_called_once_with()

    def test_main_passes_managed_fp16_fallback_to_transcriber(self):
        window = self.main_window()
        window.current_files = ["recording.wav"]
        window.model_combo.addItem("Whisper INT8", "managed-int8")
        worker = FakeTranscriber()
        with patch("main_window.Transcriber", return_value=worker) as factory, patch(
            "main_window.model_path_for", return_value=Path("managed-fp16")
        ):
            window.start_transcription()
        self.assertEqual(factory.call_args.kwargs["fallback_model_path"], "managed-fp16")
        worker.finish()


if __name__ == "__main__":
    unittest.main()
