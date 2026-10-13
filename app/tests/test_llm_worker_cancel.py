"""Deterministic worker cancellation tests without loading a language model."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from llm_processor import LLMProcessingWorker


class LLMWorkerCancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "transcript.txt"
        self.source.write_text("Test transcript", encoding="utf-8")
        self.worker = LLMProcessingWorker(str(self.base / "model"), str(self.source),
                                          {"id": "summary"})
        self.processor = Mock()
        self.processor.process.return_value = "Result"
        self.completed = []
        self.errors = []
        self.logs = []
        self.worker.completed.connect(lambda *args: self.completed.append(args))
        self.worker.error.connect(self.errors.append)
        self.worker.log.connect(self.logs.append)
        self.environment = patch.dict(os.environ, {"LOCALAPPDATA": str(self.base / "cache")})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def assert_cancelled_without_result(self):
        self.assertFalse(self.completed)
        self.assertFalse(self.errors)
        self.assertTrue(any("остановлена" in line for line in self.logs))
        self.assertEqual(list(self.base.glob("*.md")), [])
        self.assertEqual(list(self.base.glob(".*.tmp")), [])
        self.assertIsNone(self.worker.processor)

    def test_cancel_before_run_prevents_read_and_processor_construction(self):
        self.worker.cancel()
        with patch("llm_processor.LocalLLMProcessor") as factory, \
                patch("llm_processor.Path.read_text") as read:
            self.worker.run()
        read.assert_not_called()
        factory.assert_not_called()
        self.assert_cancelled_without_result()

    def test_cancel_during_transcript_read_prevents_processor_construction(self):
        def read(*args, **kwargs):
            self.worker.cancel()
            return "Test transcript"

        with patch("llm_processor.Path.read_text", side_effect=read), \
                patch("llm_processor.LocalLLMProcessor") as factory:
            self.worker.run()
        factory.assert_not_called()
        self.assert_cancelled_without_result()

    def test_cancel_during_processor_construction_is_latched(self):
        def construct(*args, **kwargs):
            self.worker.cancel()
            return self.processor

        with patch("llm_processor.LocalLLMProcessor", side_effect=construct):
            self.worker.run()
        self.processor.cancel.assert_called_once_with()
        self.processor.process.assert_not_called()
        self.processor.close.assert_called_once_with()
        self.assert_cancelled_without_result()

    def test_cancel_is_forwarded_and_discards_even_a_returned_result(self):
        def process(*args, **kwargs):
            self.worker.cancel()
            return "Result returned during cancellation"

        self.processor.process.side_effect = process
        with patch("llm_processor.LocalLLMProcessor", return_value=self.processor), \
                patch.object(self.worker, "write_result") as write:
            self.worker.run()
        self.assertTrue(self.processor.cancel.called)
        self.processor.close.assert_called_once_with()
        write.assert_not_called()
        self.assert_cancelled_without_result()

    def test_cancel_after_processing_before_publication_prevents_write(self):
        def choose_path():
            self.worker.cancel()
            return str(self.base / "summary.md")

        with patch("llm_processor.LocalLLMProcessor", return_value=self.processor), \
                patch.object(self.worker, "make_output_path", side_effect=choose_path), \
                patch.object(self.worker, "write_result") as write:
            self.worker.run()
        write.assert_not_called()
        self.assert_cancelled_without_result()

    def test_cancel_during_staged_write_leaves_no_published_or_temporary_file(self):
        original_write = LLMProcessingWorker.write_result

        def write_then_cancel(path, result, output_format):
            original_write(path, result, output_format)
            self.worker.cancel()

        with patch("llm_processor.LocalLLMProcessor", return_value=self.processor), \
                patch.object(self.worker, "write_result", side_effect=write_then_cancel), \
                patch("llm_processor.os.replace") as publish:
            self.worker.run()
        publish.assert_not_called()
        self.assert_cancelled_without_result()

    def test_uncancelled_result_keeps_existing_output_and_uses_next_name(self):
        existing = self.base / "transcript_summary.md"
        existing.write_text("Previous result", encoding="utf-8")
        with patch("llm_processor.LocalLLMProcessor", return_value=self.processor):
            self.worker.run()
        result_path = self.base / "transcript_summary_2.md"
        self.assertEqual(self.completed, [(str(result_path), "Result")])
        self.assertEqual(result_path.read_text(encoding="utf-8"), "Result")
        self.assertEqual(existing.read_text(encoding="utf-8"), "Previous result")
        self.assertEqual(list(self.base.glob(".*.tmp")), [])
        self.processor.cancel.assert_not_called()
        self.processor.close.assert_called_once_with()
        self.assertFalse(self.errors)

    def test_write_failure_cleans_temporary_file_without_completion(self):
        with patch("llm_processor.LocalLLMProcessor", return_value=self.processor), \
                patch.object(self.worker, "write_result", side_effect=OSError("disk full")):
            self.worker.run()
        self.assertFalse(self.completed)
        self.assertEqual(self.errors, ["disk full"])
        self.assertEqual(list(self.base.glob("*.md")), [])
        self.assertEqual(list(self.base.glob(".*.tmp")), [])
        self.processor.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
