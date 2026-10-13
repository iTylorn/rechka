"""Whisper fallback selection with managed and legacy model directories."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from transcriber import Transcriber


class WhisperFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()

    def folder(self, relative):
        path = self.base / relative
        path.mkdir(parents=True)
        return path

    def transcriber(self, model_path, fallback=None):
        kwargs = {} if fallback is None else {"fallback_model_path": str(fallback)}
        return Transcriber([], str(model_path), "GPU", 5, **kwargs)

    def test_managed_fingerprint_directory_uses_explicit_fp16(self):
        primary = self.folder("_versions/whisper/111122223333444455556666")
        fallback = self.folder("_versions/whisper_fp16/aaaabbbbccccddddeeeeffff")
        pipeline = object()
        with patch("transcriber.ov_genai.WhisperPipeline", side_effect=[RuntimeError("INT8 GPU"), pipeline]) as load:
            actual = self.transcriber(primary, fallback)._load_whisper_pipeline()
        self.assertEqual(actual, (pipeline, str(fallback)))
        self.assertEqual(load.call_args_list, [call(str(primary), "GPU"), call(str(fallback), "GPU")])

    def test_existing_caller_still_uses_legacy_sibling(self):
        primary = self.folder("whisper-large-v3-turbo-int8-ov")
        fallback = self.folder("whisper-large-v3-turbo-fp16-ov")
        pipeline = object()
        with patch("transcriber.ov_genai.WhisperPipeline", side_effect=[RuntimeError("INT8 GPU"), pipeline]) as load:
            actual = self.transcriber(primary)._load_whisper_pipeline()
        self.assertEqual(actual, (pipeline, str(fallback)))
        self.assertEqual(load.call_args_list, [call(str(primary), "GPU"), call(str(fallback), "GPU")])

    def test_explicit_fallback_is_preferred_to_existing_legacy_sibling(self):
        primary = self.folder("whisper-large-v3-turbo-int8-ov")
        self.folder("whisper-large-v3-turbo-fp16-ov")
        configured = self.folder("_versions/whisper_fp16/fingerprint")
        pipeline = object()
        with patch("transcriber.ov_genai.WhisperPipeline", side_effect=[RuntimeError("primary"), pipeline]) as load:
            actual = self.transcriber(primary, configured)._load_whisper_pipeline()
        self.assertEqual(actual, (pipeline, str(configured)))
        self.assertEqual(load.call_args_list, [call(str(primary), "GPU"), call(str(configured), "GPU")])

    def test_legacy_sibling_is_tried_if_explicit_fallback_fails(self):
        primary = self.folder("whisper-large-v3-turbo-int8-ov")
        legacy = self.folder("whisper-large-v3-turbo-fp16-ov")
        configured = self.folder("_versions/whisper_fp16/fingerprint")
        pipeline = object()
        with patch("transcriber.ov_genai.WhisperPipeline", side_effect=[RuntimeError("primary"), RuntimeError("configured"), pipeline]) as load:
            actual = self.transcriber(primary, configured)._load_whisper_pipeline()
        self.assertEqual(actual, (pipeline, str(legacy)))
        self.assertEqual(load.call_args_list, [call(str(primary), "GPU"), call(str(configured), "GPU"), call(str(legacy), "GPU")])

    def test_primary_success_does_not_load_fallback(self):
        primary = self.folder("_versions/whisper/fingerprint")
        fallback = self.folder("_versions/whisper_fp16/fingerprint")
        pipeline = object()
        with patch("transcriber.ov_genai.WhisperPipeline", return_value=pipeline) as load:
            actual = self.transcriber(primary, fallback)._load_whisper_pipeline()
        self.assertEqual(actual, (pipeline, str(primary)))
        load.assert_called_once_with(str(primary), "GPU")

    def test_identical_primary_and_configured_path_is_not_retried(self):
        primary = self.folder("_versions/whisper_fp16/fingerprint")
        same = primary / ".." / primary.name
        with patch("transcriber.ov_genai.WhisperPipeline", side_effect=RuntimeError("GPU failed")) as load:
            with self.assertRaisesRegex(RuntimeError, "Whisper на GPU: GPU failed"):
                self.transcriber(primary, same)._load_whisper_pipeline()
        load.assert_called_once_with(str(primary), "GPU")

    def test_same_explicit_and_legacy_fallback_is_not_retried(self):
        primary = self.folder("whisper-large-v3-turbo-int8-ov")
        fallback = self.folder("whisper-large-v3-turbo-fp16-ov")
        with patch("transcriber.ov_genai.WhisperPipeline", side_effect=[RuntimeError("primary failed"), RuntimeError("FP16 failed")]) as load:
            with self.assertRaisesRegex(RuntimeError, "primary failed.*FP16 failed"):
                self.transcriber(primary, fallback / ".." / fallback.name)._load_whisper_pipeline()
        self.assertEqual(load.call_count, 2)
        self.assertTrue(all(c.args[1] == "GPU" for c in load.call_args_list))

    def test_missing_fallback_reports_primary_gpu_failure_without_cpu(self):
        primary = self.folder("_versions/whisper/fingerprint")
        with patch("transcriber.ov_genai.WhisperPipeline", side_effect=RuntimeError("no GPU")) as load:
            with self.assertRaisesRegex(RuntimeError, "Whisper на GPU: no GPU"):
                self.transcriber(primary, self.base / "missing")._load_whisper_pipeline()
        load.assert_called_once_with(str(primary), "GPU")

    def test_all_gpu_failures_are_reported_without_cpu_fallback(self):
        primary = self.folder("whisper-large-v3-turbo-int8-ov")
        self.folder("whisper-large-v3-turbo-fp16-ov")
        configured = self.folder("_versions/whisper_fp16/fingerprint")
        with patch("transcriber.ov_genai.WhisperPipeline", side_effect=[RuntimeError("primary failed"), RuntimeError("configured failed"), RuntimeError("legacy failed")]) as load:
            with self.assertRaisesRegex(RuntimeError, "primary failed.*configured failed.*legacy failed"):
                self.transcriber(primary, configured)._load_whisper_pipeline()
        self.assertEqual(load.call_count, 3)
        self.assertTrue(all(c.args[1] == "GPU" for c in load.call_args_list))


if __name__ == "__main__":
    unittest.main()
