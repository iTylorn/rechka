"""Small offline checks of manifest validation and resumable downloads."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_installer


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200, content_range=""):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


class ModelInstallerTests(unittest.TestCase):
    def setUp(self):
        self.spec = model_installer.ModelSpec(
            "whisper", "Tiny test model", "Example/tiny", "tiny",
            ("model.bin", "config.json"), 100,
        )
        self.payloads = {"model.bin": b"abcdefghijkl", "config.json": b"{}"}
        self.manifest = [
            {
                "name": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_sha1": None,
            }
            for name, data in self.payloads.items()
        ]

    def test_manifest_uses_pinned_revision_and_checks_all_files(self):
        revision = "a" * 40
        tree = [
            {
                "path": entry["name"], "type": "file", "size": entry["size"],
                "lfs": {"oid": entry["sha256"]},
            }
            for entry in self.manifest
        ]
        with patch.object(model_installer, "_request_json", side_effect=[
            {"sha": revision}, tree,
        ]) as request:
            actual_revision, files = model_installer.repository_manifest(self.spec)
        self.assertEqual(actual_revision, revision)
        self.assertEqual(files, self.manifest)
        self.assertIn(revision, request.call_args_list[1].args[0])

    def test_interrupted_download_resumes_and_verifies_sha256(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            worker = model_installer.ModelDownloadWorker(temp_dir, [self.spec])
            calls = []

            def fake_urlopen(request, timeout):
                name = Path(request.full_url).name
                offset = request.get_header("Range")
                calls.append((name, offset))
                if name == "model.bin" and len([c for c in calls if c[0] == name]) == 1:
                    return FakeResponse(self.payloads[name][:4])
                if name == "model.bin":
                    self.assertEqual(offset, "bytes=4-")
                    return FakeResponse(
                        self.payloads[name][4:], 206, "bytes 4-11/12"
                    )
                return FakeResponse(self.payloads[name])

            with patch.object(model_installer, "repository_manifest", return_value=(
                "a" * 40, self.manifest,
            )), patch.object(model_installer.urllib.request, "urlopen", side_effect=fake_urlopen):
                with self.assertRaisesRegex(RuntimeError, "докачки"):
                    worker._install(self.spec)
                partial = Path(temp_dir) / "_downloads" / "tiny" / "model.bin.part"
                self.assertEqual(partial.read_bytes(), b"abcd")
                installed = worker._install(self.spec)

            self.assertTrue(model_installer.model_looks_complete(installed, self.spec))
            for name, data in self.payloads.items():
                self.assertEqual((installed / name).read_bytes(), data)
            self.assertFalse(partial.exists())
            self.assertTrue((installed / "_whispergui_manifest.json").is_file())
            self.assertEqual(calls[1], ("model.bin", "bytes=4-"))

    def test_regular_hub_file_uses_git_blob_hash(self):
        data = b"{}"
        git_hash = hashlib.sha1(b"blob 2\0" + data).hexdigest()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_bytes(data)
            self.assertTrue(model_installer._valid_file(path, 2, git_sha1=git_hash))
            path.write_bytes(b"[]")
            self.assertFalse(model_installer._valid_file(path, 2, git_sha1=git_hash))

    def test_cancel_request_stops_before_download(self):
        worker = model_installer.ModelDownloadWorker("unused", [self.spec])
        worker.cancel()
        with self.assertRaises(model_installer.DownloadCancelled):
            worker._check_cancel()

    def test_stale_settings_paths_are_replaced(self):
        from settings import Settings

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "WhisperGUI"
            config_dir.mkdir()
            stale = str(Path(temp_dir) / "missing")
            (config_dir / "settings.json").write_text(json.dumps({
                "settings_version": 6,
                "models_folder": stale,
                "llm_models_folder": stale,
                "llm_model_path": stale,
                "embed_model_path": stale,
            }), encoding="utf-8")
            with patch.dict("os.environ", {"APPDATA": temp_dir}):
                settings = Settings()
            self.assertNotEqual(settings.get("models_folder"), stale)
            self.assertNotEqual(settings.get("llm_models_folder"), stale)
            self.assertNotEqual(settings.get("llm_model_path"), stale)
            self.assertNotEqual(settings.get("embed_model_path"), stale)

    def test_first_run_offers_download_when_whisper_absent(self):
        from PySide6.QtWidgets import QApplication, QMessageBox
        from main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        with patch("main_window.ModelManager.find_models", return_value=[]), patch(
            "main_window.QMessageBox.question", return_value=QMessageBox.Yes
        ), patch.object(MainWindow, "open_model_installer") as open_installer:
            window = MainWindow()
            window.show()
            window.offer_missing_model_install()
            open_installer.assert_called_once_with(initial_selection=("whisper",))
            window.close()
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
