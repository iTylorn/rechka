"""Profile isolation and entry-point checks for the beta model manager."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import settings as settings_module
from settings import Settings


class BetaProfileTests(unittest.TestCase):
    def test_stable_profile_is_imported_without_writing_to_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stable = root / "WhisperGUI"
            stable.mkdir()
            original = {"settings_version": 7, "speaker_names": ["Анна"],
                        "model_store_folder": str(root / "stable-models")}
            original_bytes = json.dumps(original).encode("utf-8")
            (stable / "settings.json").write_bytes(original_bytes)
            (stable / "prompts.json").write_text("[]", encoding="utf-8")
            beta_store = root / "beta-models"
            with patch.dict(os.environ, {"APPDATA": folder}), patch.object(
                settings_module, "MODEL_STORE_FOLDER", beta_store
            ):
                settings = Settings()
                self.assertEqual(settings.get("speaker_names"), ["Анна"])
                self.assertEqual(settings.get("model_store_folder"), str(beta_store))
                self.assertEqual(settings.config_dir, root / "WhisperGUI-Beta")
                self.assertTrue((settings.config_dir / "prompts.json").is_file())
                settings.update({"speaker_names": ["Борис"]})
                reopened = Settings()
                self.assertEqual(reopened.get("speaker_names"), ["Борис"])
            self.assertEqual((stable / "settings.json").read_bytes(), original_bytes)
            self.assertEqual((stable / "prompts.json").read_text(), "[]")

    def test_failed_atomic_update_preserves_active_model(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"APPDATA": folder}):
            settings = Settings()
            settings.update({"model_paths": {"whisper": "old-model"}})
            original = settings.config_file.read_bytes()
            with patch("settings.os.replace", side_effect=OSError("write denied")):
                with self.assertRaises(OSError):
                    settings.update({"model_paths": {"whisper": "new-model"}})
            self.assertEqual(settings.get("model_paths"), {"whisper": "old-model"})
            self.assertEqual(settings.config_file.read_bytes(), original)

    def test_independent_settings_do_not_share_mutable_defaults(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"APPDATA": folder}):
            first = Settings()
            first.data["model_paths"]["whisper"] = "first"
            second = Settings()
            self.assertNotIn("whisper", second.get("model_paths"))

    def test_first_run_offers_download_when_whisper_absent(self):
        from PySide6.QtWidgets import QApplication, QMessageBox
        from main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"APPDATA": folder}), patch(
            "main_window.ModelManager.find_models", return_value=[]
        ), patch("main_window.model_path_for", return_value=None), patch(
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
