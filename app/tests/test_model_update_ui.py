"""Offline Qt integration checks for activation, rollback and safe cancellation."""

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

import model_installer as ui
from model_updates import UpdateResult


class MemorySettings:
    def __init__(self, values):
        self.data = copy.deepcopy(values)
        self.updates = []
        self.fail_save = False

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, values):
        if self.fail_save:
            raise OSError("disk is read only")
        self.updates.append(copy.deepcopy(values))
        self.data.update(values)


class ModelUpdateUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec = ui.MODEL_CATALOG[0]
        self.old = self.make_model(self.spec, self.root / "old", "a" * 40)
        self.new = self.make_model(self.spec, self.root / "new", "b" * 40)
        self.settings = MemorySettings({
            "model_store_folder": str(self.root / "store"),
            "models_folder": str(self.root / "legacy"),
            "llm_models_folder": str(self.root / "legacy-llm"),
            "last_model": str(self.old),
            "model_paths": {self.spec.key: str(self.old)},
            "model_previous_paths": {},
        })
        for name in ("MODEL_STORE_FOLDER", "DEFAULT_MODELS_FOLDER", "DEFAULT_LLM_FOLDER"):
            patcher = patch.object(ui, name, self.root / "missing")
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(ui, "DEFAULT_EMBEDDING_MODEL", self.root / "missing.xml")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dialogs = []
        self.addCleanup(self.close_dialogs)

    def make_model(self, spec, folder, revision=None):
        folder.mkdir(parents=True, exist_ok=True)
        files = ("wespeaker_fixed.xml", "wespeaker_fixed.bin") if spec.kind == "wespeaker" else spec.files
        for name in files:
            (folder / name).write_bytes(b"test model bytes")
        if revision:
            (folder / ui.MARKER_NAME).write_text(json.dumps({
                "repository": spec.repo, "revision": revision,
            }), encoding="utf-8")
        return folder

    def dialog(self, selection=("whisper",)):
        dialog = ui.ModelInstallerDialog(self.settings, initial_selection=selection)
        self.dialogs.append(dialog)
        return dialog

    def close_dialogs(self):
        for dialog in self.dialogs:
            if dialog.worker is not None:
                dialog.worker.cancel()
                dialog.worker.wait(4000)
                self.app.processEvents()
            dialog.reject()
            dialog.deleteLater()
        self.app.processEvents()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(predicate(), "Qt worker did not settle in time")

    def result(self, status="update_available"):
        return UpdateResult(status, "a" * 40, "b" * 40, (), object(), "test result")

    def test_open_is_offline_and_active_path_wins_without_history_scan(self):
        history = self.make_model(self.spec, self.root / "store" / "_versions" / "whisper" / "unused")
        with patch.object(ui, "check_update", side_effect=AssertionError("network on open")), patch.object(
            ui, "install_model", side_effect=AssertionError("installation on open")
        ):
            dialog = self.dialog()
            self.assertEqual(dialog.table.rowCount(), 5)
            self.assertEqual(ui.model_path_for(self.settings, "whisper"), self.old)
            self.assertEqual(dialog.table.item(0, 2).text(), "a" * 12)
            self.assertEqual(dialog.table.item(0, 3).text(), "Не проверялась")
            self.settings.data["model_paths"] = {}
            self.settings.data["last_model"] = str(history)
            # The folder has no identifying marker and must not become a catalog preset.
            self.assertIsNone(ui.find_installed_model(self.settings, self.spec))
        self.assertEqual(self.settings.updates, [])

    def test_one_network_error_does_not_abort_other_models_or_change_settings(self):
        dialog = self.dialog()
        calls = []

        def check(spec, *_args, **_kwargs):
            calls.append(spec.key)
            if spec.key == "qwen8":
                raise OSError("network unavailable")
            return self.result()

        with patch.object(ui, "check_update", side_effect=check):
            dialog.check_updates()
            self.assertFalse(dialog.choose_button.isEnabled())
            self.wait_for(lambda: dialog.worker is None)
        self.assertEqual(calls, [spec.key for spec in ui.MODEL_CATALOG])
        self.assertIn("qwen8", dialog._errors)
        self.assertIn("wespeaker", dialog._results)
        self.assertEqual(self.settings.updates, [])

    def test_manual_selection_overrides_older_managed_catalog_path(self):
        for key, setting_key in (("whisper", "last_model"), ("qwen14", "llm_model_path"),
                                 ("wespeaker", "embed_model_path")):
            with self.subTest(key=key):
                spec = next(spec for spec in ui.MODEL_CATALOG if spec.key == key)
                managed = self.make_model(spec, self.root / key / "managed", "a" * 40)
                manual = self.make_model(spec, self.root / key / "manual", "b" * 40)
                if key == "wespeaker":
                    custom_xml = manual / "my_voice.xml"
                    custom_xml.write_bytes(b"xml")
                    custom_xml.with_suffix(".bin").write_bytes(b"weights")
                    manual = custom_xml
                self.settings.data["model_paths"][key] = str(managed)
                self.settings.data[setting_key] = str(manual)
                self.assertEqual(ui.find_installed_model(self.settings, spec), manual)
                dialog = self.dialog((key,))
                self.assertEqual(dialog._current_paths[key], manual)

    def test_install_uses_pinned_plan_and_activates_only_validated_result(self):
        dialog = self.dialog()
        dialog.show()
        checked = self.result()
        dialog.on_checked(self.spec.key, checked)

        def install(spec, _base, **kwargs):
            self.assertIs(kwargs["plan"], checked.plan)
            self.assertEqual(kwargs["current_path"], self.old)
            self.assertEqual(self.settings.get("last_model"), str(self.old))
            return self.new

        with patch.object(ui, "check_update", side_effect=AssertionError("pinned plan should be reused")), patch.object(
            ui, "install_model", side_effect=install
        ):
            dialog.start_download()
            self.wait_for(lambda: dialog.worker is None)
        self.assertTrue(dialog.isVisible(), "Successful installation should keep dialog open")
        self.assertEqual(self.settings.get("last_model"), str(self.new))
        self.assertEqual(self.settings.get("model_previous_paths")[self.spec.key], str(self.old))
        self.assertEqual(self.settings.get("models_folder"), str(self.root / "legacy"))
        self.assertEqual(len(self.settings.updates), 1)
        self.assertEqual(dialog.table.item(0, 3).text(), "b" * 12)
        dialog.reject()
        self.assertEqual(dialog.result(), QDialog.Accepted)

    def test_failed_install_or_failed_settings_save_preserves_active_model(self):
        for save_failure in (False, True):
            with self.subTest(save_failure=save_failure):
                dialog = self.dialog()
                before = copy.deepcopy(self.settings.data)
                self.settings.fail_save = save_failure
                with patch.object(ui, "check_update", return_value=self.result()), patch.object(
                    ui, "install_model", return_value=self.new,
                    side_effect=None if save_failure else RuntimeError("GPU validation failed"),
                ):
                    dialog.start_download()
                    self.wait_for(lambda: dialog.worker is None)
                self.assertEqual(self.settings.data, before)
                self.assertFalse(dialog._changed)
                self.assertIn(self.spec.key, dialog._errors)

    def test_rollback_validates_before_switching_and_failure_preserves_paths(self):
        self.settings.data["model_previous_paths"] = {self.spec.key: str(self.new)}
        dialog = self.dialog()
        before = copy.deepcopy(self.settings.data)
        with patch.object(ui, "validate_model", side_effect=RuntimeError("old GPU model unusable")):
            dialog.start_rollback()
            self.wait_for(lambda: dialog.worker is None)
        self.assertEqual(self.settings.data, before)

        def validate(spec, path, **_kwargs):
            self.assertEqual(Path(path), self.new)
            self.assertEqual(self.settings.data, before)

        with patch.object(ui, "validate_model", side_effect=validate), patch.object(
            ui, "check_update", side_effect=AssertionError("rollback must work offline")
        ):
            dialog.start_rollback()
            self.wait_for(lambda: dialog.worker is None)
        self.assertEqual(self.settings.get("model_paths")[self.spec.key], str(self.new))
        self.assertEqual(self.settings.get("model_previous_paths")[self.spec.key], str(self.old))

    def test_wespeaker_custom_xml_survives_install_and_rollback(self):
        spec = next(spec for spec in ui.MODEL_CATALOG if spec.key == "wespeaker")
        custom_xml = self.root / "custom_voice.xml"
        custom_xml.write_bytes(b"xml")
        custom_xml.with_suffix(".bin").write_bytes(b"weights")
        installed = self.make_model(spec, self.root / "new-speaker")
        self.settings.data["embed_model_path"] = str(custom_xml)
        dialog = self.dialog((spec.key,))
        with patch.object(ui, "check_update", return_value=self.result()), patch.object(ui, "install_model", return_value=installed):
            dialog.start_download()
            self.wait_for(lambda: dialog.worker is None)
        self.assertEqual(self.settings.get("model_previous_paths")[spec.key], str(custom_xml))
        dialog.table.setCurrentCell(dialog._rows[spec.key], 1)
        with patch.object(ui, "validate_model") as validate:
            dialog.start_rollback()
            self.wait_for(lambda: dialog.worker is None)
        self.assertEqual(validate.call_args.args[1], str(custom_xml))
        self.assertEqual(self.settings.get("embed_model_path"), str(custom_xml))
        self.assertEqual(self.settings.get("model_previous_paths")[spec.key], str(installed))

    def test_escape_waits_for_thread_and_cancellation_prevents_activation(self):
        dialog = self.dialog()
        entered = threading.Event()
        release = threading.Event()

        def install(*_args, **_kwargs):
            entered.set()
            release.wait(2)
            return self.new

        dialog.show()
        with patch.object(ui, "check_update", return_value=self.result()), patch.object(ui, "install_model", side_effect=install):
            dialog.start_download()
            self.wait_for(entered.is_set)
            worker = dialog.worker
            QTest.keyClick(dialog, Qt.Key_Escape)
            self.assertIs(dialog.worker, worker)
            self.assertTrue(dialog.isVisible())
            self.assertTrue(worker._cancel.is_set())
            release.set()
            self.wait_for(lambda: dialog.worker is None)
        self.assertFalse(dialog.isVisible())
        self.assertEqual(self.settings.get("last_model"), str(self.old))
        self.assertEqual(self.settings.updates, [])

    def test_cancel_before_queued_success_is_delivered_preserves_active_path(self):
        dialog = self.dialog()
        with patch.object(ui, "check_update", return_value=self.result()), patch.object(
            ui, "install_model", return_value=self.new
        ):
            dialog.start_download()
            # Hold the GUI event queue until success is queued by the real thread.
            self.assertTrue(dialog.worker.wait(2000))
            dialog.cancel_operation()
            self.wait_for(lambda: dialog.worker is None)
        self.assertEqual(self.settings.get("last_model"), str(self.old))
        self.assertEqual(self.settings.updates, [])
        self.assertIn("отменена", dialog.status_label.text())


if __name__ == "__main__":
    unittest.main()
