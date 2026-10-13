"""Offline update safety checks; no model downloads or real GPU are required."""

from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import model_updates as updates


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200, content_range="", chunk_size=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}
        self.chunk_size = chunk_size

    def read(self, size=-1):
        return super().read(min(size, self.chunk_size) if self.chunk_size else size)


class ModelUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.spec = updates.ModelSpec("whisper", "Tiny", "Example/tiny", "tiny",
                                      ("model.bin", "config.json"), 1)
        self.payloads = {"model.bin": b"abcdefghijkl", "config.json": b"{}"}
        self.files = self.artifacts(self.payloads)
        self.plan = self.plan_for(self.files)

    @staticmethod
    def artifacts(payloads):
        return tuple(updates.Artifact(name, len(data), hashlib.sha256(data).hexdigest())
                     for name, data in payloads.items())

    def plan_for(self, files, revision="a" * 40, spec=None):
        spec = spec or self.spec
        return updates.UpdatePlan(spec.key, spec.repo, revision, files,
                                  updates._fingerprint(spec, files))

    def write_model(self, directory, payloads=None, plan=None, marker=False):
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in (payloads or self.payloads).items():
            (directory / name).write_bytes(data)
        if marker:
            plan = plan or self.plan
            (directory / updates.MARKER_NAME).write_text(json.dumps({
                "repository": plan.repository, "revision": plan.revision,
                "fingerprint": plan.fingerprint,
                "files": [asdict(a) for a in plan.files],
            }), encoding="utf-8")
        return directory

    def fake_urlopen(self, request, timeout):
        self.assertIn(f"/resolve/{self.plan.revision}/", request.full_url)
        return FakeResponse(self.payloads[Path(request.full_url).name])

    def install(self, **kwargs):
        with patch.object(updates.urllib.request, "urlopen", side_effect=self.fake_urlopen), \
                patch.object(updates, "validate_model"):
            return updates.install_model(self.spec, self.base, self.plan, **kwargs)

    def test_manifest_pins_revision_and_hashes_required_files(self):
        tree = [{"type": "file", "path": a.name, "size": a.size,
                 "lfs": {"oid": "sha256:" + a.sha256}} for a in self.files]
        tree.append({"type": "file", "path": "README.md", "size": 3, "oid": "f" * 40})
        with patch.object(updates, "_request_json", side_effect=[{"sha": "a" * 40}, tree]) as fetch:
            plan = updates.repository_manifest(self.spec)
        self.assertEqual(plan, self.plan)
        self.assertIn("/tree/" + "a" * 40, fetch.call_args_list[1].args[0])
        self.assertEqual(plan.total_size, 14)

    def test_manifest_without_any_hash_fails_closed(self):
        tree = [{"type": "file", "path": a.name, "size": a.size} for a in self.files]
        with patch.object(updates, "_request_json", side_effect=[{"sha": "a" * 40}, tree]):
            with self.assertRaisesRegex(ValueError, "контрольной суммы"):
                updates.repository_manifest(self.spec)

    def test_regular_git_blob_checksum(self):
        payload = b"{}"
        artifact = updates.Artifact("config.json", 2, git_sha1=hashlib.sha1(b"blob 2\0" + payload).hexdigest())
        path = self.base / artifact.name
        path.write_bytes(payload)
        self.assertTrue(updates._valid_file(path, artifact))
        path.write_bytes(b"[]")
        self.assertFalse(updates._valid_file(path, artifact))

    def test_metadata_only_revision_does_not_offer_model_update(self):
        old = self.plan_for(self.files, "b" * 40)
        current = self.write_model(self.base / "current", plan=old, marker=True)
        marker_before = (current / updates.MARKER_NAME).read_bytes()
        with patch.object(updates, "repository_manifest", return_value=self.plan):
            result = updates.check_update(self.spec, current)
        self.assertEqual(result.status, "up_to_date")
        self.assertEqual(result.current_revision, old.revision)
        self.assertEqual(result.latest_revision, self.plan.revision)
        self.assertEqual((current / updates.MARKER_NAME).read_bytes(), marker_before)

    def test_real_artifact_change_offers_update(self):
        previous_payloads = {**self.payloads, "model.bin": b"old old old!"}
        previous = self.plan_for(self.artifacts(previous_payloads), "b" * 40)
        current = self.write_model(self.base / "current", previous_payloads, previous, True)
        with patch.object(updates, "repository_manifest", return_value=self.plan):
            result = updates.check_update(self.spec, current)
        self.assertEqual(result.status, "update_available")
        self.assertEqual(result.plan, self.plan)

    def test_legacy_model_is_recognized_only_by_matching_bytes(self):
        current = self.write_model(self.base / "legacy")
        with patch.object(updates, "repository_manifest", return_value=self.plan):
            self.assertEqual(updates.check_update(self.spec, current).status, "up_to_date")
            (current / "model.bin").write_bytes(b"not trusted!")
            self.assertEqual(updates.check_update(self.spec, current).status, "unknown")
        self.assertFalse((current / updates.MARKER_NAME).exists())

    def test_local_wespeaker_ir_without_provenance_remains_unknown(self):
        spec = next(s for s in updates.MODEL_CATALOG if s.kind == "wespeaker")
        files = (updates.Artifact(spec.files[0], 3, hashlib.sha256(b"src").hexdigest()),)
        plan = self.plan_for(files, spec=spec)
        directory = self.write_model(self.base / "embed", {
            updates.WESPEAKER_XML: b"xml", "wespeaker_fixed.bin": b"bin",
        })
        with patch.object(updates, "repository_manifest", return_value=plan):
            result = updates.check_update(spec, directory / updates.WESPEAKER_XML)
        self.assertTrue(updates.installed_model_complete(spec, directory))
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.current_revision)

    def test_incomplete_marker_does_not_assert_known_provenance(self):
        current = self.write_model(self.base / "current", marker=True)
        (current / "config.json").unlink()
        with patch.object(updates, "repository_manifest", return_value=self.plan):
            self.assertEqual(updates.check_update(self.spec, current).status, "unknown")

    def test_no_installation_is_distinct_from_unknown(self):
        with patch.object(updates, "repository_manifest", return_value=self.plan):
            self.assertEqual(updates.check_update(self.spec).status, "not_installed")

    def test_download_is_pinned_and_published_only_after_gpu_validation(self):
        seen = []
        progress = []

        def validate(spec, directory, cancel):
            seen.append(directory)
            self.assertIn("_downloads", directory.parts)
            self.assertFalse((self.base / "_versions").exists())
            self.assertFalse((directory / updates.MARKER_NAME).exists())
            self.assertEqual((directory / "model.bin").read_bytes(), self.payloads["model.bin"])

        with patch.object(updates.urllib.request, "urlopen", side_effect=self.fake_urlopen), \
                patch.object(updates, "validate_model", side_effect=validate), \
                patch.object(updates, "repository_manifest") as new_manifest:
            target = updates.install_model(self.spec, self.base, self.plan,
                                            progress=lambda *a: progress.append(a))
        new_manifest.assert_not_called()
        self.assertEqual(target.parent, self.base / "_versions" / self.spec.key)
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0].exists())
        self.assertEqual(progress[-1][0], 100)
        self.assertEqual(json.loads((target / updates.MARKER_NAME).read_text())["revision"], self.plan.revision)

    def test_interrupted_download_resumes_only_same_revision(self):
        calls = []

        def response(request, timeout):
            name = Path(request.full_url).name
            calls.append((name, request.get_header("Range")))
            if len(calls) == 1:
                return FakeResponse(b"abcd")
            if name == "model.bin":
                self.assertEqual(request.get_header("Range"), "bytes=4-")
                return FakeResponse(b"efghijkl", 206, "bytes 4-11/12")
            return FakeResponse(b"{}")

        with patch.object(updates.urllib.request, "urlopen", side_effect=response), \
                patch.object(updates, "validate_model"):
            with self.assertRaisesRegex(RuntimeError, "докачки"):
                updates.install_model(self.spec, self.base, self.plan)
            partials = list((self.base / "_downloads").rglob("*.part"))
            self.assertEqual(len(partials), 1)
            self.assertIn(self.plan.revision, partials[0].parent.name)
            target = updates.install_model(self.spec, self.base, self.plan)
        self.assertEqual((target / "model.bin").read_bytes(), self.payloads["model.bin"])

    def test_new_revision_does_not_mix_old_partial_bytes(self):
        old_stage = self.base / "_downloads" / self.spec.folder / ("b" * 40 + "-" + self.plan.fingerprint[:24])
        old_stage.mkdir(parents=True)
        (old_stage / "model.bin.part").write_bytes(b"OLD!")
        requests = []

        def response(request, timeout):
            requests.append(request)
            return self.fake_urlopen(request, timeout)

        with patch.object(updates.urllib.request, "urlopen", side_effect=response), \
                patch.object(updates, "validate_model"):
            updates.install_model(self.spec, self.base, self.plan)
        self.assertTrue(all(r.get_header("Range") is None for r in requests))
        self.assertEqual((old_stage / "model.bin.part").read_bytes(), b"OLD!")

    def test_cancel_preserves_partial_and_old_active_then_resumes(self):
        old = self.write_model(self.base / "active", {"model.bin": b"old", "config.json": b"[]"})
        cancelled = threading.Event()

        def progress(percent, text):
            if "model.bin" in text:
                cancelled.set()

        with patch.object(updates.urllib.request, "urlopen",
                          return_value=FakeResponse(self.payloads["model.bin"], chunk_size=4)), \
                patch.object(updates, "validate_model") as validate:
            with self.assertRaises(updates.DownloadCancelled):
                updates.install_model(self.spec, self.base, self.plan, current_path=old,
                                      progress=progress, cancel=cancelled)
        validate.assert_not_called()
        partial = next((self.base / "_downloads").rglob("*.part"))
        self.assertEqual(partial.read_bytes(), b"abcd")
        self.assertEqual((old / "model.bin").read_bytes(), b"old")
        self.assertFalse((self.base / "_versions").exists())
        cancelled.clear()

        def resume(request, timeout):
            if Path(request.full_url).name == "model.bin":
                self.assertEqual(request.get_header("Range"), "bytes=4-")
                return FakeResponse(b"efghijkl", 206, "bytes 4-11/12")
            return FakeResponse(b"{}")

        with patch.object(updates.urllib.request, "urlopen", side_effect=resume), \
                patch.object(updates, "validate_model"):
            target = updates.install_model(self.spec, self.base, self.plan, current_path=old, cancel=cancelled)
        self.assertEqual((target / "model.bin").read_bytes(), self.payloads["model.bin"])
        self.assertEqual((old / "config.json").read_bytes(), b"[]")

    def test_bad_hash_never_reaches_gpu_or_publishes(self):
        old = self.write_model(self.base / "active")
        with patch.object(updates.urllib.request, "urlopen", return_value=FakeResponse(b"X" * 12)), \
                patch.object(updates, "validate_model") as validate:
            with self.assertRaisesRegex(RuntimeError, "проверку"):
                updates.install_model(self.spec, self.base, self.plan)
        validate.assert_not_called()
        self.assertFalse(list((self.base / "_downloads").rglob("*.part")))
        self.assertFalse((self.base / "_versions").exists())
        self.assertEqual((old / "model.bin").read_bytes(), self.payloads["model.bin"])

    def test_wrong_range_total_rejected_before_appending(self):
        stage = self.base / "stage"
        stage.mkdir()
        partial = stage / "model.bin.part"
        partial.write_bytes(b"abcd")
        with patch.object(updates.urllib.request, "urlopen",
                          return_value=FakeResponse(b"efghijkl", 206, "bytes 4-11/99")):
            with self.assertRaisesRegex(RuntimeError, "диапазон"):
                updates._download_file(self.spec, self.plan, self.files[0], stage, Mock(), None)
        self.assertEqual(partial.read_bytes(), b"abcd")

    def test_ignored_range_restarts_without_duplicating_prefix(self):
        stage = self.base / "stage"
        stage.mkdir()
        (stage / "model.bin.part").write_bytes(b"abcd")
        with patch.object(updates.urllib.request, "urlopen", side_effect=self.fake_urlopen):
            updates._download_file(self.spec, self.plan, self.files[0], stage, Mock(), None)
        self.assertEqual((stage / "model.bin").read_bytes(), self.payloads["model.bin"])

    def test_gpu_failure_preserves_existing_active_model(self):
        old = self.write_model(self.base / "active", marker=True)
        before = {p.name: p.read_bytes() for p in old.iterdir()}
        with patch.object(updates.urllib.request, "urlopen", side_effect=self.fake_urlopen), \
                patch.object(updates, "validate_model", side_effect=RuntimeError("GPU compile failed")):
            with self.assertRaisesRegex(RuntimeError, "GPU compile failed"):
                updates.install_model(self.spec, self.base, self.plan, current_path=old)
        self.assertEqual(before, {p.name: p.read_bytes() for p in old.iterdir()})
        self.assertFalse((self.base / "_versions").exists())
        self.assertFalse(list((self.base / "_downloads").rglob(updates.MARKER_NAME)))

    def test_cancel_after_gpu_validation_still_does_not_publish(self):
        cancelled = threading.Event()
        old = self.write_model(self.base / "active", marker=True)
        before = {p.name: p.read_bytes() for p in old.iterdir()}
        with patch.object(updates.urllib.request, "urlopen", side_effect=self.fake_urlopen), \
                patch.object(updates, "validate_model", side_effect=lambda *a: cancelled.set()):
            with self.assertRaises(updates.DownloadCancelled):
                updates.install_model(self.spec, self.base, self.plan, current_path=old, cancel=cancelled)
        self.assertFalse((self.base / "_versions").exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in old.iterdir()})

    def test_wespeaker_conversion_provenance_covers_source_and_both_outputs(self):
        spec = next(s for s in updates.MODEL_CATALOG if s.kind == "wespeaker")
        files = self.artifacts({spec.files[0]: b"fake onnx"})
        plan = self.plan_for(files, spec=spec)

        def convert(spec, directory, cancel):
            self.assertEqual((directory / spec.files[0]).read_bytes(), b"fake onnx")
            (directory / updates.WESPEAKER_XML).write_bytes(b"fixed xml")
            (directory / "wespeaker_fixed.bin").write_bytes(b"fp16 bin")

        with patch.object(updates.urllib.request, "urlopen", return_value=FakeResponse(b"fake onnx")), \
                patch.object(updates, "_convert_wespeaker", side_effect=convert), \
                patch.object(updates, "validate_model"):
            target = updates.install_model(spec, self.base, plan)
        self.assertIsNotNone(updates._read_provenance(spec, target))
        with patch.object(updates, "repository_manifest", return_value=plan):
            self.assertEqual(updates.check_update(spec, target).status, "up_to_date")
            (target / "wespeaker_fixed.bin").write_bytes(b"bad bin!")
            self.assertEqual(updates.check_update(spec, target).status, "unknown")

    def test_existing_version_is_reused_without_mutation_or_download(self):
        target = self.install()
        original = {p.name: p.read_bytes() for p in target.iterdir()}
        plan = replace(self.plan, revision="b" * 40)
        with patch.object(updates.urllib.request, "urlopen") as request, \
                patch.object(updates, "validate_model"):
            again = updates.install_model(self.spec, self.base, plan, current_path=target)
        request.assert_not_called()
        self.assertEqual(again, target)
        self.assertEqual(original, {p.name: p.read_bytes() for p in target.iterdir()})

    def test_corrupt_active_version_is_repaired_to_different_directory(self):
        target = self.install()
        (target / "model.bin").write_bytes(b"bad")
        new = self.install(current_path=target)
        self.assertNotEqual(target, new)
        self.assertIn("-repair-", new.name)
        self.assertEqual((target / "model.bin").read_bytes(), b"bad")
        self.assertEqual((new / "model.bin").read_bytes(), self.payloads["model.bin"])

    def test_disk_check_uses_actual_manifest_size_and_precedes_download(self):
        with patch.object(updates.shutil, "disk_usage", return_value=types.SimpleNamespace(free=64 * 1024 * 1024)), \
                patch.object(updates.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(RuntimeError, "места"):
                updates.install_model(self.spec, self.base, self.plan)
        request.assert_not_called()

    def test_same_model_installation_cannot_run_concurrently(self):
        with updates._installation_lock(self.base, self.spec):
            with self.assertRaises(updates.InstallationBusy):
                with updates._installation_lock(self.base, self.spec):
                    self.fail("second installation acquired lock")
        with updates._installation_lock(self.base, self.spec):
            pass

    def test_cancel_before_start_has_no_network_or_filesystem_effects(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(updates, "repository_manifest") as manifest:
            with self.assertRaises(updates.DownloadCancelled):
                updates.install_model(self.spec, self.base / "unused", cancel=cancel)
        manifest.assert_not_called()
        self.assertFalse((self.base / "unused").exists())

    def test_invalid_plan_does_not_download(self):
        with patch.object(updates.urllib.request, "urlopen") as request:
            for plan in (replace(self.plan, repository="Other/repo"),
                         replace(self.plan, revision="main"),
                         replace(self.plan, fingerprint="0" * 64),
                         replace(self.plan, files=self.files[:1])):
                with self.assertRaises(ValueError):
                    updates.install_model(self.spec, self.base, plan)
        request.assert_not_called()

    def test_gpu_validator_uses_exact_pipeline_and_device(self):
        current = self.write_model(self.base / "current")
        genai = types.SimpleNamespace(WhisperPipeline=Mock(), LLMPipeline=Mock())
        with patch.dict("sys.modules", {"openvino_genai": genai}):
            updates.validate_model(self.spec, current)
            updates.validate_model(replace(self.spec, kind="llm"), current)
        genai.WhisperPipeline.assert_called_once_with(str(current), "GPU")
        genai.LLMPipeline.assert_called_once_with(str(current), "GPU")

    def test_gpu_validator_rejects_tampered_marked_model_before_load(self):
        current = self.write_model(self.base / "current", marker=True)
        (current / "model.bin").write_bytes(b"X" * 12)
        with patch.dict("sys.modules", {"openvino_genai": types.SimpleNamespace(WhisperPipeline=Mock())}):
            with self.assertRaisesRegex(RuntimeError, "Контрольные суммы"):
                updates.validate_model(self.spec, current)

    def test_unsafe_artifact_paths_rejected(self):
        for name in ("../secret", "C:/secret", "/secret", "a\\b", "a/../b"):
            with self.assertRaises(ValueError):
                updates.Artifact(name, 1, "a" * 64)


if __name__ == "__main__":
    unittest.main()
