"""Verified, resumable model updates, independent of Qt and application settings.

Nothing contacts the network at import time. A check pins both the Hub revision
and every required artifact hash. Installation stages that exact plan, validates
it on GPU, and returns an immutable directory; the caller owns activation and
rollback. Existing model directories are never modified.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import urllib.parse
import urllib.request
import uuid


MARKER_NAME = "_whispergui_manifest.json"
WESPEAKER_XML = "wespeaker_fixed.xml"
CONVERSION_RECIPE = "wespeaker-feats-1x200x80-fp16-v1"
USER_AGENT = "WhisperGUI/2.0-beta model updates"
WHISPER_FILES = (
    "added_tokens.json", "config.json", "generation_config.json", "merges.txt",
    "normalizer.json", "openvino_config.json", "openvino_decoder_model.bin",
    "openvino_decoder_model.xml", "openvino_detokenizer.bin",
    "openvino_detokenizer.xml", "openvino_encoder_model.bin",
    "openvino_encoder_model.xml", "openvino_tokenizer.bin",
    "openvino_tokenizer.xml", "preprocessor_config.json",
    "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json",
)
QWEN_FILES = (
    "added_tokens.json", "chat_template.jinja", "config.json",
    "generation_config.json", "merges.txt", "openvino_config.json",
    "openvino_detokenizer.bin", "openvino_detokenizer.xml",
    "openvino_model.bin", "openvino_model.xml", "openvino_tokenizer.bin",
    "openvino_tokenizer.xml", "special_tokens_map.json", "tokenizer_config.json",
    "tokenizer.json", "vocab.json",
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    repo: str
    folder: str
    files: tuple[str, ...]
    approximate_size: int
    kind: str = "whisper"

    @property
    def relative_dir(self):
        category = {"whisper": "turbo-gpu", "llm": "llm-models",
                    "wespeaker": "diarization"}[self.kind]
        return Path(category) / self.folder


MODEL_CATALOG = (
    ModelSpec("whisper", "Whisper large-v3-turbo INT8 — ~0,83 ГБ",
              "OpenVINO/whisper-large-v3-turbo-int8-ov",
              "whisper-large-v3-turbo-int8-ov", WHISPER_FILES, 828_087_998),
    ModelSpec("whisper_fp16", "Whisper large-v3-turbo FP16 — ~1,63 ГБ",
              "OpenVINO/whisper-large-v3-turbo-fp16-ov",
              "whisper-large-v3-turbo-fp16-ov",
              tuple(n for n in WHISPER_FILES if n != "openvino_config.json"),
              1_627_649_856),
    ModelSpec("qwen8", "Qwen3-8B INT4 — ~4,88 ГБ",
              "OpenVINO/Qwen3-8B-int4-ov", "Qwen3-8B-int4-ov",
              QWEN_FILES, 4_882_858_414, "llm"),
    ModelSpec("qwen14", "Qwen3-14B INT4 — ~9,73 ГБ",
              "OpenVINO/Qwen3-14B-int4-ov", "Qwen3-14B-int4-ov",
              tuple(n for n in QWEN_FILES
                    if n not in ("chat_template.jinja", "openvino_config.json")),
              9_731_513_980, "llm"),
    ModelSpec("wespeaker", "WeSpeaker ECAPA-TDNN512 LM — ~25 МБ",
              "Wespeaker/wespeaker-ecapa-tdnn512-LM", "wespeaker-ecapa-tdnn512-LM",
              ("voxceleb_ECAPA512_LM.onnx",), 24_861_931, "wespeaker"),
)


class DownloadCancelled(Exception):
    """Cancellation retains resumable bytes and never activates a model."""


class InstallationBusy(RuntimeError):
    pass


def _check_cancel(cancel):
    if cancel and (cancel.is_set() if hasattr(cancel, "is_set") else cancel()):
        raise DownloadCancelled()


def _safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(p in (".", "..", "") for p in name.split("/"))):
        raise ValueError("Недопустимое имя файла модели.")
    return name


@dataclass(frozen=True)
class Artifact:
    name: str
    size: int
    sha256: str | None = None
    git_sha1: str | None = None

    def __post_init__(self):
        _safe_name(self.name)
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size <= 0:
            raise ValueError(f"Неизвестный размер файла {self.name}.")
        if not self.sha256 and not self.git_sha1:
            raise ValueError(f"Нет контрольной суммы файла {self.name}.")
        for digest, length in ((self.sha256, 64), (self.git_sha1, 40)):
            if digest is not None and not re.fullmatch(rf"[a-f0-9]{{{length}}}", digest):
                raise ValueError(f"Неверная контрольная сумма файла {self.name}.")


def _fingerprint(spec, files):
    content = {"repository": spec.repo,
               "files": [asdict(a) for a in sorted(files, key=lambda a: a.name)],
               "conversion": CONVERSION_RECIPE if spec.kind == "wespeaker" else None}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UpdatePlan:
    key: str
    repository: str
    revision: str
    files: tuple[Artifact, ...]
    fingerprint: str

    @property
    def total_size(self):
        return sum(a.size for a in self.files)


@dataclass(frozen=True)
class UpdateResult:
    status: str  # not_installed, up_to_date, update_available, unknown
    current_revision: str | None
    latest_revision: str
    files: tuple[Artifact, ...]
    plan: UpdatePlan
    detail: str


def _request_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)


def repository_manifest(spec, cancel=None):
    """Fetch required artifacts from one commit, never from a moving branch."""
    _check_cancel(cancel)
    repo = urllib.parse.quote(spec.repo, safe="/")
    info = _request_json(f"https://huggingface.co/api/models/{repo}")
    revision = info.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise RuntimeError(f"Hugging Face не сообщил ревизию {spec.repo}.")
    _check_cancel(cancel)
    # The small official catalogs keep every required artifact at the root.
    # Missing entries fail closed rather than creating a partial model plan.
    tree = _request_json(
        f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=false"
    )
    by_path = {item.get("path"): item for item in tree if item.get("type") == "file"}
    missing = sorted(set(spec.files) - set(by_path))
    if missing:
        raise RuntimeError(f"В {spec.repo} отсутствуют файлы: {', '.join(missing)}")
    artifacts = []
    for name in spec.files:
        entry = by_path[name]
        lfs = entry.get("lfs") or {}
        digest = lfs.get("oid") if isinstance(lfs, dict) else None
        if digest and digest.startswith("sha256:"):
            digest = digest[7:]
        artifacts.append(Artifact(name, entry.get("size"), digest,
                                  entry.get("oid") if not digest else None))
    _check_cancel(cancel)
    files = tuple(artifacts)
    return UpdatePlan(spec.key, spec.repo, revision, files, _fingerprint(spec, files))


def _validate_plan(spec, plan):
    if (plan.key != spec.key or plan.repository != spec.repo
            or not re.fullmatch(r"[a-f0-9]{40}", plan.revision)
            or len(plan.files) != len(spec.files)
            or {a.name for a in plan.files} != set(spec.files)
            or plan.fingerprint != _fingerprint(spec, plan.files)):
        raise ValueError("План установки не соответствует выбранной модели.")
    _safe_name(spec.folder)
    _safe_name(spec.key)


def _valid_file(path, artifact, cancel=None):
    _check_cancel(cancel)
    if not path.is_file() or path.stat().st_size != artifact.size:
        return False
    digest = hashlib.sha256() if artifact.sha256 else hashlib.sha1()
    if not artifact.sha256:
        digest.update(f"blob {artifact.size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            _check_cancel(cancel)
            digest.update(block)
    return digest.hexdigest() == (artifact.sha256 or artifact.git_sha1)


def _model_dir(spec, path):
    path = Path(path)
    return path.parent if spec.kind == "wespeaker" and path.suffix.lower() == ".xml" else path


def installed_model_complete(spec, path):
    """Cheap offline presence check, not an integrity or GPU certificate."""
    if not path:
        return False
    directory = _model_dir(spec, path)
    names = (WESPEAKER_XML, "wespeaker_fixed.bin") if spec.kind == "wespeaker" else spec.files
    if spec.kind == "wespeaker" and Path(path).suffix.lower() == ".xml":
        names = (Path(path).name, Path(path).with_suffix(".bin").name)
    try:
        return all((directory / name).is_file() and (directory / name).stat().st_size > 0
                   for name in names)
    except OSError:
        return False


def _read_provenance(spec, directory, cancel=None):
    """Return only provenance whose recorded source and output bytes still match."""
    try:
        marker = json.loads((directory / MARKER_NAME).read_text(encoding="utf-8"))
        files = tuple(Artifact(**item) for item in marker["files"])
        if (marker.get("repository") != spec.repo
                or len(files) != len(spec.files)
                or {a.name for a in files} != set(spec.files)
                or not re.fullmatch(r"[a-f0-9]{40}", marker.get("revision", ""))):
            return None
        if not all(_valid_file(directory / a.name, a, cancel) for a in files):
            return None
        if spec.kind == "wespeaker":
            generated = tuple(Artifact(**item) for item in marker.get("generated_files", []))
            if (marker.get("conversion") != CONVERSION_RECIPE
                    or {a.name for a in generated} != {WESPEAKER_XML, "wespeaker_fixed.bin"}
                    or len(generated) != 2
                    or not all(_valid_file(directory / a.name, a, cancel) for a in generated)):
                return None
        return marker, _fingerprint(spec, files)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def check_update(spec, installed_path=None, cancel=None):
    plan = repository_manifest(spec, cancel)
    _validate_plan(spec, plan)
    current = None
    if not installed_path or not _model_dir(spec, installed_path).is_dir():
        status, detail = "not_installed", "Модель не установлена."
    else:
        directory = _model_dir(spec, installed_path)
        provenance = _read_provenance(spec, directory, cancel)
        if provenance:
            marker, fingerprint = provenance
            current = marker["revision"]
            status = "up_to_date" if fingerprint == plan.fingerprint else "update_available"
            detail = ("Файлы модели актуальны." if status == "up_to_date"
                      else "Доступны новые файлы модели.")
        elif spec.kind != "wespeaker" and all(
                _valid_file(directory / a.name, a, cancel) for a in plan.files):
            current, status = plan.revision, "up_to_date"
            detail = "Установленная модель опознана по контрольным суммам; файлы актуальны."
        else:
            status = "unknown"
            detail = ("Происхождение или целостность локальной модели не подтверждены. "
                      "Можно установить проверенную версию, сохранив прежнюю.")
    _check_cancel(cancel)
    return UpdateResult(status, current, plan.revision, plan.files, plan, detail)


@contextmanager
def _installation_lock(base, spec):
    lock_dir = base / "_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    # Keep this inode: unlinking a lock file permits two processes to own
    # different inodes for the same logical lock. OS locks release on crashes.
    with (lock_dir / f"{spec.key}.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise InstallationBusy("Эта модель уже устанавливается в другом окне.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _download_file(spec, plan, artifact, stage, report, cancel):
    final = stage / artifact.name
    if _valid_file(final, artifact, cancel):
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > artifact.size:
        partial.unlink()
        offset = 0
    if offset == artifact.size:
        if _valid_file(partial, artifact, cancel):
            os.replace(partial, final)
            return
        partial.unlink()
        report(-offset, artifact.name)
        offset = 0
    url = (f"https://huggingface.co/{urllib.parse.quote(spec.repo, safe='/')}"
           f"/resolve/{plan.revision}/{urllib.parse.quote(artifact.name, safe='/')}")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    _check_cancel(cancel)
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status not in (200, 206):
            raise RuntimeError(f"Неожиданный ответ сервера: {response.status}.")
        if response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                 response.headers.get("Content-Range", ""))
            if (not match or int(match[1]) != offset or int(match[3]) != artifact.size
                    or int(match[2]) != artifact.size - 1):
                raise RuntimeError(f"Сервер вернул неверный диапазон для {artifact.name}.")
        elif offset:
            report(-offset, artifact.name)
            offset = 0
        written = offset
        with partial.open("ab" if offset else "wb") as output:
            while True:
                _check_cancel(cancel)
                block = response.read(1024 * 1024)
                if not block:
                    break
                if written + len(block) > artifact.size:
                    raise RuntimeError(f"Сервер превысил размер файла {artifact.name}.")
                output.write(block)
                written += len(block)
                report(len(block), artifact.name)
    _check_cancel(cancel)
    if not _valid_file(partial, artifact, cancel):
        actual = partial.stat().st_size
        if actual == artifact.size:
            partial.unlink()  # Retry must not resume a complete corrupt file.
        raise RuntimeError(f"Файл {artifact.name} не прошёл проверку "
                           f"({actual:,} из {artifact.size:,} байт). "
                           "Повторите установку для докачки.")
    os.replace(partial, final)


def _convert_wespeaker(spec, stage, cancel):
    _check_cancel(cancel)
    import openvino as ov
    model = ov.convert_model(str(stage / spec.files[0]), input=[("feats", [1, 200, 80])])
    _check_cancel(cancel)
    ov.save_model(model, str(stage / WESPEAKER_XML), compress_to_fp16=True)
    _check_cancel(cancel)


def validate_model(spec, path, cancel=None):
    """Load the actual application pipeline on GPU; never fall back to CPU."""
    _check_cancel(cancel)
    directory = _model_dir(spec, path)
    if not installed_model_complete(spec, path):
        raise RuntimeError("Модель неполна: отсутствуют необходимые файлы.")
    if (directory / MARKER_NAME).is_file() and not _read_provenance(spec, directory, cancel):
        raise RuntimeError("Контрольные суммы установленной модели не совпадают с паспортом.")
    if spec.kind == "wespeaker":
        import numpy as np
        import openvino as ov
        xml = Path(path) if Path(path).suffix.lower() == ".xml" else directory / WESPEAKER_XML
        core = ov.Core()
        model = core.read_model(str(xml))
        if (len(model.inputs) != 1 or len(model.outputs) != 1
                or list(model.input(0).shape) != [1, 200, 80]
                or list(model.output(0).shape) != [1, 192]
                or model.input(0).element_type != ov.Type.f32):
            raise RuntimeError("WeSpeaker должен принимать фиксированный вход [1, 200, 80].")
        compiled = core.compile_model(model, "GPU")
        for features in (np.zeros((1, 200, 80), dtype=np.float32),
                         np.random.default_rng(0).normal(size=(1, 200, 80)).astype(np.float32)):
            _check_cancel(cancel)
            output = np.asarray(compiled({compiled.input(0): features})[compiled.output(0)])
            if (output.shape != (1, 192) or not np.isfinite(output).all()
                    or np.linalg.norm(output) < 1e-8):
                raise RuntimeError("WeSpeaker вернул некорректный результат проверки на GPU.")
    elif spec.kind in ("whisper", "llm"):
        import openvino_genai
        factory = (openvino_genai.WhisperPipeline if spec.kind == "whisper"
                   else openvino_genai.LLMPipeline)
        pipeline = factory(str(directory), "GPU")
        del pipeline
    else:
        raise ValueError(f"Неизвестный вид модели: {spec.kind}")
    _check_cancel(cancel)


def _generated_artifacts(spec, stage, cancel):
    if spec.kind != "wespeaker":
        return ()
    artifacts = []
    for name in (WESPEAKER_XML, "wespeaker_fixed.bin"):
        path = stage / name
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                _check_cancel(cancel)
                digest.update(block)
        artifacts.append(Artifact(name, path.stat().st_size, digest.hexdigest()))
    return tuple(artifacts)


def install_model(spec, base_folder, plan=None, current_path=None, progress=None, cancel=None):
    """Return a verified version directory; activation/settings are caller-owned.

    `progress(percent, text)` and `cancel() -> bool` run on the calling thread.
    `cancel` may also be a threading.Event.
    Cancellation is checked around GPU compilation (the native compilation call
    itself cannot be interrupted safely). Retrying reuses revision-scoped parts.
    """
    progress = progress or (lambda percent, text: None)
    _check_cancel(cancel)
    plan = plan or repository_manifest(spec, cancel)
    _validate_plan(spec, plan)
    base = Path(base_folder).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    with _installation_lock(base, spec):
        target = base / "_versions" / spec.key / plan.fingerprint[:24]
        stage = base / "_downloads" / spec.folder / f"{plan.revision}-{plan.fingerprint[:24]}"
        if current_path:
            current = _model_dir(spec, current_path).resolve()
            if stage == current or stage in current.parents:
                raise ValueError("Активная модель не может быть папкой временной загрузки.")
        else:
            current = None
        if target.exists():
            provenance = _read_provenance(spec, target, cancel)
            if provenance and provenance[1] == plan.fingerprint:
                progress(96, "Проверка уже загруженной версии на GPU…")
                validate_model(spec, target, cancel)
                progress(100, "Модель проверена и готова.")
                return target
            # A corrupt/unmanaged directory may still be selected by a user.
            # Repair to a new directory, never delete or replace that version.
            target = target.with_name(target.name + "-repair-" + uuid.uuid4().hex[:8])
        stage.mkdir(parents=True, exist_ok=True)
        completed = 0
        for artifact in plan.files:
            if _valid_file(stage / artifact.name, artifact, cancel):
                completed += artifact.size
            else:
                partial = stage / (artifact.name + ".part")
                if partial.is_file() and partial.stat().st_size <= artifact.size:
                    completed += partial.stat().st_size
        conversion_space = plan.total_size * 2 if spec.kind == "wespeaker" else 0
        needed = plan.total_size - completed + conversion_space + 64 * 1024 * 1024
        if shutil.disk_usage(base).free < needed:
            raise RuntimeError(f"Недостаточно свободного места: требуется ещё "
                               f"{needed / 1e9:.2f} ГБ с учётом временных файлов.")

        def report(count, filename):
            nonlocal completed
            completed = max(0, min(plan.total_size, completed + count))
            progress(min(94, int(94 * completed / plan.total_size)),
                     f"{spec.folder}: {filename} — "
                     f"{completed / 1e9:.2f} / {plan.total_size / 1e9:.2f} ГБ")

        for artifact in plan.files:
            _check_cancel(cancel)
            staged = stage / artifact.name
            if _valid_file(staged, artifact, cancel):
                continue
            if current and _valid_file(current / artifact.name, artifact, cancel):
                # Copy into an inactive staging path; hard links could permit a
                # future edit in staging to modify the active model's bytes.
                staged.parent.mkdir(parents=True, exist_ok=True)
                partial = staged.with_name(staged.name + ".part")
                previous = partial.stat().st_size if partial.is_file() else 0
                with (current / artifact.name).open("rb") as source, partial.open("wb") as output:
                    report(-min(previous, artifact.size), artifact.name)
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        _check_cancel(cancel)
                        output.write(block)
                        report(len(block), artifact.name)
                if not _valid_file(partial, artifact, cancel):
                    raise RuntimeError("Локальный файл изменился во время копирования.")
                os.replace(partial, staged)
            else:
                _download_file(spec, plan, artifact, stage, report, cancel)
        _check_cancel(cancel)
        if spec.kind == "wespeaker":
            progress(95, "Преобразование WeSpeaker для входа [1, 200, 80]…")
            _convert_wespeaker(spec, stage, cancel)
        progress(96, "Проверка совместимости модели с GPU…")
        # An earlier cancelled publication may have left a valid old marker.
        # Regenerate conversion provenance only after validating this attempt.
        old_marker = stage / MARKER_NAME
        if old_marker.exists():
            old_marker.unlink()
        validate_model(spec, stage, cancel)
        generated = _generated_artifacts(spec, stage, cancel)
        marker = {"schema": 2, "key": spec.key, "repository": spec.repo,
                  "revision": plan.revision, "fingerprint": plan.fingerprint,
                  "files": [asdict(a) for a in plan.files],
                  "generated_files": [asdict(a) for a in generated],
                  "conversion": CONVERSION_RECIPE if spec.kind == "wespeaker" else None,
                  "validation": {"device": "GPU", "kind": spec.kind,
                                 "utc": datetime.now(timezone.utc).isoformat()}}
        marker_temp = stage / (MARKER_NAME + ".tmp")
        marker_temp.write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(marker_temp, stage / MARKER_NAME)
        _check_cancel(cancel)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Same-volume rename is the sole publication step; a .part directory is
        # never returned to the caller or made active.
        stage.rename(target)
        progress(100, f"Модель установлена и проверена на GPU: {spec.title}")
        return target
