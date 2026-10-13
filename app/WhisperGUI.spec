# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from importlib.util import find_spec

import PySide6
from PyInstaller.utils.hooks import collect_all


datas = [
    ("README.md", "."),
    ("RELEASE_README.txt", "."),
    ("THIRD_PARTY_NOTICES.txt", "."),
    ("requirements.txt", "."),
    (str(Path(SPECPATH).resolve().parent / "diarization" / "wespeaker_fixed.xml"), "diarization"),
    (str(Path(SPECPATH).resolve().parent / "diarization" / "wespeaker_fixed.bin"), "diarization"),
]
binaries = []
hiddenimports = []

# OpenVINO uses runtime-loaded DLLs/plugins, while imageio-ffmpeg stores its
# executable as package data. Collecting these packages explicitly makes the
# release independent of the development Python installation.
for package in ("openvino", "openvino_genai", "imageio_ffmpeg"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

# GenAI loads this extension by bare DLL name at pipeline construction time.
# PyInstaller's static analysis does not see the load, so place it in its
# bootloader DLL search directory.
tokenizer_module = find_spec("openvino_tokenizers")
if tokenizer_module is None or tokenizer_module.origin is None:
    raise RuntimeError("openvino_tokenizers package is required for the release")
tokenizer_dll = Path(tokenizer_module.origin).resolve().parent / "lib" / "openvino_tokenizers.dll"
if not tokenizer_dll.is_file():
    raise RuntimeError(f"OpenVINO tokenizer extension is missing: {tokenizer_dll}")
binaries.append((str(tokenizer_dll), "."))

# Python 3.10 ships an older VC runtime in its root directory. Qt 6.11 was
# built with a newer compatible runtime and otherwise Windows loads the old
# DLL first, causing "procedure not found" while importing QtCore.
pyside_dir = Path(PySide6.__file__).resolve().parent
for runtime_name in (
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll",
    "MSVCP140.dll",
    "MSVCP140_1.dll",
    "MSVCP140_2.dll",
    "MSVCP140_CODECVT_IDS.dll",
    "concrt140.dll",
):
    runtime_path = pyside_dir / runtime_name
    if runtime_path.is_file():
        binaries.append((str(runtime_path), "."))

hiddenimports += [
    "soundfile",
    "sklearn.cluster._agglomerative",
    "librosa.core",
    "librosa.feature",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "matplotlib", "IPython", "notebook"],
    noarchive=False,
    optimize=0,
)

# The development shell may expose unrelated Poppler/ICU DLLs on PATH. They
# must not be bundled: Qt 6 uses Windows' ICU API, while Poppler's ICU exports
# version-suffixed symbols and prevents QtWidgets from loading.
a.binaries = [
    entry for entry in a.binaries
    if "codex-runtimes" not in str(entry[1]).lower()
    and Path(entry[0]).name.lower() not in {"icuuc.dll", "icudt78.dll"}
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version="version_info.txt",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="WhisperGUI",
)
