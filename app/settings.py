import json
import os
import sys
from pathlib import Path


def _model_store_folder():
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / "WhisperGUI" / "models"


def _find_project_root():
    """Find the folder that contains the external neural-network models."""
    if getattr(sys, "frozen", False):
        start = Path(sys.executable).resolve().parent
        candidates = [start, *list(start.parents)[:3]]
    else:
        candidates = [Path(__file__).resolve().parent.parent]

    model_folders = ("turbo-gpu", "diarization", "llm-models")
    scored = [
        (sum((candidate / folder).is_dir() for folder in model_folders), candidate)
        for candidate in candidates
    ]
    best_score = max(score for score, _ in scored)
    return next(candidate for score, candidate in scored if score == best_score)


PROJECT_ROOT = _find_project_root()
MODEL_STORE_FOLDER = _model_store_folder()


def _bundled_embedding_model():
    """Find the compact WeSpeaker IR included with a frozen release."""
    candidates = []
    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        candidates.append(bundle_root / "diarization" / "wespeaker_fixed.xml")
        candidates.append(Path(sys.executable).resolve().parent / "diarization" / "wespeaker_fixed.xml")
    candidates.append(PROJECT_ROOT / "diarization" / "wespeaker_fixed.xml")
    for xml_path in candidates:
        if xml_path.is_file() and xml_path.with_suffix(".bin").is_file():
            return xml_path
    return MODEL_STORE_FOLDER / "diarization" / "wespeaker_fixed.xml"


def _existing_or_store(folder):
    candidate = PROJECT_ROOT / folder
    return candidate if candidate.is_dir() else MODEL_STORE_FOLDER / folder


def _usable_llm_folder(folder):
    path = Path(folder)
    return all((path / name).is_file() for name in (
        "openvino_model.xml", "openvino_model.bin", "tokenizer.json"
    ))


DEFAULT_MODELS_FOLDER = _existing_or_store("turbo-gpu")
DEFAULT_LLM_FOLDER = _existing_or_store("llm-models")
DEFAULT_EMBEDDING_MODEL = _bundled_embedding_model()

DEFAULT_SETTINGS = {
    "model_store_folder": str(MODEL_STORE_FOLDER),
    "models_folder": str(DEFAULT_MODELS_FOLDER),
    "last_model": "",
    "device": "GPU",
    "chunk_minutes": 5,
    "save_mode": "next_to_source",
    "save_folder": "",
    "filename_template": "{original_name}_transcript.txt",
    "multi_mode": False,
    "keep_extracted_audio": False,
    "llm_models_folder": str(DEFAULT_LLM_FOLDER),
    "llm_model_path": str(DEFAULT_LLM_FOLDER / "Qwen3-8B-int4-ov"),
    "last_transcript_path": "",
    "last_prompt_id": "meeting_protocol",
    "ai_output_format": "md",
    "auto_ai_processing": False,
    "language": "ru",
    "task": "transcribe",
    "enable_diarization": True,
    "embed_model_path": str(DEFAULT_EMBEDDING_MODEL),
    "diar_threshold": 0.74,
    "num_speakers": 0,
    "max_auto_speakers": 8,
    "prompt_speaker_names": True,
    "speaker_names": [],
    "settings_version": 7
}

class Settings:
    def __init__(self):
        self.config_dir = Path(os.environ.get("APPDATA", Path.home())) / "WhisperGUI"
        self.config_file = self.config_dir / "settings.json"
        self.data = DEFAULT_SETTINGS.copy()
        self.load()

    def load(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    user_data = json.load(f)
                self.data.update(user_data)
                if int(user_data.get("settings_version", 1)) < 2:
                    self.data["diar_threshold"] = DEFAULT_SETTINGS["diar_threshold"]
                    self.data["enable_diarization"] = True
                    self.data["last_model"] = ""
                    self.data["settings_version"] = 2
                if int(user_data.get("settings_version", 1)) < 3:
                    self.data["keep_extracted_audio"] = False
                    self.data["settings_version"] = 3
                if int(user_data.get("settings_version", 1)) < 4:
                    self.data["llm_models_folder"] = DEFAULT_SETTINGS["llm_models_folder"]
                    self.data["llm_model_path"] = DEFAULT_SETTINGS["llm_model_path"]
                    self.data["last_prompt_id"] = "meeting_protocol"
                    self.data["ai_output_format"] = "md"
                    self.data["auto_ai_processing"] = False
                    self.data["settings_version"] = 4
                if int(user_data.get("settings_version", 1)) < 5:
                    self.data["prompt_speaker_names"] = True
                    self.data["settings_version"] = 5
                if int(user_data.get("settings_version", 1)) < 6:
                    self.data["max_auto_speakers"] = 8
                    self.data["speaker_names"] = []
                    self.data["settings_version"] = 6
                if int(user_data.get("settings_version", 1)) < 7:
                    self.data["model_store_folder"] = str(MODEL_STORE_FOLDER)
                    self.data["settings_version"] = 7
            except Exception as e:
                print(f"Ошибка загрузки настроек: {e}")
        # Migrate stale paths from the former C:\Whisper installation.
        if not os.path.isdir(self.data.get("models_folder", "")):
            installed = Path(self.data.get("model_store_folder") or MODEL_STORE_FOLDER) / "turbo-gpu"
            self.data["models_folder"] = str(installed if installed.is_dir() else DEFAULT_MODELS_FOLDER)
        embed_path = Path(self.data.get("embed_model_path") or "")
        if not (embed_path.is_file() and embed_path.with_suffix(".bin").is_file()):
            self.data["embed_model_path"] = str(DEFAULT_EMBEDDING_MODEL)
        if not os.path.isdir(self.data.get("llm_models_folder", "")):
            installed = Path(self.data.get("model_store_folder") or MODEL_STORE_FOLDER) / "llm-models"
            self.data["llm_models_folder"] = str(installed if installed.is_dir() else DEFAULT_LLM_FOLDER)
        if not _usable_llm_folder(self.data.get("llm_model_path", "")):
            self.data["llm_model_path"] = str(
                Path(self.data["llm_models_folder"]) / "Qwen3-8B-int4-ov"
            )
        self.data["device"] = "GPU"

    def save(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()
