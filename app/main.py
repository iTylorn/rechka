import ctypes
import os
import sys
import traceback
from pathlib import Path

from app_version import APP_VERSION


def _write_startup_error(smoke_test):
    base_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperGUI"
    error_path = base_dir / "startup_error.log"
    details = traceback.format_exc()
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        error_path.write_text(details, encoding="utf-8")
    except OSError:
        pass
    if os.name == "nt" and not smoke_test:
        ctypes.windll.user32.MessageBoxW(
            0,
            f"Не удалось запустить WhisperGUI.\n\nПодробности: {error_path}",
            "Ошибка запуска WhisperGUI",
            0x10,
        )


def main():
    smoke_test = "--smoke-test" in sys.argv
    gpu_smoke_test = "--gpu-smoke-test" in sys.argv
    llm_smoke_test = "--llm-smoke-test" in sys.argv
    try:
        if gpu_smoke_test:
            from openvino_genai import WhisperPipeline
            from openvino import Core
            from media_loader import MediaLoader
            from model_manager import ModelManager
            from settings import DEFAULT_EMBEDDING_MODEL, Settings

            if not MediaLoader().find_ffmpeg():
                raise RuntimeError("Встроенный FFmpeg не найден")
            core = Core()
            if "GPU" not in core.available_devices:
                raise RuntimeError("OpenVINO не видит GPU")
            model = core.read_model(str(DEFAULT_EMBEDDING_MODEL))
            core.compile_model(model, "GPU")
            whisper_models = ModelManager.find_models(Settings().get("models_folder", ""))
            if whisper_models:
                pipeline = WhisperPipeline(whisper_models[0], "GPU")
                del pipeline
            return 0

        if llm_smoke_test:
            from openvino_genai import LLMPipeline

            index = sys.argv.index("--llm-smoke-test")
            if index + 1 >= len(sys.argv):
                raise ValueError("Укажите папку модели после --llm-smoke-test")
            pipeline = LLMPipeline(sys.argv[index + 1], "GPU")
            pipeline.get_tokenizer()
            del pipeline
            return 0

        from PySide6.QtWidgets import QApplication
        from main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("WhisperGUI")
        app.setApplicationVersion(APP_VERSION)
        window = MainWindow()
        if smoke_test:
            # Import and construct the complete GUI, then return without entering
            # the interactive event loop. This is used to validate packaged builds.
            app.processEvents()
            window.close()
            return 0
        window.show()
        return app.exec()
    except Exception:
        _write_startup_error(smoke_test or gpu_smoke_test or llm_smoke_test)
        return 1

if __name__ == "__main__":
    sys.exit(main())
