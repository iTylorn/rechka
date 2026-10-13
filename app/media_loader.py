import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma",
    ".aac", ".opus", ".aiff", ".aif",
}
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv",
    ".m4v", ".mpeg", ".mpg", ".flv", ".ts", ".mts", ".m2ts",
}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class MediaCancelled(Exception):
    pass


@dataclass
class PreparedMedia:
    source_path: str
    audio_path: str
    is_video: bool
    temporary: bool = False

    def cleanup(self):
        if self.temporary and os.path.isfile(self.audio_path):
            try:
                os.remove(self.audio_path)
            except OSError:
                pass


class MediaLoader:
    def __init__(self, app_dir=None):
        self.app_dir = Path(app_dir or Path(__file__).resolve().parent)
        self._active_process = None

    @staticmethod
    def is_supported(path):
        return Path(path).suffix.lower() in MEDIA_EXTENSIONS

    @staticmethod
    def is_video(path):
        return Path(path).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def dialog_filter():
        audio = " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
        video = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
        media = " ".join(f"*{ext}" for ext in sorted(MEDIA_EXTENSIONS))
        return (
            f"Аудио и видео ({media});;"
            f"Видеофайлы ({video});;"
            f"Аудиофайлы ({audio});;"
            "Все файлы (*.*)"
        )

    def find_ffmpeg(self):
        local_names = ("ffmpeg.exe", "ffmpeg")
        for name in local_names:
            candidate = self.app_dir / "tools" / name
            if candidate.is_file():
                return str(candidate)

        try:
            import imageio_ffmpeg
            candidate = imageio_ffmpeg.get_ffmpeg_exe()
            if candidate and os.path.isfile(candidate):
                return candidate
        except Exception:
            pass

        return shutil.which("ffmpeg")

    def cancel(self):
        process = self._active_process
        if process is not None and process.poll() is None:
            process.terminate()

    def prepare(self, source_path, keep_extracted_audio=False,
                cancel_check=None, log=None):
        source_path = os.path.abspath(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Файл не найден: {source_path}")
        if not self.is_supported(source_path):
            raise ValueError(f"Неподдерживаемый формат: {Path(source_path).suffix or 'без расширения'}")
        if not self.is_video(source_path):
            return PreparedMedia(source_path, source_path, is_video=False)

        ffmpeg = self.find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg не найден. Установите пакет imageio-ffmpeg или поместите "
                "ffmpeg.exe в папку программы tools."
            )

        if keep_extracted_audio:
            output_path = self._unique_audio_path(source_path)
            temporary = False
        else:
            descriptor, output_path = tempfile.mkstemp(prefix="whisper_video_", suffix=".wav")
            os.close(descriptor)
            temporary = True

        if log:
            destination = output_path if keep_extracted_audio else "временный WAV"
            log(f"Извлечение звуковой дорожки: {destination}")

        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", source_path, "-map", "0:a:0", "-vn",
            "-ar", "16000", "-ac", "2", "-c:a", "pcm_s16le",
            output_path,
        ]
        started = time.time()
        try:
            self._active_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            while True:
                if cancel_check and cancel_check():
                    self.cancel()
                    self._active_process.wait(timeout=5)
                    raise MediaCancelled("Извлечение звука остановлено пользователем")
                try:
                    _, stderr = self._active_process.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue

            if self._active_process.returncode != 0:
                details = stderr.decode("utf-8", errors="replace").strip()
                if "matches no streams" in details or "does not contain any stream" in details:
                    raise RuntimeError("В видеофайле не найдена звуковая дорожка")
                raise RuntimeError(f"FFmpeg не смог извлечь звук: {details or 'неизвестная ошибка'}")
            if not os.path.isfile(output_path) or os.path.getsize(output_path) < 44:
                raise RuntimeError("FFmpeg не создал корректный аудиофайл")
            if log:
                log(f"Звуковая дорожка извлечена за {time.time() - started:.1f} сек.")
            return PreparedMedia(source_path, output_path, is_video=True, temporary=temporary)
        except Exception:
            if os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise
        finally:
            self._active_process = None

    @staticmethod
    def _unique_audio_path(source_path):
        source = Path(source_path)
        candidate = source.with_name(f"{source.stem}_audio.wav")
        counter = 2
        while candidate.exists():
            candidate = source.with_name(f"{source.stem}_audio_{counter}.wav")
            counter += 1
        return str(candidate)
