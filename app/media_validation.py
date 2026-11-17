"""Validate the M4A decode path with locally generated audio, including in the EXE."""

import os
from pathlib import Path
import subprocess
import tempfile
import wave

from media_loader import MediaLoader


def check_m4a(ffmpeg, codec, channels):
    with tempfile.TemporaryDirectory(prefix="whisper_m4a_check_") as directory:
        folder = Path(directory)
        source = folder / "Тест аудио.M4A"
        signals = "0.5*sin(2*PI*440*t)" + ("|0.25*sin(2*PI*880*t)" if channels == 2 else "")
        subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", f"aevalsrc={signals}:s=48000:d=0.6", "-c:a", codec, str(source)],
            check=True, timeout=20, capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        original = source.read_bytes()
        loader = MediaLoader()
        if not loader.is_supported(source) or "*.m4a" not in loader.dialog_filter():
            raise RuntimeError("M4A is missing from the file selection filters")
        prepared = loader.prepare(source, keep_extracted_audio=True)
        try:
            if prepared.is_video or not prepared.temporary or os.path.samefile(prepared.audio_path, source):
                raise RuntimeError("M4A must decode to a temporary WAV")
            with wave.open(prepared.audio_path, "rb") as audio:
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, channels, 2):
                    raise RuntimeError(f"Wrong WAV format for {codec}")
                if not 9000 <= audio.getnframes() <= 11000:
                    raise RuntimeError(f"Unexpected decoded duration for {codec}")
                samples = memoryview(audio.readframes(1024)).cast('h')
                if not all(any(samples[channel::channels]) for channel in range(channels)):
                    raise RuntimeError(f"No decoded audio for {codec}")
                if channels == 2 and samples[::2] == samples[1::2]:
                    raise RuntimeError(f"Stereo channels were mixed for {codec}")
        finally:
            prepared.cleanup()
        if Path(prepared.audio_path).exists() or source.read_bytes() != original or list(folder.glob("*.wav")):
            raise RuntimeError("M4A original changed or temporary WAV was not removed")


def run_media_validation():
    ffmpeg = MediaLoader().find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("Встроенный FFmpeg не найден")
    check_m4a(ffmpeg, "aac", 1)
    check_m4a(ffmpeg, "alac", 2)
