import os
import re
import subprocess
import tempfile
import winsound
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)


LINE_PATTERN = re.compile(
    r"^\[(?P<start>\d{1,4}:\d{2}(?:\.\d{1,3})?)\s*-\s*"
    r"(?P<end>\d{1,4}:\d{2}(?:\.\d{1,3})?)\]\s+"
    r"(?:(?P<name>[^:\r\n]+?)\s+\((?P<named>SPEAKER_\d+)\)|"
    r"(?P<plain>SPEAKER_\d+)):\s*(?P<text>.*)$"
)


@dataclass
class SpeechLine:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class SpeakerSample:
    speaker: str
    current_name: str = ""
    text: str = ""
    start: float | None = None
    end: float | None = None
    lines: list[SpeechLine] = field(default_factory=list)

    @property
    def playable(self):
        return self.start is not None and self.end is not None and self.end > self.start


def parse_time(value):
    minutes, seconds = value.split(":", 1)
    return int(minutes) * 60 + float(seconds)


def collect_speaker_samples(transcript):
    parsed = []
    names = {}
    for raw_line in transcript.splitlines():
        match = LINE_PATTERN.match(raw_line.strip())
        if not match:
            continue
        speaker = match.group("named") or match.group("plain")
        if match.group("name"):
            names[speaker] = match.group("name").strip()
        parsed.append(
            SpeechLine(
                speaker=speaker,
                start=parse_time(match.group("start")),
                end=parse_time(match.group("end")),
                text=match.group("text").strip(),
            )
        )

    speakers = {}
    for line in parsed:
        speakers.setdefault(
            line.speaker,
            SpeakerSample(line.speaker, current_name=names.get(line.speaker, "")),
        ).lines.append(line)

    for sample in speakers.values():
        groups = []
        current = []
        for line in parsed:
            if line.speaker != sample.speaker:
                if current:
                    groups.append(current)
                    current = []
                continue
            if current and line.start - current[-1].end > 1.5:
                groups.append(current)
                current = []
            current.append(line)
        if current:
            groups.append(current)

        playable = [group for group in groups if group[-1].end - group[0].start >= 5.0]
        if playable:
            best = max(playable, key=lambda group: group[-1].end - group[0].start)
            sample.start = best[0].start
            sample.end = min(best[-1].end, sample.start + 18.0)
            sample.text = " ".join(line.text for line in best)[:700]
        else:
            longest = sorted(sample.lines, key=lambda line: line.end - line.start, reverse=True)
            if longest:
                chosen = longest[0]
                sample.start = max(0.0, chosen.start - 0.25)
                sample.end = chosen.end + 0.25
                sample.text = chosen.text[:700]

    return [speakers[key] for key in sorted(speakers)]


def rename_speakers_in_text(transcript, mapping):
    result = transcript
    for speaker, raw_name in mapping.items():
        name = re.sub(r"[\r\n:]+", " ", raw_name).strip()
        pattern = re.compile(
            rf"(?m)^(\[[^\]\r\n]+\]\s+)"
            rf"(?:(?:[^:\r\n]+?)\s+\()?{re.escape(speaker)}\)?:"
        )
        label = f"{name} ({speaker}):" if name else f"{speaker}:"
        result = pattern.sub(lambda match: match.group(1) + label, result)
    return result


class SpeakerRenameDialog(QDialog):
    def __init__(self, transcript_path, media_path="", parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings
        self.transcript_path = str(transcript_path)
        self.media_path = str(media_path or "")
        self.original_text = Path(self.transcript_path).read_text(encoding="utf-8")
        self.samples = collect_speaker_samples(self.original_text)
        self.name_edits = {}
        self.saved = False
        self.play_token = 0
        self.sample_audio_cache = {}
        self.temp_audio_dir = tempfile.TemporaryDirectory(prefix="whisper_speakers_")

        self.setWindowTitle("Имена спикеров")
        self.resize(850, min(760, 280 + len(self.samples) * 190))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Прослушайте характерный фрагмент или прочитайте пример речи, затем введите имя. "
            "Одного участника можно выбрать для нескольких технических меток."
        ))

        participants_group = QGroupBox("Реальные участники")
        participants_layout = QHBoxLayout(participants_group)
        known_names = list(settings.get("speaker_names", [])) if settings else []
        known_names.extend(sample.current_name for sample in self.samples if sample.current_name)
        known_names = self._unique_names(known_names)
        self.participants_edit = QLineEdit("; ".join(known_names))
        self.participants_edit.setPlaceholderText("Например: Андрей; Филипп; Екатерина")
        refresh_button = QPushButton("Обновить списки")
        refresh_button.clicked.connect(self.refresh_participant_choices)
        participants_layout.addWidget(QLabel("Имена через точку с запятой:"))
        participants_layout.addWidget(self.participants_edit, 1)
        participants_layout.addWidget(refresh_button)
        layout.addWidget(participants_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)

        for sample in self.samples:
            group = QGroupBox(sample.speaker)
            group_layout = QVBoxLayout(group)
            form = QFormLayout()
            edit = QComboBox()
            edit.setEditable(True)
            edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            self.name_edits[sample.speaker] = edit
            form.addRow("Реальный участник:", edit)
            group_layout.addLayout(form)

            excerpt = QLabel(sample.text or "Текстовый фрагмент отсутствует")
            excerpt.setWordWrap(True)
            excerpt.setTextInteractionFlags(excerpt.textInteractionFlags())
            palette = self.palette()
            background = palette.color(QPalette.ColorRole.Base).name()
            foreground = palette.color(QPalette.ColorRole.Text).name()
            border = palette.color(QPalette.ColorRole.Mid).name()
            excerpt.setStyleSheet(
                "QLabel {"
                f"background-color: {background}; color: {foreground}; "
                f"border: 1px solid {border}; "
                "padding: 8px; border-radius: 4px;"
                "}"
            )
            group_layout.addWidget(excerpt)

            playback = QHBoxLayout()
            if sample.playable and self.media_path and os.path.isfile(self.media_path):
                button = QPushButton(
                    f"▶ Прослушать {sample.end - sample.start:.1f} сек. "
                    f"(с {self._format_time(sample.start)})"
                )
                button.clicked.connect(lambda _=False, item=sample: self.play_sample(item))
                playback.addWidget(button)
            else:
                reason = "Исходный медиафайл недоступен — показан текст"
                playback.addWidget(QLabel(reason))
            playback.addStretch()
            group_layout.addLayout(playback)
            body_layout.addWidget(group)

        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.refresh_participant_choices()
        for sample in self.samples:
            if sample.current_name:
                self.name_edits[sample.speaker].setCurrentText(sample.current_name)

        self.stop_button = QPushButton("■ Остановить звук")
        self.stop_button.clicked.connect(self.stop_audio)
        layout.addWidget(self.stop_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("Сохранить имена")
        buttons.accepted.connect(self.save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def play_sample(self, sample):
        self.play_token += 1
        token = self.play_token
        self._stop_system_audio()
        try:
            audio_path = self._prepare_sample_audio(sample)
            winsound.PlaySound(
                audio_path,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
            duration_ms = int((sample.end - sample.start) * 1000)
            QTimer.singleShot(duration_ms, lambda: self._stop_if_current(token))
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Не удалось воспроизвести фрагмент",
                f"Ошибка подготовки звука:\n{exc}",
            )

    def _prepare_sample_audio(self, sample):
        key = (sample.speaker, round(sample.start, 3), round(sample.end, 3))
        cached = self.sample_audio_cache.get(key)
        if cached and os.path.isfile(cached):
            return cached

        output_path = os.path.join(
            self.temp_audio_dir.name,
            f"{sample.speaker}_{int(sample.start * 1000)}.wav",
        )
        duration = max(0.1, sample.end - sample.start)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{sample.start:.3f}",
            "-i", self.media_path,
            "-t", f"{duration:.3f}",
            "-vn",
            "-ac", "2",
            "-ar", "48000",
            "-c:a", "pcm_s16le",
            "-y",
            output_path,
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if (
            completed.returncode != 0
            or not os.path.isfile(output_path)
            or os.path.getsize(output_path) <= 44
        ):
            details = completed.stderr.strip() or "FFmpeg не создал WAV-файл"
            raise RuntimeError(details)
        self.sample_audio_cache[key] = output_path
        return output_path

    def _stop_if_current(self, token):
        if token == self.play_token:
            self._stop_system_audio()

    def stop_audio(self):
        self.play_token += 1
        self._stop_system_audio()

    @staticmethod
    def _stop_system_audio():
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except RuntimeError:
            pass

    def participant_names(self):
        return self._unique_names(
            re.split(r"[;\r\n]+", self.participants_edit.text())
        )

    def refresh_participant_choices(self):
        names = self.participant_names()
        for combo in self.name_edits.values():
            current = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")
            combo.addItems(names)
            if current and current not in names:
                combo.addItem(current)
            combo.setCurrentText(current)
            combo.blockSignals(False)

    @staticmethod
    def _unique_names(values):
        result = []
        seen = set()
        for value in values:
            name = str(value).strip()
            key = name.casefold()
            if name and key not in seen:
                result.append(name)
                seen.add(key)
        return result

    def save_and_accept(self):
        mapping = {
            speaker: combo.currentText() for speaker, combo in self.name_edits.items()
        }
        if self.settings:
            all_names = self.participant_names() + [
                name for name in mapping.values() if name.strip()
            ]
            self.settings.set("speaker_names", self._unique_names(all_names))
        updated = rename_speakers_in_text(self.original_text, mapping)
        if updated == self.original_text:
            self.stop_audio()
            self.temp_audio_dir.cleanup()
            self.accept()
            return
        try:
            target = Path(self.transcript_path)
            temporary = target.with_name(target.name + ".speaker_names.tmp")
            temporary.write_text(updated, encoding="utf-8")
            os.replace(temporary, target)
            self.saved = True
            self.stop_audio()
            self.temp_audio_dir.cleanup()
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def reject(self):
        self.stop_audio()
        self.temp_audio_dir.cleanup()
        super().reject()

    def closeEvent(self, event):
        self.stop_audio()
        self.temp_audio_dir.cleanup()
        super().closeEvent(event)

    @staticmethod
    def _format_time(seconds):
        minutes = int(seconds // 60)
        remainder = seconds % 60
        return f"{minutes:02d}:{remainder:05.2f}"
