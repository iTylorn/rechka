import os
import time
import gc
from pathlib import Path
import librosa
import openvino_genai as ov_genai
from PySide6.QtCore import QThread, Signal
from diarizer import OpenVINODiarizer
from media_loader import MediaCancelled, MediaLoader


class Transcriber(QThread):
    # Сигналы
    log = Signal(str)                     # обычное сообщение
    progress = Signal(int, int, str)      # текущий файл, всего файлов, описание
    file_done = Signal(str, int, int)     # путь сохранённого файла, индекс файла, всего файлов
    finished_ok = Signal(str)             # путь к последнему сохранённому файлу (для multi или отдельного)
    error = Signal(str)
    model_loaded = Signal(float)

    def __init__(self, file_list, model_path, device, chunk_minutes,
                 save_mode="next_to_source", save_folder="",
                 filename_template="{original_name}_transcript.txt",
                 multi_mode=False, keep_extracted_audio=False,
                 language="ru", enable_diarization=False,
                 embed_model_path="", diar_threshold=0.74,
                 num_speakers=0, max_auto_speakers=8, parent=None, *,
                 fallback_model_path=""):
        super().__init__(parent)
        self.file_list = file_list
        self.model_path = model_path
        self.fallback_model_path = fallback_model_path
        self.device = device
        self.chunk_minutes = chunk_minutes
        self.save_mode = save_mode
        self.save_folder = save_folder
        self.filename_template = filename_template
        self.multi_mode = multi_mode
        self.keep_extracted_audio = keep_extracted_audio
        self.language = language
        self.enable_diarization = enable_diarization
        self.embed_model_path = embed_model_path
        self.diar_threshold = diar_threshold
        self.num_speakers = num_speakers
        self.max_auto_speakers = max_auto_speakers
        self.diarizer = None
        self.media_loader = MediaLoader(Path(__file__).resolve().parent)
        self._durations = {}
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        self.media_loader.cancel()

    def run(self):
        try:
            self._process_files()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.diarizer = None
            gc.collect()

    def _process_files(self):
        total_files = len(self.file_list)
        if total_files == 0:
            self.error.emit("Нет файлов для обработки.")
            return

        self.log.emit("=" * 70)
        self.log.emit("WHISPER / OPENVINO / РАСПОЗНАВАНИЕ МЕДИА")
        self.log.emit("=" * 70)
        self.log.emit(f"Всего файлов: {total_files}")
        if self.multi_mode:
            self.log.emit("Режим сохранения: один общий файл")
        else:
            self.log.emit("Режим сохранения: отдельные файлы")

        # Загрузка модели Whisper
        self.log.emit("\nЗагрузка модели Whisper...")
        self.log.emit(f"Модель: {self.model_path}")
        self.log.emit(f"Устройство: {self.device}")
        model_start = time.time()
        pipe, loaded_model_path = self._load_whisper_pipeline()
        model_time = time.time() - model_start
        if loaded_model_path != self.model_path:
            self.log.emit(f"Использована резервная модель: {loaded_model_path}")
        self.log.emit(f"Модель загружена за {model_time:.1f} сек.")
        self.model_loaded.emit(model_time)

        # Если включена диаризация – загружаем OpenVINO модели
        if self.enable_diarization:
            self._load_diarization_pipeline()

        all_texts = []  # список кортежей (file_name, list_of_lines)

        for idx, audio_file in enumerate(self.file_list):
            if self._is_cancelled:
                self.log.emit("Распознавание прервано пользователем.")
                return

            self.progress.emit(idx, total_files, f"Обработка файла {idx+1} из {total_files}: {os.path.basename(audio_file)}")
            self.log.emit(f"\n{'='*70}")
            self.log.emit(f"Файл {idx+1}/{total_files}: {audio_file}")
            self.log.emit("="*70)

            prepared = None
            try:
                prepared = self.media_loader.prepare(
                    audio_file,
                    keep_extracted_audio=self.keep_extracted_audio,
                    cancel_check=lambda: self._is_cancelled,
                    log=self.log.emit,
                )
                text_lines = self._transcribe_file(
                    pipe, prepared.audio_path, source_file=audio_file
                )
            except MediaCancelled:
                self.log.emit("Обработка прервана пользователем.")
                return
            finally:
                if prepared is not None and prepared.temporary:
                    prepared.cleanup()
                    self.log.emit("Временный WAV удалён.")

            if self._is_cancelled:
                self.log.emit("Распознавание прервано пользователем.")
                return

            if self.multi_mode:
                all_texts.append((os.path.basename(audio_file), text_lines))
            else:
                save_path = self._get_save_path(audio_file)
                try:
                    self._save_text_to_file(save_path, audio_file, text_lines)
                    self.file_done.emit(save_path, idx, total_files)
                    self.log.emit(f"Файл сохранён: {save_path}")
                except Exception as e:
                    self.error.emit(f"Ошибка сохранения файла {audio_file}: {e}")
                    return

        if self._is_cancelled:
            self.log.emit("Распознавание прервано пользователем.")
            return

        if self.multi_mode:
            if len(all_texts) == 0:
                self.error.emit("Нет данных для сохранения.")
                return
            base_name = os.path.splitext(os.path.basename(self.file_list[0]))[0] if total_files == 1 else "transcript"
            multi_filename = f"{base_name}_multi.txt"
            save_path = self._get_multi_save_path(multi_filename)
            try:
                self._save_multi_file(save_path, all_texts)
                self.file_done.emit(save_path, total_files-1, total_files)
                self.finished_ok.emit(save_path)
                self.log.emit(f"Общий файл сохранён: {save_path}")
            except Exception as e:
                self.error.emit(f"Ошибка сохранения общего файла: {e}")
                return
        else:
            if total_files > 0:
                last_save_path = self._get_save_path(self.file_list[-1])
                self.finished_ok.emit(last_save_path)

    def _transcribe_file(self, pipe, audio_file, source_file=None):
        """Распознаёт один аудиофайл, возвращает список строк текста."""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Файл не найден: {audio_file}")

        self.log.emit("\nЗагрузка аудио...")
        load_start = time.time()
        source_audio, sample_rate = librosa.load(audio_file, sr=16000, mono=False)
        if source_audio.ndim == 2:
            audio = source_audio.mean(axis=0)
            channel_count = source_audio.shape[0]
        else:
            audio = source_audio
            channel_count = 1
        load_time = time.time() - load_start
        total_duration = len(audio) / 16000
        self._durations[source_file or audio_file] = total_duration
        self.log.emit(f"Длительность: {self._format_time(total_duration)}")
        self.log.emit(f"Sample rate: {sample_rate} Hz")
        self.log.emit(f"Каналов: {channel_count}")
        self.log.emit(f"Загрузка: {load_time:.1f} сек.")

        chunk_samples = self.chunk_minutes * 60 * 16000
        total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
        self.log.emit(f"Кусков: {total_chunks}, размер куска: {self.chunk_minutes} мин")

        all_lines = []
        transcription_start = time.time()

        for i in range(total_chunks):
            if self._is_cancelled:
                return all_lines

            start_sample = i * chunk_samples
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]
            chunk_start_sec = start_sample / 16000
            chunk_duration = len(chunk) / 16000

            self.log.emit(f"\n[{i+1}/{total_chunks}] {self._format_time(chunk_start_sec)} → {self._format_time(chunk_start_sec + chunk_duration)}")

            start = time.time()
            generation_config = {
                "task": "transcribe",
                "return_timestamps": True,
            }
            if self.language != "auto":
                generation_config["language"] = f"<|{self.language}|>"
            result = pipe.generate(chunk.tolist(), **generation_config)
            elapsed = time.time() - start
            speed = chunk_duration / elapsed if elapsed > 0 else 0
            self.log.emit(f"Время: {elapsed:.1f} сек. | Скорость: {speed:.2f}x realtime")

            if hasattr(result, "chunks") and result.chunks:
                for segment in result.chunks:
                    text = segment.text.strip()
                    if not text:
                        continue
                    global_start = chunk_start_sec + segment.start_ts
                    global_end = chunk_start_sec + segment.end_ts
                    line = f"[{self._format_timestamp(global_start)} - {self._format_timestamp(global_end)}] {text}"
                    all_lines.append(line)
                    self.log.emit(text)
            else:
                text = str(result).strip()
                if text:
                    all_lines.append(text)
                    self.log.emit(text)

        total_time = time.time() - transcription_start
        avg_speed = total_duration / total_time if total_time > 0 else 0
        self.log.emit(f"\nОбработка файла завершена: {self._format_time(total_time)} ({avg_speed:.2f}x realtime)")

        # Если включена диаризация – применяем её
        if self.enable_diarization and self.diarizer is not None:
            all_lines = self._apply_diarization(all_lines, source_audio, sample_rate)

        return all_lines

    def _load_diarization_pipeline(self):
        """Загружает WeSpeaker на Intel GPU."""
        try:
            self.log.emit("Загрузка WeSpeaker на GPU...")
            self.diarizer = OpenVINODiarizer(
                embed_model_path=self.embed_model_path,
                device="GPU",
                distance_threshold=self.diar_threshold,
                num_speakers=self.num_speakers,
                max_auto_speakers=self.max_auto_speakers,
            )
            mode = "автоматически" if not self.num_speakers else str(self.num_speakers)
            self.log.emit(f"WeSpeaker загружен на GPU. Количество спикеров: {mode}.")
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить GPU-диаризацию: {e}") from e

    def _apply_diarization(self, all_lines, audio, sample_rate):
        """Добавляет метки спикеров к строкам."""
        if self.diarizer is None:
            return all_lines

        self.log.emit("Выполняется диаризация...")
        speech_regions = []
        for line in all_lines:
            try:
                time_part = line.split('] ', 1)[0][1:]
                start_str, end_str = time_part.split(' - ')
                speech_regions.append((self._time_to_seconds(start_str), self._time_to_seconds(end_str)))
            except Exception:
                continue
        segments = self.diarizer.diarize(audio, sample_rate, speech_regions=speech_regions)
        speakers = sorted({speaker for _, _, speaker in segments})
        mode = "раздельные аудиоканалы" if self.diarizer.last_mode == "separate_channels" else "WeSpeaker на GPU"
        self.log.emit(
            f"Диаризация завершена ({mode}): {len(segments)} сегментов, спикеров: {len(speakers)}"
        )
        if (
            self.diarizer.last_mode == "wespeaker_gpu"
            and self.diarizer.last_raw_cluster_count > self.diarizer.last_cluster_count
        ):
            self.log.emit(
                "Авторежим объединил "
                f"{self.diarizer.last_raw_cluster_count} голосовых кластеров до заданного "
                f"максимума {self.diarizer.last_cluster_count}."
            )
        if all_lines and not segments:
            raise RuntimeError("WeSpeaker не смог получить ни одного голосового сегмента")

        result_lines = []
        for line in all_lines:
            try:
                time_part, text = line.split('] ', 1)
                time_part = time_part[1:]  # убираем '['
                start_str, end_str = time_part.split(' - ')
                start_sec = self._time_to_seconds(start_str)
                end_sec = self._time_to_seconds(end_str)
            except Exception:
                result_lines.append(line)
                continue

            max_overlap = 0
            best_speaker = "Unknown"
            for seg_start, seg_end, speaker in segments:
                overlap = max(0, min(end_sec, seg_end) - max(start_sec, seg_start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker

            if max_overlap > 0:
                new_line = f"[{start_str} - {end_str}] {best_speaker}: {text}"
            else:
                new_line = f"[{start_str} - {end_str}] Unknown: {text}"
            result_lines.append(new_line)

        return result_lines

    # --- Вспомогательные методы ---

    def _get_save_path(self, audio_file):
        if self.save_mode == "custom_folder":
            if not self.save_folder or not os.path.isdir(self.save_folder):
                raise ValueError("Папка для сохранения не выбрана или не существует")
            base_dir = self.save_folder
        elif self.save_mode == "ask":
            base_dir = os.path.dirname(audio_file)
        else:
            base_dir = os.path.dirname(audio_file)

        original_name = os.path.splitext(os.path.basename(audio_file))[0]
        filename = self.filename_template.replace("{original_name}", original_name)
        return os.path.join(base_dir, filename)

    def _get_multi_save_path(self, multi_filename):
        if self.save_mode == "custom_folder":
            if not self.save_folder or not os.path.isdir(self.save_folder):
                raise ValueError("Папка для сохранения не выбрана или не существует")
            base_dir = self.save_folder
        else:
            base_dir = os.path.dirname(self.file_list[0])
        return os.path.join(base_dir, multi_filename)

    def _save_text_to_file(self, save_path, audio_file, text_lines):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("WHISPER / OPENVINO / РАСПОЗНАВАНИЕ МЕДИА\n")
            f.write(f"Источник: {os.path.basename(audio_file)}\n")
            f.write(f"Длительность: {self._format_time(self._get_duration(audio_file))}\n")
            f.write("\n" + "=" * 70 + "\n\n")
            for line in text_lines:
                f.write(line + "\n")

    def _save_multi_file(self, save_path, all_texts):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("WHISPER / OPENVINO / РАСПОЗНАВАНИЕ МЕДИА (MULTI)\n")
            f.write(f"Количество файлов: {len(all_texts)}\n")
            f.write("\n" + "=" * 70 + "\n\n")
            for file_name, text_lines in all_texts:
                f.write(f"=== {file_name} ===\n\n")
                for line in text_lines:
                    f.write(line + "\n")
                f.write("\n" + "-" * 70 + "\n\n")

    def _get_duration(self, audio_file):
        if audio_file in self._durations:
            return self._durations[audio_file]
        try:
            return librosa.get_duration(path=audio_file)
        except:
            return 0.0

    @staticmethod
    def _format_time(seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _format_timestamp(seconds):
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining = seconds % 60
        if hours:
            return f"{hours:02d}:{minutes:02d}:{remaining:05.2f}"
        return f"{minutes:02d}:{remaining:05.2f}"

    @staticmethod
    def _time_to_seconds(time_str):
        parts = [float(part) for part in time_str.split(':')]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return parts[0] * 60 + parts[1]

    def _load_whisper_pipeline(self):
        """Load on GPU, trying configured FP16 before the legacy sibling."""
        try:
            return ov_genai.WhisperPipeline(self.model_path, "GPU"), self.model_path
        except Exception as primary_error:
            model_path = Path(self.model_path)
            fallback_name = model_path.name.replace("-int8-ov", "-fp16-ov")
            candidates = []
            if self.fallback_model_path:
                candidates.append(Path(self.fallback_model_path))
            candidates.append(model_path.with_name(fallback_name))
            tried = {os.path.normcase(str(model_path.resolve()))}
            fallback_errors = []
            for fallback_path in candidates:
                identity = os.path.normcase(str(fallback_path.resolve()))
                if identity in tried or not fallback_path.is_dir():
                    continue
                tried.add(identity)
                if not fallback_errors:
                    self.log.emit(f"Основная модель не загрузилась: {primary_error}")
                self.log.emit(f"Пробую резервную FP16-модель: {fallback_path}")
                try:
                    return ov_genai.WhisperPipeline(str(fallback_path), "GPU"), str(fallback_path)
                except Exception as fallback_error:
                    fallback_errors.append(f"FP16 ({fallback_path}): {fallback_error}")
            if fallback_errors:
                raise RuntimeError(
                    "Не удалось загрузить Whisper и резервные FP16-модели на GPU. "
                    f"Основная: {primary_error}; " + "; ".join(fallback_errors)
                ) from primary_error
            raise RuntimeError(f"Не удалось загрузить Whisper на GPU: {primary_error}") from primary_error
