import gc
import html
import os
import re
import time
from pathlib import Path

import openvino_genai as ov_genai
from PySide6.QtCore import QThread, Signal


class ProcessingCancelled(Exception):
    pass


class LocalLLMProcessor:
    def __init__(self, model_path, device="GPU", cache_dir=None):
        self.model_path = str(model_path)
        self.device = device
        self.cache_dir = str(cache_dir) if cache_dir else ""
        self.pipe = None
        self.tokenizer = None
        self.cancelled = False

    def load(self, log=None):
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Папка LLM не найдена: {self.model_path}")
        if not os.path.isfile(os.path.join(self.model_path, "openvino_model.xml")):
            raise ValueError("В папке LLM отсутствует openvino_model.xml")
        started = time.time()
        if log:
            log(f"Загрузка локальной LLM на {self.device}: {self.model_path}")
        properties = {"CACHE_DIR": self.cache_dir} if self.cache_dir else {}
        self.pipe = ov_genai.LLMPipeline(self.model_path, self.device, properties)
        self.tokenizer = self.pipe.get_tokenizer()
        if log:
            log(f"LLM загружена за {time.time() - started:.1f} сек.")

    def close(self):
        self.pipe = None
        self.tokenizer = None
        gc.collect()

    def cancel(self):
        self.cancelled = True

    @staticmethod
    def split_transcript(text, max_chars=14000):
        lines = text.splitlines()
        chunks = []
        current = []
        current_size = 0
        for line in lines:
            if len(line) > max_chars:
                if current:
                    chunks.append("\n".join(current))
                    current, current_size = [], 0
                for start in range(0, len(line), max_chars):
                    chunks.append(line[start:start + max_chars])
                continue
            added = len(line) + 1
            if current and current_size + added > max_chars:
                chunks.append("\n".join(current))
                current, current_size = [], 0
            current.append(line)
            current_size += added
        if current:
            chunks.append("\n".join(current))
        return [chunk for chunk in chunks if chunk.strip()]

    @staticmethod
    def normalize_transcript(text):
        """Make Whisper MM:SS.xx timestamps unambiguous for the language model."""
        timestamp = re.compile(
            r"\[(\d{1,4}):(\d{2})(?:\.(\d{1,3}))?\s*-\s*"
            r"(\d{1,4}):(\d{2})(?:\.(\d{1,3}))?\]"
        )

        def convert(minutes, seconds, fraction):
            total_minutes = int(minutes)
            hours, minute = divmod(total_minutes, 60)
            fraction = (fraction or "00")[:2].ljust(2, "0")
            return f"T+{hours:02d}:{minute:02d}:{int(seconds):02d}.{fraction}"

        def replace(match):
            start = convert(match.group(1), match.group(2), match.group(3))
            end = convert(match.group(4), match.group(5), match.group(6))
            return f"[{start} — {end}]"

        normalized = timestamp.sub(replace, text)
        numbered_lines = []
        evidence_number = 0
        for line in normalized.splitlines():
            if re.match(r"^\[T\+\d{2}:\d{2}:\d{2}\.\d{2}\s+—", line):
                evidence_number += 1
                line = f"[E{evidence_number:04d} | {line[1:]}"
            numbered_lines.append(line)
        normalized = "\n".join(numbered_lines)
        legend = (
            "СЛУЖЕБНОЕ ПОЯСНЕНИЕ К РАСШИФРОВКЕ:\n"
            "- Метки T+ЧЧ:ММ:СС.сс означают время от начала записи.\n"
            "- E0001, E0002 и т. п. — неизменяемые идентификаторы реплик.\n"
            "- SPEAKER_01, SPEAKER_02 и т. п. — говорящие участники записи.\n"
            "- Подпись «Имя (SPEAKER_01)» означает имя, подтверждённое пользователем.\n"
            "- Имена внутри реплик могут быть обращениями или упоминаниями; не связывай "
            "их с SPEAKER без однозначного подтверждения в тексте.\n\n"
        )
        return legend + normalized

    @staticmethod
    def resolve_evidence_references(result, normalized_transcript):
        evidence = dict(
            re.findall(
                r"\[(E\d{4})\s+\|\s+(T\+\d{2}:\d{2}:\d{2}\.\d{2})\s+—",
                normalized_transcript,
            )
        )
        for evidence_id, timestamp in evidence.items():
            result = re.sub(
                rf"\[?\b{re.escape(evidence_id)}\b\]?",
                f"[{timestamp}]",
                result,
            )

        valid_timestamps = set(
            re.findall(r"T\+\d{2}:\d{2}:\d{2}\.\d{2}", normalized_transcript)
        )

        def validate_timestamp(match):
            value = match.group(0)
            return value if value in valid_timestamps else "метка не подтверждена"

        return re.sub(
            r"T\+\d{2}:\d{2}:\d{2}\.\d{2}", validate_timestamp, result
        )

    @staticmethod
    def fill_prompt(template, transcript):
        if "{transcript}" in template:
            return template.replace("{transcript}", transcript)
        return f"{template}\n\n<transcript>\n{transcript}\n</transcript>"

    def process(self, transcript, prompt, log=None, progress=None, partial=None):
        processing_started = time.time()
        if self.pipe is None:
            self.load(log=log)
        transcript = self.normalize_transcript(transcript)
        chunks = self.split_transcript(transcript)
        if not chunks:
            raise ValueError("Расшифровка пуста")
        if log:
            log(f"Размер расшифровки: {len(transcript):,} символов; частей: {len(chunks)}")

        system_prompt = prompt.get("system_prompt", "")
        user_template = prompt.get("user_prompt", "{transcript}")
        max_new_tokens = int(prompt.get("max_new_tokens", 1400))
        temperature = float(prompt.get("temperature", 0.0))
        verification_source = transcript

        if len(chunks) == 1:
            if progress:
                progress(15, "Формирование результата")
            result = self._generate(
                system_prompt,
                self.fill_prompt(user_template, chunks[0]),
                max_new_tokens,
                temperature,
                partial,
            )
        else:
            analyses = []
            for index, chunk in enumerate(chunks, 1):
                self._check_cancelled()
                if progress:
                    progress(
                        int(5 + 65 * (index - 1) / len(chunks)),
                        f"Анализ части {index} из {len(chunks)}",
                    )
                map_instruction = (
                    f"Это часть {index} из {len(chunks)} длинной расшифровки. "
                    "Извлеки все сведения, необходимые для итоговой задачи: факты, темы, "
                    "решения, поручения, исполнителей, сроки, числа, риски и открытые вопросы. "
                    "Не делай предположений и сохраняй временные метки. Не пиши вступление.\n\n"
                    f"Целевая задача:\n{user_template.replace('{transcript}', '[полная расшифровка]')}\n\n"
                    f"<transcript_part>\n{chunk}\n</transcript_part>"
                )
                analyses.append(
                    self._generate(
                        system_prompt,
                        map_instruction,
                        min(max_new_tokens, 1100),
                        0.0,
                        None,
                    )
                )

            combined = self._compress_analyses(analyses, system_prompt, log, progress)
            if progress:
                progress(85, "Формирование итогового документа")
            final_source = (
                "Ниже приведены промежуточные результаты анализа всех частей. "
                "Объедини их, устрани повторы, ничего не добавляй от себя.\n\n" + combined
            )
            result = self._generate(
                system_prompt,
                self.fill_prompt(user_template, final_source),
                max_new_tokens,
                temperature,
                partial,
            )
            verification_source = combined

        if prompt.get("verify_facts", False):
            self._check_cancelled()
            if progress:
                progress(82, "Фактологическая проверка черновика")
            if log:
                log("Запущен второй проход: сверка каждого вывода с репликами")
            result = self._verify_factual_result(
                verification_source,
                result,
                max_new_tokens,
                partial,
            )

        if progress:
            progress(100, "Обработка завершена")
        if log:
            log(f"LLM-обработка выполнена за {time.time() - processing_started:.1f} сек.")
        return self.resolve_evidence_references(self._clean_result(result), transcript)

    def _verify_factual_result(self, source, draft, max_new_tokens, partial):
        verification_system = (
            "Ты — строгий фактчекер протоколов. Исправь черновик, сверяя каждое утверждение "
            "только с исходными репликами. Верни целиком готовый протокол на русском языке, "
            "без отчёта о проверке и без комментариев о своих действиях."
        )
        instruction = (
            "Проведи построчную проверку черновика по источнику. Обязательные правила:\n"
            "1. Удали любой пункт, если приведённая реплика не подтверждает именно этот вывод.\n"
            "2. Просьба говорящего со словом «ты» адресована другому SPEAKER. Обещание со "
            "словом «я» выполняет сам говорящий. Не пиши «поручил себе».\n"
            "3. Рассуждение, оценка, описание ставки или позиции — не поручение и не решение.\n"
            "4. Вопрос не открыт, если ниже по тексту на него дан ответ или принято действие.\n"
            "5. Не называй предложение или позицию окончательно принятым обеими сторонами "
            "решением. При сомнении перенеси его в обсуждённые темы либо удали.\n"
            "6. Участники — только SPEAKER_*. Упомянутые лица не являются участниками звонка.\n"
            "7. Не приписывай SPEAKER ставки, организации, должности или имена без прямого "
            "подтверждения. Не добавляй слова «возможно», «предполагаемый» и аналоги.\n"
            "8. Для решения или поручения укажи ровно один существующий E-идентификатор "
            "реплики, которая прямо его подтверждает. Не пиши метки T+ самостоятельно.\n"
            "9. Короткая цитата должна дословно соответствовать указанной E-реплике.\n"
            "10. Если подтверждённых элементов нет, напиши «Не выявлено».\n\n"
            f"<source>\n{source}\n</source>\n\n"
            f"<draft>\n{draft}\n</draft>"
        )
        return self._generate(
            verification_system,
            instruction,
            max_new_tokens,
            0.0,
            partial,
        )

    def _compress_analyses(self, analyses, system_prompt, log, progress):
        round_number = 1
        parts = list(analyses)
        while len("\n\n".join(parts)) > 24000 and len(parts) > 1:
            if log:
                log(f"Сжатие промежуточных результатов: проход {round_number}")
            reduced = []
            group = []
            group_size = 0
            for part in parts:
                if group and group_size + len(part) > 18000:
                    reduced.append(self._merge_group(group, system_prompt))
                    group, group_size = [], 0
                group.append(part)
                group_size += len(part)
            if group:
                reduced.append(self._merge_group(group, system_prompt))
            parts = reduced
            round_number += 1
            if progress:
                progress(78, "Объединение промежуточных результатов")
        return "\n\n".join(
            f"### Анализ части {index}\n{part}" for index, part in enumerate(parts, 1)
        )

    def _merge_group(self, group, system_prompt):
        text = "\n\n".join(group)
        instruction = (
            "Объедини промежуточные заметки без потери решений, поручений, сроков, "
            "исполнителей, чисел и временных меток. Удали только повторы. Не добавляй факты.\n\n"
            + text
        )
        return self._generate(system_prompt, instruction, 1200, 0.0, None)

    def _generate(self, system_prompt, user_prompt, max_new_tokens, temperature, partial):
        self._check_cancelled()
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "/no_think\n" + user_prompt},
        ]
        try:
            formatted = self.tokenizer.apply_chat_template(
                history,
                add_generation_prompt=True,
                extra_context={"enable_thinking": False},
            )
        except TypeError:
            formatted = self.tokenizer.apply_chat_template(history, True)

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = int(max_new_tokens)
        config.do_sample = temperature > 0
        if config.do_sample:
            config.temperature = max(0.01, float(temperature))
            config.top_p = 0.9
        config.repetition_penalty = 1.05
        config.apply_chat_template = False
        streamed = []
        last_emit_size = 0

        def streamer(fragment):
            nonlocal last_emit_size
            streamed.append(fragment)
            current_size = sum(map(len, streamed))
            if partial and current_size - last_emit_size >= 160:
                partial("".join(streamed))
                last_emit_size = current_size
            return self.cancelled

        result = self.pipe.generate(
            formatted,
            generation_config=config,
            streamer=streamer,
        )
        self._check_cancelled()
        if streamed:
            text = "".join(streamed)
        elif isinstance(result, str):
            text = result
        elif hasattr(result, "texts") and result.texts:
            text = result.texts[0]
        else:
            text = str(result)
        return self._clean_result(text)

    def _check_cancelled(self):
        if self.cancelled:
            raise ProcessingCancelled("Обработка остановлена пользователем")

    @staticmethod
    def _clean_result(text):
        text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
        text = text.replace("<|im_end|>", "").strip()
        return text


class LLMProcessingWorker(QThread):
    log = Signal(str)
    progress = Signal(int, str)
    partial = Signal(str)
    completed = Signal(str, str)
    error = Signal(str)

    def __init__(self, model_path, transcript_path, prompt, output_format="md", parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.transcript_path = transcript_path
        self.prompt = prompt
        self.output_format = output_format
        self.processor = None

    def cancel(self):
        if self.processor:
            self.processor.cancel()

    def run(self):
        try:
            transcript = Path(self.transcript_path).read_text(encoding="utf-8")
            cache_dir = (
                Path(os.environ.get("LOCALAPPDATA", Path.home()))
                / "WhisperGUI" / "llm_cache" / Path(self.model_path).name
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.processor = LocalLLMProcessor(
                self.model_path, "GPU", cache_dir=cache_dir
            )
            result = self.processor.process(
                transcript,
                self.prompt,
                log=self.log.emit,
                progress=self.progress.emit,
                partial=self.partial.emit,
            )
            output_path = self.make_output_path()
            self.write_result(output_path, result, self.output_format)
            self.completed.emit(output_path, result)
        except ProcessingCancelled as exc:
            self.log.emit(str(exc))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if self.processor:
                self.processor.close()
            self.processor = None

    def make_output_path(self):
        source = Path(self.transcript_path)
        slug = self.prompt.get("slug") or self.prompt.get("id", "ai_result")
        candidate = source.with_name(f"{source.stem}_{slug}.{self.output_format}")
        counter = 2
        while candidate.exists():
            candidate = source.with_name(
                f"{source.stem}_{slug}_{counter}.{self.output_format}"
            )
            counter += 1
        return str(candidate)

    @staticmethod
    def write_result(path, result, output_format):
        if output_format == "html":
            content = (
                "<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\">"
                "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;"
                "margin:40px auto;padding:0 24px;line-height:1.55}pre{white-space:pre-wrap;"
                "font:inherit}</style></head><body><pre>"
                + html.escape(result)
                + "</pre></body></html>"
            )
        else:
            content = result
        Path(path).write_text(content, encoding="utf-8")
