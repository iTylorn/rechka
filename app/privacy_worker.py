"""GPU discovery only: text and replacement dictionaries never go to disk."""

from dataclasses import replace
import json
import os
from pathlib import Path
import threading

from PySide6.QtCore import QThread, Signal

from app_version import APP_DATA_DIR
from llm_processor import LocalLLMProcessor, ProcessingCancelled
from transcript_privacy import (
    SYSTEM_PROMPT, TECHNICAL_PROMPT, PrivacyError, approved_mapping, ensure_removed, mask_reviewed, matches,
    merge_findings, parse_findings, replace_text, residual_source, rule_findings,
    split_for_discovery, variant_findings,
)


FINDINGS_SCHEMA = json.dumps({'type': 'array', 'items': {
    'type': 'object', 'properties': {name: {'type': 'string'} for name in ('original', 'category', 'entity')},
    'required': ['original', 'category', 'entity'], 'additionalProperties': False,
}})
COMPACT_RETRY = ('\nОтвечай кратким JSON без пояснений. Выпиши каждую точную находку один раз. '
                 'Укажи entity только если он нужен для связи двух разных написаний; иначе "".')


class IncompleteDiscovery(PrivacyError):
    """A sub-fragment could not be scanned; no preview from this run is eligible."""


class PrivacyWorker(QThread):
    completed = Signal(object, str)  # complete findings list and candidate preview
    progress = Signal(int, str)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, model_path, source, findings=(), verify=False, parent=None, session=None):
        super().__init__(parent)
        self.model_path = model_path
        self.source = source
        self.findings = [replace(item) for item in findings]
        self.verify = verify
        self.session = session
        self.processor = None
        self._cancel_requested = threading.Event()
        self._publication_lock = threading.RLock()

    def cancel(self):
        with self._publication_lock:
            self._cancel_requested.set()
            processor = self.processor
        if processor is not None:
            processor.cancel()

    def _check_cancelled(self):
        if self._cancel_requested.is_set():
            if self.processor is not None:
                self.processor.cancel()
            raise ProcessingCancelled()

    def _structured_findings(self, chunk, prompt, token_limit):
        self._check_cancelled()
        raw = self.processor.generate_text(
            prompt, 'SOURCE:\n' + chunk, max_new_tokens=token_limit, json_schema=FINDINGS_SCHEMA,
        )
        self._check_cancelled()
        return parse_findings(raw, chunk)

    def _discover_chunk(self, chunk, system_prompt=SYSTEM_PROMPT, retry_notice=None):
        self._check_cancelled()
        try:
            return self._structured_findings(chunk, system_prompt, 3000)
        except PrivacyError:
            if retry_notice is not None:
                retry_notice()
        # Keep a 200-character overlap (the longest allowed quoted finding).
        # Each child is strictly smaller; the threshold bounds the retry tree.
        if len(chunk) > 600:
            middle = len(chunk) // 2
            left = self._discover_chunk(chunk[:middle + 100], system_prompt)
            right = self._discover_chunk(chunk[middle - 100:], system_prompt)
            return left + right
        try:
            return self._structured_findings(chunk, system_prompt + COMPACT_RETRY, 4500)
        except PrivacyError as exc:
            raise IncompleteDiscovery('Qwen не завершила список в одном из фрагментов.') from exc

    def run(self):
        try:
            self._check_cancelled()
            if not self.source.strip():
                raise PrivacyError('Расшифровка пуста.')
            preview = ''
            if self.session is not None:
                self.session.assert_current()
            if self.verify and self.session is not None:
                outputs, scan_text = self.session.render(self.findings)
                preview = '\n\n'.join(outputs)
            elif self.verify:
                mapping = approved_mapping(self.findings, self.source)
                preview = replace_text(self.source, mapping)
                ensure_removed(preview, mapping)
                scan_text = residual_source(preview, self.findings)
            else:
                scan_text = self.source
                if self.session is not None:
                    scan_text = mask_reviewed(scan_text, self.session.known_aliases)
            proposed = rule_findings(scan_text)
            chunks = list(split_for_discovery(mask_reviewed(scan_text), limit=3600, overlap=300))
            cache_dir = (Path(os.environ.get('LOCALAPPDATA', Path.home()))
                         / APP_DATA_DIR / 'llm_cache' / Path(self.model_path).name)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.progress.emit(0, 'Загрузка Qwen на Intel GPU…')
            self.processor = LocalLLMProcessor(self.model_path, 'GPU', cache_dir=cache_dir)
            self._check_cancelled()
            self.processor.load()
            self._check_cancelled()
            passes = (('Имена и реквизиты', SYSTEM_PROMPT),
                      ('Технические и коммерческие сведения', TECHNICAL_PROMPT))
            for index, chunk in enumerate(chunks):
                for pass_index, (label, prompt) in enumerate(passes):
                    percent = int(100 * (index * len(passes) + pass_index) / (len(chunks) * len(passes)))
                    location = f'фрагмент {index + 1} из {len(chunks)}'
                    self.progress.emit(percent, f'{label} · {location}')
                    try:
                        proposed.extend(self._discover_chunk(chunk, prompt, retry_notice=lambda:
                            self.progress.emit(percent, f'Уточняем ответ Qwen · {location}')))
                    except IncompleteDiscovery as exc:
                        raise PrivacyError(f'Qwen не закончила проверку: {location} ({label.lower()}). '
                                           'Выберите другую локальную Qwen и повторите этот этап.') from exc
            self._check_cancelled()
            # Resolve model/rule entities first, then propose observed spellings
            # the model may have skipped. Every new row still requires review.
            seeds = merge_findings(self.findings, proposed, self.source)
            proposed.extend(variant_findings(seeds, scan_text))
            if self.verify and any(not matches(self.source, f.original) for f in proposed):
                raise PrivacyError('Повторная проверка обнаружила сведения внутри новых замен. '
                                 'Используйте нейтральные обозначения и повторите проверку.')
            result = merge_findings(self.findings, proposed, self.source)
            if self.session is not None:
                self.session.assert_current()
            with self._publication_lock:
                self._check_cancelled()
                self.completed.emit(result, preview)
        except ProcessingCancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self._cancel_requested.is_set():
                self.cancelled.emit()
            else:
                # Model/native exceptions may embed prompts. Do not expose or log them.
                message = (str(exc) if isinstance(exc, PrivacyError) else
                           'Не удалось выполнить локальную проверку. Проверьте модель '
                           'Qwen OpenVINO и доступность Intel GPU; затем повторите поиск.')
                self.error.emit(message)
        finally:
            if self.processor is not None:
                self.processor.close()
                self.processor = None
