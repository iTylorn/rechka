"""In-memory batch review with source snapshots and occurrence-level decisions."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from transcript_privacy import (
    PrivacyError, approved_mapping, mask_reviewed, matches, save_result, publish_document,
)


@dataclass(frozen=True)
class SourceSnapshot:
    label: str
    text: str
    path: str = ''
    digest: str = ''

    @classmethod
    def read(cls, path):
        path = Path(path).resolve()
        if path.suffix.lower() not in ('.txt', '.md'):
            raise PrivacyError('Выберите TXT или Markdown в UTF-8.')
        try:
            raw = path.read_bytes()
            text = raw.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
        except (OSError, UnicodeError) as exc:
            raise PrivacyError('Не удалось прочитать один из файлов в UTF-8.') from exc
        return cls(path.name, text, str(path), hashlib.sha256(raw).hexdigest())

    def assert_current(self):
        if not self.path:
            return
        try:
            current = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        except OSError as exc:
            raise PrivacyError('Исходный файл недоступен. Откройте пачку заново.') from exc
        if current != self.digest:
            raise PrivacyError('Исходный файл изменён. Откройте пачку заново и повторите проверку.')


@dataclass(frozen=True)
class Occurrence:
    start: int
    end: int
    source_index: int
    context: str


class PrivacySession:
    def __init__(self, sources=(), known_aliases=()):
        self.sources = list(sources)
        self.known_aliases = tuple(known_aliases)
        # None means explicitly keep this occurrence, a string overrides its alias.
        self.overrides = {}
        self.offsets = []
        offset = 0
        for source in self.sources:
            self.offsets.append(offset)
            offset += len(source.text) + 2
        self.text = '\n\n'.join(source.text for source in self.sources)

    def assert_current(self):
        for source in self.sources:
            source.assert_current()

    def occurrences(self, value):
        result = []
        for index, source in enumerate(self.sources):
            offset = self.offsets[index]
            for match in matches(source.text, value):
                start, end = match.span()
                context = source.text[max(0, start - 65):min(len(source.text), end + 65)]
                result.append(Occurrence(offset + start, offset + end, index, context))
        return result

    def render(self, findings):
        """Return per-document outputs and a masked residual scan source.

        Replacements are made once at original offsets, longest at each position.
        Explicitly kept occurrences are masked only at their exact new offsets.
        """
        active = [item for item in findings if item.selected]
        approved_mapping(active, self.text)
        proposals = {}
        for finding in findings:
            for occurrence in self.occurrences(finding.original):
                span = (occurrence.start, occurrence.end)
                value = self.overrides.get(span, finding.replacement if finding.selected else None)
                if value is not None:
                    # Validate custom occurrence aliases with the same policy as rows.
                    from transcript_privacy import Finding
                    approved_mapping([Finding(finding.original, replacement=value)], self.text)
                    if any(matches(value, other.original) for other in active):
                        raise PrivacyError('Замена содержит выбранное чувствительное значение.')
                proposals.setdefault(span, value)
        outputs, scanned = [], []
        for index, source in enumerate(self.sources):
            base = self.offsets[index]
            cursor, parts, masked = 0, [], []
            spans = sorted((start - base, end - base, value)
                           for (start, end), value in proposals.items()
                           if base <= start < end <= base + len(source.text))
            spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
            for start, end, alias in spans:
                if start < cursor:
                    if alias is not None:
                        # A longer replacement cannot silently leave part of another
                        # approved value visible. Both spans must be fully covered.
                        if end > cursor:
                            raise PrivacyError('Пересекающиеся вхождения требуют уточнения замен.')
                    continue
                parts.append(source.text[cursor:start])
                masked.append(source.text[cursor:start])
                value = source.text[start:end] if alias is None else alias
                parts.append(value)
                masked.append('█' * len(value))
                cursor = end
            parts.append(source.text[cursor:])
            masked.append(source.text[cursor:])
            outputs.append(''.join(parts))
            scanned.append(mask_reviewed(''.join(masked), self.known_aliases))
        return outputs, '\n\n'.join(scanned)

    def save_batch(self, folder, outputs):
        self.assert_current()
        if len(outputs) != len(self.sources):
            raise PrivacyError('Результат не соответствует выбранной пачке.')
        paths = []
        for index, source in enumerate(self.sources, 1):
            suffix = Path(source.path).suffix.lower() if source.path else '.txt'
            base = Path(folder) / f'Документ_{index}_обезличено{suffix}'
            path, counter = base, 2
            while path.exists():
                path = base.with_name(f'{base.stem}_{counter}{base.suffix}')
                counter += 1
            save_result(path, outputs[index - 1], source.path or path.with_suffix('.source'), [])
            paths.append(str(path))
        return paths
