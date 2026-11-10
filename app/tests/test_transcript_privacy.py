import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from transcript_privacy import (
    Finding, PrivacyError, approved_mapping, ensure_removed, matches, merge_findings,
    parse_findings, replace_text, residual_source, rule_findings, save_result, split_for_discovery, variant_findings,
)


class TranscriptPrivacyTests(unittest.TestCase):
    def test_money_does_not_consume_preceding_date_even_in_model_response(self):
        text = 'Срок 15.04.2026 125 000 рублей, цена 500 рублей.'
        amounts = [f.original for f in rule_findings(text) if f.category == 'сумма']
        self.assertEqual(amounts, ['125 000 рублей', '500 рублей'])
        found = parse_findings('[{"original":"15.04.2026 125 000 рублей","category":"сумма"}]', text)
        self.assertEqual([f.original for f in found], ['125 000 рублей'])
        found = parse_findings('[{"original":"2026 125 000 рублей","category":"сумма"}]', text)
        self.assertEqual([f.original for f in found], ['125 000 рублей'])

    def test_generic_field_labels_are_not_proposed_by_model(self):
        import json
        labels = ['телефон', 'срок', 'Температура', 'Цена', 'Пароль']
        raw = json.dumps([{'original': label, 'category': 'другое'} for label in labels])
        self.assertEqual(parse_findings(raw, 'телефон срок Температура Цена Пароль'), [])

    def test_longest_first_case_whitespace_and_word_boundaries(self):
        source = 'Анна Иванова, АННА, Жанна, Анна   Иванова. Анне.'
        mapping = [('Анна', '[Участник]'), ('Анна Иванова', '[Участник]'), ('Анне', '[Участник]')]
        self.assertEqual(replace_text(source, mapping),
                         '[Участник], [Участник], Жанна, [Участник]. [Участник].')

    def test_no_cascading_replacement(self):
        self.assertEqual(replace_text('Альфа Бета', [('Альфа', 'Бета'), ('Бета', 'Гамма')]), 'Бета Гамма')

    def test_timing_and_speaker_ids_survive_numeric_replacements(self):
        source = '[01:01.00 - 02:01.00] Иван (SPEAKER_01): проект 01.'
        result = replace_text(source, [('Иван', '[Участник]'), ('01', '[Код]')])
        self.assertEqual(result, '[01:01.00 - 02:01.00] [Участник] (SPEAKER_01): проект [Код].')
        self.assertEqual(len(matches(source, '01')), 1)

    def test_rules_find_filename_speaker_and_contact_without_times(self):
        source = ('Источник: Иванов_совещание.wav\n[00:01.00 - 00:02.00] Иван Иванов (SPEAKER_01): '
                  'Пишите ivan@example.org, +7 (900) 123-45-67. ИНН 1234567890, 10.11.2026, 50 000 руб.')
        found = {f.original for f in rule_findings(source)}
        self.assertTrue({'Иванов_совещание.wav', 'Иван Иванов', 'ivan@example.org',
                         '+7 (900) 123-45-67', 'ИНН 1234567890', '10.11.2026', '50 000 руб.'} <= found)
        self.assertFalse(any('00:01' in value for value in found))

    def test_parser_rejects_incomplete_json_and_invented_findings(self):
        for raw in ('[{"original":"Иван"}', '{}', '[12]'):
            with self.subTest(raw=raw), self.assertRaises(PrivacyError):
                parse_findings(raw, 'Иван')
        raw = json.dumps([{'original': 'Иван', 'entity': 'Чужое имя'},
                          {'original': 'несуществующий'}, {'original': '█ Иван'}])
        found = parse_findings('```json\n' + raw + '\n```', 'Иван пришёл')
        self.assertEqual([(f.original, f.entity) for f in found], [('Иван', '')])

    def test_labelled_project_names_codes_and_secrets_are_not_left_to_model(self):
        source = 'Проект «Вектор-73», сплав АЛ-407. Пароль стенда: TestSecret-42. Обычная речь.'
        found = {f.original for f in rule_findings(source)}
        self.assertEqual(found, {'Вектор-73', 'АЛ-407', 'TestSecret-42'})

    def test_entity_chains_cycles_and_rules_share_one_alias(self):
        source = 'Иван Иванов, Ивану, Иванова, Иван'
        proposed = [Finding('Иван Иванов', 'ФИО'),
                    Finding('Иван Иванов', 'ФИО', entity='Иван'),
                    Finding('Ивану', 'ФИО', entity='Иван Иванов'),
                    Finding('Иванова', 'ФИО', entity='Ивану'),
                    Finding('Иван', 'ФИО', entity='Иванова')]
        found = merge_findings([], proposed, source)
        self.assertEqual(len(found), 4)
        self.assertEqual(len({f.entity for f in found}), 1)
        self.assertEqual(len({f.replacement for f in found}), 1)

    def test_reviewed_aliases_and_exclusions_survive_rescan(self):
        source = 'Иван, Ивану, Альфа'
        existing = [Finding('Иван', 'ФИО', '[Автор]', 'иван', False)]
        result = merge_findings(existing, [Finding('Иван', entity='Альфа'),
                                          Finding('Ивану', entity='Иван')], source)
        self.assertFalse(result[0].selected)
        self.assertEqual([f.replacement for f in result], ['[Автор]', '[Автор]'])
        self.assertEqual(existing[0].entity, 'иван')

    def test_observed_declensions_and_short_names_are_proposed(self):
        seeds = [Finding('Валентин Соколов', 'ФИО'), Finding('Анна Петрова', 'ФИО'),
                 Finding('ООО «Северный контур»', 'организация')]
        source = 'Передайте Валентину и Анне. Для Северного контура готово. Соколова предупредили.'
        found = variant_findings(seeds, source)
        self.assertEqual({f.original for f in found}, {'Валентину', 'Анне', 'Северного контура', 'Соколова'})
        self.assertEqual(next(f.entity for f in found if f.original == 'Валентину'), 'Валентин Соколов')

    def test_shared_short_name_does_not_link_two_different_people(self):
        seeds = [Finding('Иван Иванов', 'ФИО'), Finding('Иван Петров', 'ФИО')]
        found = variant_findings(seeds, 'Передайте Ивану.')
        self.assertEqual([(f.original, f.entity) for f in found], [('Ивану', 'Ивану')])

    def test_same_alias_for_different_values_is_allowed(self):
        mapping = approved_mapping([Finding('Иван', replacement='[Автор]'),
                                    Finding('Ивану', replacement='[Автор]')], 'Иван Ивану')
        self.assertEqual(replace_text('Иван Ивану', mapping), '[Автор] [Автор]')

    def test_alias_cannot_reintroduce_sensitive_text_or_metadata(self):
        for alias in ('Иван', '[Иван]', 'SPEAKER_99', 'строка\nстрока', ''):
            with self.subTest(alias=alias), self.assertRaises(PrivacyError):
                approved_mapping([Finding('Иван', replacement=alias)], 'Иван')

    def test_ambiguous_duplicate_mapping_is_rejected(self):
        with self.assertRaises(PrivacyError):
            approved_mapping([Finding('Иван', replacement='[А]'), Finding('ИВАН', replacement='[Б]')], 'Иван')

    def test_alias_cannot_hide_an_unreviewed_source_value_from_rescan(self):
        with self.assertRaises(PrivacyError):
            approved_mapping([Finding('Заказчик', replacement='Валентин')], 'Заказчик Валентин')

    def test_residual_scan_keeps_leftover_name_visible(self):
        source = 'Заказчик Валентин согласовал проект. Публичный термин.'
        findings = [Finding('Заказчик', replacement='[Компания]'),
                    Finding('Публичный термин', selected=False)]
        mapping = approved_mapping(findings, source)
        result = replace_text(source, mapping)
        scan = residual_source(result, findings)
        self.assertIn('Валентин', scan)
        self.assertNotIn('Компания', scan)
        self.assertNotIn('Публичный термин', scan)
        ensure_removed(result, mapping)

    def test_chunk_overlap_preserves_entity_crossing_boundary(self):
        text = 'а' * 4400 + ' Важное Название ' + 'б' * 5000
        chunks = list(split_for_discovery(text))
        self.assertTrue(any(' Важное Название ' in part for part in chunks))
        self.assertEqual(chunks[-1][-50:], text[-50:])

    def test_save_utf8_without_original_mapping_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'private.txt'
            target = Path(directory) / 'public.txt'
            source.write_text('Иван', encoding='utf-8')
            save_result(target, '[Автор]', source, [('Иван', '[Автор]')])
            self.assertEqual(target.read_text(encoding='utf-8'), '[Автор]')
            self.assertEqual(source.read_text(encoding='utf-8'), 'Иван')
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {'private.txt', 'public.txt'})
            for path in (source, target, Path(directory) / 'public.docx'):
                with self.assertRaises(PrivacyError):
                    save_result(path, '[Автор]', source, [])

    def test_failed_publication_cleans_staged_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'output.txt'
            with patch('transcript_privacy.os.link', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_result(target, '[Автор]', Path(directory) / 'source.txt', [])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_save_refuses_selected_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PrivacyError):
                save_result(Path(directory) / 'out.txt', 'Иван', Path(directory) / 'in.txt', [('Иван', '[Автор]')])
            self.assertEqual(list(Path(directory).iterdir()), [])
