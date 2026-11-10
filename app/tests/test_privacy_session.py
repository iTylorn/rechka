"""Data-loss and disclosure regressions using only synthetic source files."""
import tempfile
from pathlib import Path
import unittest

from privacy_session import PrivacySession, SourceSnapshot
from transcript_privacy import Finding, PrivacyError, merge_findings


class PrivacySessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def session(self, *texts):
        paths = []
        for index, text in enumerate(texts):
            path = self.folder / f'private-client-{index}.md'
            path.write_text(text, encoding='utf-8')
            paths.append(path)
        return PrivacySession([SourceSnapshot.read(path) for path in paths])

    def test_batch_reuses_alias_and_preserves_markdown_timing(self):
        session = self.session('# Встреча\n[00:00 - 00:06] Иван (SPEAKER_01): Привет.', 'Иван подтвердил.')
        findings = merge_findings([], [Finding('Иван', 'ФИО')], session.text)
        outputs, scan = session.render(findings)
        self.assertIn('[Участник 1]', outputs[0])
        self.assertIn('[Участник 1]', outputs[1])
        self.assertIn('[00:00 - 00:06]', outputs[0])
        self.assertIn('SPEAKER_01', outputs[0])
        self.assertNotIn('Иван', scan)

    def test_same_name_can_be_two_people_or_kept_at_one_occurrence(self):
        session = self.session('Иван слева, Иван справа, Иван в цитате.')
        findings = [Finding('Иван', replacement='[Участник 1]')]
        occurrences = session.occurrences('Иван')
        session.overrides[(occurrences[1].start, occurrences[1].end)] = '[Участник 2]'
        session.overrides[(occurrences[2].start, occurrences[2].end)] = None
        outputs, scan = session.render(findings)
        self.assertEqual(outputs, ['[Участник 1] слева, [Участник 2] справа, Иван в цитате.'])
        self.assertNotIn('Иван', scan)

    def test_occurrence_exclusion_does_not_mask_neighboring_secrets(self):
        session = self.session('Заказчик Валентин')
        outputs, scan = session.render([Finding('Заказчик', replacement='[Организация 1]')])
        self.assertIn('Валентин', scan)
        self.assertEqual(outputs, ['[Организация 1] Валентин'])

    def test_file_change_and_deletion_are_detected(self):
        session = self.session('Иван')
        path = Path(session.sources[0].path)
        path.write_text('Пётр', encoding='utf-8')
        with self.assertRaises(PrivacyError):
            session.assert_current()
        path.unlink()
        with self.assertRaises(PrivacyError):
            session.assert_current()

    def test_custom_alias_cannot_reintroduce_secret_or_collide_with_source(self):
        session = self.session('Иван и Пётр')
        occurrence = session.occurrences('Иван')[0]
        session.overrides[(occurrence.start, occurrence.end)] = 'Пётр'
        with self.assertRaises(PrivacyError):
            session.render([Finding('Иван', replacement='[Участник 1]')])

    def test_batch_export_uses_neutral_names_and_never_clobbers(self):
        session = self.session('Иван', 'Иван')
        outputs, _ = session.render([Finding('Иван', replacement='[Участник 1]')])
        paths = session.save_batch(self.folder, outputs)
        next_paths = session.save_batch(self.folder, outputs)
        self.assertTrue(set(paths).isdisjoint(next_paths))
        self.assertTrue(all('private' not in Path(path).name for path in paths))
        self.assertTrue(all(Path(path).read_text(encoding='utf-8') == '[Участник 1]' for path in paths))
        self.assertEqual(Path(session.sources[0].path).read_text(encoding='utf-8'), 'Иван')

    def test_protocol_aliases_are_masked_but_new_name_remains_visible(self):
        session = PrivacySession([SourceSnapshot('Протокол', '[Участник 1] передал Валентину')],
                                 known_aliases=['[Участник 1]'])
        _, scan = session.render([])
        self.assertNotIn('Участник', scan)
        self.assertIn('Валентину', scan)

    def test_source_filename_is_not_sent_as_hidden_metadata(self):
        session = self.session('Только разговор')
        self.assertNotIn('private-client', session.text)

    def test_declined_spellings_share_reviewed_alias(self):
        session = self.session('Иван позвонил Ивану.')
        findings = [Finding('Иван', replacement='[Участник 1]', entity='иван'),
                    Finding('Ивану', replacement='[Участник 1]', entity='иван')]
        self.assertEqual(session.render(findings)[0], ['[Участник 1] позвонил [Участник 1].'])
