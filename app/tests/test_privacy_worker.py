import os
import itertools
import tempfile
import unittest
from unittest.mock import Mock, patch

from privacy_worker import COMPACT_RETRY, FINDINGS_SCHEMA, PrivacyWorker
from transcript_privacy import Finding


class PrivacyWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        env = patch.dict(os.environ, {'LOCALAPPDATA': temporary.name})
        env.start()
        self.addCleanup(env.stop)
        self.processor = Mock()
        self.processor.generate_text.return_value = '[]'
        self.factory = patch('privacy_worker.LocalLLMProcessor', return_value=self.processor)
        self.mock_factory = self.factory.start()
        self.addCleanup(self.factory.stop)
        self.results, self.errors, self.cancelled = [], [], []

    def worker(self, source='Иван пришёл.', findings=(), verify=False):
        worker = PrivacyWorker('model', source, findings, verify)
        worker.completed.connect(lambda *args: self.results.append(args))
        worker.error.connect(self.errors.append)
        worker.cancelled.connect(lambda: self.cancelled.append(True))
        return worker

    def test_discovery_only_uses_gpu_and_returns_observed_strings(self):
        self.processor.generate_text.return_value = '[{"original":"Иван","category":"ФИО"}]'
        worker = self.worker()
        worker.run()
        self.assertEqual(self.mock_factory.call_args.args, ('model', 'GPU'))
        self.assertEqual(self.processor.generate_text.call_args.kwargs['json_schema'], FINDINGS_SCHEMA)
        self.assertEqual(self.results[0][0][0].original, 'Иван')
        self.assertEqual(self.results[0][1], '')
        self.assertFalse(self.errors)
        self.processor.close.assert_called_once()
        self.assertIsNone(worker.processor)

    def test_residual_scan_adds_name_left_after_company_replacement(self):
        self.processor.generate_text.return_value = '[{"original":"Валентин","category":"ФИО"}]'
        worker = self.worker('Заказчик Валентин', [Finding('Заказчик', replacement='[Компания]')], True)
        worker.run()
        found, preview = self.results[0]
        self.assertEqual(preview, '[Компания] Валентин')
        self.assertEqual([f.original for f in found], ['Заказчик', 'Валентин'])
        sent = self.processor.generate_text.call_args.args[1]
        self.assertNotIn('Компания', sent)
        self.assertIn('Валентин', sent)

    def test_cancel_before_start_never_loads_model(self):
        worker = self.worker()
        worker.cancel()
        worker.run()
        self.mock_factory.assert_not_called()
        self.assertEqual(self.cancelled, [True])
        self.assertFalse(self.results or self.errors)

    def test_cancel_during_model_construction_or_generation_discards_result(self):
        for stage in ('construct', 'load', 'generate'):
            with self.subTest(stage=stage):
                self.results.clear()
                worker = self.worker()
                def cancel_and_return(*args, **kwargs):
                    worker.cancel()
                    return self.processor if stage == 'construct' else '[]'
                self.mock_factory.side_effect = cancel_and_return if stage == 'construct' else None
                self.processor.load.side_effect = cancel_and_return if stage == 'load' else None
                self.processor.generate_text.side_effect = cancel_and_return if stage == 'generate' else None
                worker.run()
                self.assertFalse(self.results or self.errors)
                self.assertIsNone(worker.processor)
        self.assertEqual(len(self.cancelled), 3)

    def test_invalid_json_never_certifies_preview(self):
        self.processor.generate_text.return_value = '[{"original":'
        self.worker('Иван', [Finding('Иван', replacement='[Автор]')], True).run()
        self.assertFalse(self.results)
        self.assertEqual(len(self.errors), 1)
        self.assertIn('фрагмент 1 из 1', self.errors[0])
        self.assertGreaterEqual(self.processor.generate_text.call_count, 2)

    def test_short_fragment_retries_instead_of_discarding_valid_findings(self):
        self.processor.generate_text.side_effect = ['[{"original":',
            '[{"original":"Иван","category":"ФИО","entity":""}]', '[]']
        worker = self.worker()
        notices = []
        worker.progress.connect(lambda _value, label: notices.append(label))
        worker.run()
        self.assertFalse(self.errors)
        self.assertEqual([f.original for f in self.results[0][0]], ['Иван'])
        retry = self.processor.generate_text.call_args_list[1]
        self.assertTrue(retry.args[0].endswith(COMPACT_RETRY))
        self.assertGreater(retry.kwargs['max_new_tokens'], 3000)
        self.assertTrue(any('Уточняем' in label for label in notices))

    def test_late_failure_does_not_publish_earlier_findings(self):
        source = 'Алексей. ' + 'фраза ' * 700 + ' TITLEPRIVATE.'
        def simulate(_prompt, user, **_kwargs):
            if 'TITLEPRIVATE' in user:
                return '[{"original":'
            return '[{"original":"Алексей","category":"ФИО","entity":""}]' if 'Алексей' in user else '[]'
        self.processor.generate_text.side_effect = simulate
        self.worker(source).run()
        self.assertFalse(self.results)
        self.assertIn('фрагмент 2 из 2', self.errors[0])
        self.assertNotIn('TITLEPRIVATE', self.errors[0])

    def test_dense_fragment_can_subdivide_past_old_minimum_without_missing_boundaries(self):
        secret = 'Алексей Иванов Секретный'
        source = 'а ' * 170 + secret + ' б' * 170
        def simulate(_prompt, user, **_kwargs):
            chunk = user.removeprefix('SOURCE:\n')
            if len(chunk) > 600:
                return '['
            return ('[{"original":"' + secret + '","category":"ФИО","entity":""}]'
                    if secret in chunk else '[]')
        self.processor.generate_text.side_effect = simulate
        self.worker(source).run()
        self.assertFalse(self.errors)
        self.assertIn(secret, [f.original for f in self.results[0][0]])

    def test_truncated_dense_response_retries_smaller_chunks(self):
        self.processor.generate_text.side_effect = itertools.chain(['['], itertools.repeat('[]'))
        self.worker('слово ' * 200).run()
        self.assertTrue(self.results)
        self.assertFalse(self.errors)
        self.assertGreater(self.processor.generate_text.call_count, 1)

    def test_native_errors_do_not_disclose_prompt_or_source(self):
        self.processor.generate_text.side_effect = ValueError('PRIVATE SOURCE HERE')
        self.worker().run()
        self.assertFalse(self.results)
        self.assertNotIn('PRIVATE', self.errors[0])

    def test_technical_pass_can_find_values_missed_by_general_pass(self):
        self.processor.generate_text.side_effect = ['[]', '[{"original":"715 градусов","category":"спецификация"}]']
        self.worker('Температура выдержки 715 градусов.').run()
        self.assertEqual(self.results[0][0][0].original, '715 градусов')

    def test_verified_preview_preserves_original_format(self):
        text = '[00:00.00 - 00:01.00] Иван (SPEAKER_01): привет!\n'
        self.worker(text, [Finding('Иван', replacement='[Автор]')], True).run()
        self.assertEqual(self.results[0][1], text.replace('Иван', '[Автор]'))

    def test_declined_first_name_is_found_after_full_name_was_removed(self):
        source = 'Валентин Иванов ушёл. Передайте Валентину.'
        self.worker(source, [Finding('Валентин Иванов', 'ФИО', '[Участник 1]')], True).run()
        found, preview = self.results[0]
        self.assertIn('Валентину', [f.original for f in found])
        self.assertIn('Валентину', preview)
        # The new row must be reviewed, not silently applied to an old preview.
        self.assertGreater(len(found), 1)
