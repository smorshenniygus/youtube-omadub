import json
import os
import sys
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from dubbing.editing import CodexEditor, EditingError


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = self.root / 'calls'
        self.rows = [dict(id='a', source='There are 12 boats.', draft='Есть 12 лодок.', window_s=2)]

    def runner(self, response=None, body=None):
        script = self.root / 'fake-codex'
        code = body or ('result = ' + repr(json.dumps(response or {'items': [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]}, ensure_ascii=False)) + '\nPath(args[args.index("--output-last-message")+1]).write_text(result)\nprint(json.dumps({"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":8}}))')
        script.write_text('#!/usr/bin/env python3\nimport sys,json,time,os\nfrom pathlib import Path\nargs=sys.argv[1:]\nprompt=sys.stdin.read()\nwith open('+repr(str(self.calls))+',"a") as log: log.write(json.dumps({"args":args,"prompt":prompt})+"\\n")\n' + code + '\n')
        script.chmod(0o700)
        return CodexEditor(self.root/'cache', executable=str(script), timeout=2)

    def test_real_process_validated_cache_and_safe_metrics(self):
        editor = self.runner()
        expected = [{'id': 'a', 'text_ru': 'Там 12 лодок.'}]
        self.assertEqual(editor.edit(self.rows), expected)
        self.assertFalse(editor.last_metrics['cache_hit'])
        self.assertEqual(editor.last_metrics['usage']['input_tokens'],20)
        self.assertEqual(editor.edit(self.rows), expected)
        self.assertTrue(editor.last_metrics['cache_hit'])
        self.assertEqual(len(self.calls.read_text().splitlines()),1)
        invocation=json.loads(self.calls.read_text())
        self.assertIn('--ephemeral',invocation['args'])
        self.assertIn('--ignore-user-config',invocation['args'])
        self.assertIn('model_reasoning_effort="medium"',invocation['args'])
        self.assertEqual(editor.last_metrics['effort'],'medium')
        for feature in ('computer_use','browser_use_external','browser_use_full_cdp_access','sleep_tool'):
            self.assertIn(feature,invocation['args'])
        self.assertIn('UNTRUSTED CONTENT',invocation['prompt'])
        self.assertNotIn('There are',json.dumps(editor.last_metrics))

    def test_default_executable_uses_stable_mise_wrapper_without_shell_path(self):
        runner = self.runner()
        wrapper = self.root/'mise'/'command-wrappers'/'bin'/'codex'
        wrapper.parent.mkdir(parents=True)
        wrapper.symlink_to(runner.executable)
        system_bin = self.root/'system-bin'
        system_bin.mkdir()
        (system_bin/'python3').symlink_to(sys.executable)
        with patch.dict(os.environ, {'PATH':str(system_bin), 'MISE_DATA_DIR':str(self.root/'mise'),
                                     'DUBBING_CODEX_EXECUTABLE':''}):
            editor = CodexEditor(self.root/'cache')
            self.assertEqual(editor._command('schema', 'output')[0], str(wrapper))
            self.assertEqual(editor.edit(self.rows)[0]['id'], 'a')

    def test_default_executable_prefers_path_and_explicit_override_is_preserved(self):
        runner = self.runner()
        with patch.dict(os.environ, {'PATH':str(self.root), 'DUBBING_CODEX_EXECUTABLE':str(runner.executable)}):
            self.assertEqual(CodexEditor(self.root/'cache')._command('s','o')[0], runner.executable)
        with patch.dict(os.environ, {'DUBBING_CODEX_EXECUTABLE':str(self.root/'missing')}):
            self.assertEqual(CodexEditor(self.root/'cache', executable=runner.executable)._command('s','o')[0], runner.executable)
        (self.root/'codex').symlink_to(runner.executable)
        with patch.dict(os.environ, {'PATH':str(self.root), 'DUBBING_CODEX_EXECUTABLE':''}):
            self.assertEqual(CodexEditor(self.root/'cache')._command('s','o')[0], str(self.root/'codex'))

    def test_missing_configured_executable_fails_actionably_without_fallback(self):
        editor = CodexEditor(self.root/'cache', executable=str(self.root/'missing'))
        with self.assertRaisesRegex(EditingError, 'DUBBING_CODEX_EXECUTABLE'):
            editor.edit(self.rows)
        self.assertEqual(list((self.root/'cache').glob('*.json')), [])

    def test_cache_key_covers_context_model_feedback_and_draft(self):
        editor=self.runner()
        editor.edit(self.rows)
        editor.edit(self.rows,context=[{'source':'Prior','draft':'Ранее'}])
        editor.edit([dict(self.rows[0],feedback={'duration_s':3})])
        editor.edit([dict(self.rows[0],draft='Вот 12 лодок.')])
        editor.model='different-model'
        editor.edit(self.rows)
        self.assertEqual(len(self.calls.read_text().splitlines()),5)

    def test_bad_output_fails_closed_and_is_not_cached(self):
        for response in [{'items':[]}, {'items':[{'id':'other','text_ru':'Там 12 лодок.'}]},
                {'items':[{'id':'a','text_ru':'Там 12 лодок.'}]*2},
                {'items':[{'id':'a','text_ru':'There are 12 boats.'}]},
                {'items':[{'id':'a','text_ru':'Там 13 лодок.'}]},
                {'items':[{'id':'a','text_ru':'  '}]},
                {'items':[{'id':'a','text_ru':'Там 12 лодок.','extra':'no'}]}, {'items':[], 'extra':1}]:
            with self.subTest(response=response):
                editor=self.runner(response)
                with self.assertRaises(EditingError): editor.edit(self.rows)
                self.assertEqual(list((self.root/'cache').glob('*.json')),[])

    def test_real_math_response_preserves_subscripts_and_fraction_values(self):
        from dubbing.editing import ResponseEditingError, _numbers, _same_numeric_occurrences, _written_numbers
        fixture=json.loads((Path(__file__).parent/'fixtures/numeric-math-failure.json').read_text())
        # Historical v8 fixture checks numeric equivalence of subscripts/fractions,
        # while v13 _validate also rejects remaining Latin math tokens (log₂, p).
        for row, out in zip(fixture['items'], fixture['output']['items']):
            self.assertTrue(_same_numeric_occurrences(out['text_ru'], _numbers(row['source']), _written_numbers(row['source'])))
        with self.assertRaises(ResponseEditingError) as ctx:
            CodexEditor._validate(fixture['output'], fixture['items'])
        self.assertTrue(all(item.get('code') == 'non_cyrillic_speech' for item in ctx.exception.feedback))
        self.assertFalse(_same_numeric_occurrences(
            fixture['output']['items'][0]['text_ru'].replace('log₂','log₃'),
            _numbers(fixture['items'][0]['source']), _written_numbers(fixture['items'][0]['source'])))
        self.assertFalse(_same_numeric_occurrences(
            fixture['output']['items'][4]['text_ru'].replace('1/2','1/3'),
            _numbers(fixture['items'][4]['source']), _written_numbers(fixture['items'][4]['source'])))

    def test_numeric_notation_equivalence_is_shared_by_source_and_output(self):
        from dubbing.editing import _numbers
        for source,output in [('log base 2, log base 2','Основание 2, log₂'),
                              ('base 1 half','Основание 1/2'),
                              ('1 half','Значение 0,5'),('3/4','Значение ¾'),
                              ('1½','Значение 1.5'),('x² + x³','Степени 2 и 3'),
                              ('2⁻³','2^-3'),('two third-party tools','Два сторонних инструмента'),
                              ('GPT-4','Модель GPT4'),('1/2/2025','Дата 1/2/2025')]:
            with self.subTest(source=source):
                self.assertEqual(_numbers(source),_numbers(output))
        self.assertNotEqual(_numbers('2⁻³'),_numbers('2³'))
        self.assertNotEqual(_numbers('-20'),_numbers('20'))
        self.assertNotEqual(_numbers('1/2'),_numbers('1/3'))
        self.assertNotEqual(_numbers('2 and 2'),_numbers('log₂'))

    def test_russian_cardinals_preserve_required_values_and_counts(self):
        cases = [
            ('one half', 'Значение 0,5.'),
            ('over the last decade and a half', 'За последние полтора десятилетия.'),
            ('20-something-year-old friends', 'Друзья чуть за двадцать.'),
            ('19th century, early 20th century', '19-й и начало 20 века.'),
            ('By 27, then by 25.', 'К двадцати семи, затем к двадцати пяти.'),
            ('3 red and 3 blue boats.', 'Три красные и 3 синие лодки.'),
            ('There are 1200 boats.', 'Там тысяча двести лодок.'),
            ('It is -20 degrees.', 'Сейчас минус двадцать градусов.'),
        ]
        for source, text in cases:
            with self.subTest(source=source):
                rows = [dict(self.rows[0], source=source, draft=text)]
                self.assertEqual(CodexEditor._validate({'items':[{'id':'a','text_ru':text}]}, rows)[0]['text_ru'], text)

    def test_russian_words_cannot_hide_missing_changed_or_duplicate_numbers(self):
        cases = [
            ('19th century, early 20th century', 'XVIII и начало XX века.'),
            ('19th and 19th again', 'Снова XIX век.'),
            ('19th century', 'Век IXX.'),
            ('over the last decade and a half', 'За последние 0,5 десятилетия.'),
            ('20 friends.', 'Двадцать один друг.'),
            ('20 friends.', 'Двадцать пять сотых.'),
            ('1000 friends.', '20 тысяч друзей.'),
            ('20 friends.', 'Двадцать тысяч друзей.'),
            ('3 red and 3 blue boats.', 'Три лодки.'),
            ('3 boats.', 'Три красные и три синие лодки.'),
            ('20 friends.', 'Минус двадцать друзей.'),
            ('20 friends.', 'Двадцать друзей и 21 помощник.'),
        ]
        for source, text in cases:
            with self.subTest(source=source, text=text):
                rows = [dict(self.rows[0], source=source, draft=text)]
                with self.assertRaises(EditingError):
                    CodexEditor._validate({'items':[{'id':'a','text_ru':text}]}, rows)

    def test_numeric_contract_uses_plain_decimal_not_scientific_notation(self):
        editor = self.runner({'items':[{'id':'a','text_ru':'Друзья чуть за двадцать.'}]})
        editor.edit([dict(self.rows[0], source='20-something-year-old friends')])
        call = json.loads(self.calls.read_text())
        payload = json.loads(call['prompt'].split('UNTRUSTED CONTENT\n',1)[1])
        self.assertEqual(payload['numeric_constraints'][0]['required_digits'], [{'value':'20','count':1}])

    def test_numeric_formatting_keeps_values_and_missing_fact_fails(self):
        editor=self.runner({'items':[{'id':'a','text_ru':'Стоимость — 1200 долларов, рост 1,5 процента.'}]})
        rows=[dict(self.rows[0],source='Costs 1,200 dollars, growth 1.5 percent.',draft='Цена 1200 долларов, рост 1,5 процента.')]
        self.assertEqual(editor.edit(rows)[0]['id'],'a')

    def test_cancel_kills_descendants_and_no_partial_batch_cache(self):
        pidfile=self.root/'child.pid'
        editor=self.runner(body='import subprocess\nchild=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])\nPath('+repr(str(pidfile))+').write_text(str(child.pid))\ntime.sleep(30)')
        with self.assertRaises(InterruptedError):
            editor.edit(self.rows,cancelled=lambda:pidfile.exists())
        pid=int(pidfile.read_text())
        stat=Path('/proc')/str(pid)/'stat'
        self.assertTrue(not stat.exists() or stat.read_text().split()[2]=='Z')
        self.assertEqual(list((self.root/'cache').glob('*.json')),[])

    def test_duplicate_json_keys_rejected(self):
        raw='{"items":[],"items":[{"id":"a","text_ru":"Там 12 лодок."}]}'
        editor=self.runner(body='Path(args[args.index("--output-last-message")+1]).write_text('+repr(raw)+')')
        with self.assertRaises(EditingError):editor.edit(self.rows)

    def test_source_digits_override_mistranslated_draft_digits(self):
        editor=self.runner({'items':[{'id':'a','text_ru':'Это было в 2025 году.'}]})
        rows=[dict(self.rows[0],source='This happened in 2025.',draft='Это произошло в 2024 году.')]
        self.assertEqual(editor.edit(rows)[0]['text_ru'],'Это было в 2025 году.')

    def test_russian_decimal_comma_keeps_three_fraction_digits(self):
        editor=self.runner({'items':[{'id':'a','text_ru':'Рост — 1,234 процента.'}]})
        rows=[dict(self.rows[0],source='Growth is 1.234 percent.',draft='Рост 1,234 процента.')]
        self.assertEqual(editor.edit(rows)[0]['id'],'a')
        editor=self.runner({'items':[{'id':'a','text_ru':'Рост — 1234 процента.'}]})
        rows[0]['feedback']='distinct cache key'
        with self.assertRaises(EditingError):editor.edit(rows)

    def test_repeated_numeric_facts_cannot_be_dropped(self):
        editor=self.runner({'items':[{'id':'a','text_ru':'Есть 3 лодки.'}]})
        rows=[dict(self.rows[0],source='3 red and 3 blue boats.',draft='3 красные и 3 синие лодки.')]
        with self.assertRaises(EditingError):editor.edit(rows)

    def test_source_written_quantities_allow_digits_alongside_required_digits(self):
        cases = [
            ('An agent replaces five people over 7 days.',
             'Агент заменяет пять человек за 7 дней.', 'Агент заменяет 5 человек за 7 дней.'),
            ('An agent replaces five people over 7 days.',
             'Агент заменяет пять человек за7дней.', 'Агент заменяет5человек за7дней.'),
            ('Twenty-five people work for 7 days.',
             'Двадцать пять человек работают 7 дней.', '25 человек работают 7 дней.'),
            ('A five-year plan takes 7 days to prepare.',
             'Пятилетний план готовят 7 дней.', 'План на 5 лет готовят 7 дней.'),
            ('A twenty-five-year plan takes 7 days to prepare.',
             'План на двадцать пять лет готовят 7 дней.', 'План на 25 лет готовят 7 дней.'),
            ('Five people arrive.', 'Приходят пять человек.', 'Приходят 5 человек.'),
            ('Five people arrive.', 'Приходят 6 человек.', 'Приходят 5 человек.'),
            ('One example takes 7 days.', 'Один пример занимает 7 дней.', 'Пример занимает 7 дней.'),
            ('Two hundred and twenty-five people work for 7 days.',
             'Работают двести двадцать пять человек 7 дней.', 'Работают 225 человек 7 дней.'),
            ('One million two hundred thousand people wait 7 days.',
             'Люди ждут 7 дней.', '1200000 человек ждут 7 дней.'),
        ]
        for source, draft, output in cases:
            with self.subTest(source=source, output=output):
                rows = [dict(self.rows[0], source=source, draft=draft)]
                self.assertEqual(CodexEditor._validate({'items':[{'id':'a','text_ru':output}]}, rows),
                                 [{'id':'a','text_ru':output}])

    def test_written_quantities_do_not_authorize_invented_or_missing_digit_facts(self):
        cases = [
            ('Five people work 7 days.', 'Работают 5 человек 7 дней.', 'Работают 6 человек 7 дней.'),
            ('Five people arrive.', 'Приходят 6 человек.', 'Приходят 6 человек.'),
            ('A twenty-five-year plan takes 7 days to prepare.',
             'План на двадцать пять лет готовят 7 дней.', 'План на 20 лет готовят 7 дней.'),
            ('Five people work 7 days and 7 nights.', 'Работают пять человек 7 дней и 7 ночей.',
             'Работают 5 человек 7 дней.'),
            ('Seven people work 7 days and 7 nights.', 'Работают семь человек 7 дней и 7 ночей.',
             'Работают 7 человек.'),
            ('Seven people and five helpers work 7 days and 7 nights.',
             'Семь человек и пять помощников работают 7 дней и 7 ночей.',
             '5 человек работают 7 дней.'),
            ('Five people work 7 days.', 'Работают пять человек 7 дней.', '5 человек и 5 помощников работают 7 дней.'),
            ('Five people have growth of 1.234 percent.', 'Пять человек, рост 1,234 процента.',
             '5 человек, рост 1234 процента.'),
        ]
        for source, draft, output in cases:
            with self.subTest(source=source, output=output):
                rows = [dict(self.rows[0], source=source, draft=draft)]
                with self.assertRaises(EditingError):
                    CodexEditor._validate({'items':[{'id':'a','text_ru':output}]}, rows)

    def test_optional_written_quantities_keep_decimal_comma_capacity_matching(self):
        rows = [dict(self.rows[0], source='One thousand two hundred people; growth 1.200 percent.',
                     draft='Тысяча двести человек; рост 1,200 процента.')]
        for output in ('1200 человек; рост 1,200 процента.', 'Люди; рост 1,200 процента.'):
            with self.subTest(output=output):
                self.assertEqual(CodexEditor._validate({'items':[{'id':'a','text_ru':output}]}, rows)[0]['text_ru'], output)

    def repair_runner(self, first, second):
        return self.runner(body='data=json.loads(prompt.split("UNTRUSTED CONTENT\\n",1)[1])\n'
            'result='+repr(first)+' if "repair" not in data else '+repr(second)+'\n'
            'Path(args[args.index("--output-last-message")+1]).write_text(json.dumps(result))\n'
            'print(json.dumps({"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":8}}))')

    def test_numeric_validation_repairs_once_and_retains_both_calls_usage(self):
        rows = [dict(self.rows[0], source='Five people work 7 days.', draft='Пять человек работают 7 дней.')]
        first = {'items':[{'id':'a','text_ru':'6 человек работают 7 дней.'}]}
        second = {'items':[{'id':'a','text_ru':'5 человек работают 7 дней.'}]}
        editor = self.repair_runner(first, second)
        self.assertEqual(editor.edit(rows), second['items'])
        self.assertEqual(editor.last_metrics['repair_attempts'], 1)
        self.assertEqual(editor.last_metrics['usage'], {'input_tokens':40, 'output_tokens':16})
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        initial = json.loads(calls[0]['prompt'].split('UNTRUSTED CONTENT\n',1)[1])
        self.assertEqual(initial['numeric_constraints'][0]['required_digits'], [{'value':'7','count':1}])
        self.assertEqual(initial['numeric_constraints'][0]['allowed_written_digits'], [{'value':'5','count':1}])
        repair = json.loads(calls[1]['prompt'].split('UNTRUSTED CONTENT\n',1)[1])['repair']
        self.assertEqual(repair['previous_output'], first)
        self.assertEqual(repair['validation_feedback'][0]['id'], 'a')
        self.assertEqual(repair['validation_feedback'][0]['required_digits'], [{'value':'7','count':1}])
        self.assertEqual(editor.edit(rows), second['items'])
        self.assertTrue(editor.last_metrics['cache_hit'])
        self.assertEqual(editor.last_metrics['repair_attempts'], 0)
        self.assertEqual(len(self.calls.read_text().splitlines()), 2)

    def test_numeric_repair_still_fails_closed_after_two_invalid_calls(self):
        bad = {'items':[{'id':'a','text_ru':'Там 13 лодок.'}]}
        editor = self.repair_runner(bad, bad)
        with self.assertRaises(EditingError):
            editor.edit(self.rows)
        self.assertEqual(len(self.calls.read_text().splitlines()), 2)
        self.assertEqual(editor.last_metrics['repair_attempts'], 1)
        self.assertEqual(editor.last_metrics['usage']['input_tokens'], 40)
        self.assertEqual(list((self.root/'cache').glob('*.json')), [])

    def test_numeric_repair_uses_words_for_unsupported_ordinal_or_indefinite_quantity(self):
        rows = [dict(self.rows[0], source='The second part takes a day.', draft='Вторая часть занимает день.')]
        first = {'items':[{'id':'a','text_ru':'2 часть занимает 1 день.'}]}
        second = {'items':[{'id':'a','text_ru':'Вторая часть занимает день.'}]}
        editor = self.repair_runner(first, second)
        self.assertEqual(editor.edit(rows), second['items'])
        self.assertEqual(len(self.calls.read_text().splitlines()), 2)

    def test_process_failure_and_cancellation_do_not_trigger_numeric_repair(self):
        editor = self.runner(body='sys.exit(3)')
        with self.assertRaises(EditingError):
            editor.edit(self.rows)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)
        self.assertEqual(editor.last_metrics['repair_attempts'], 0)
        self.calls.unlink()
        bad = {'items':[{'id':'a','text_ru':'Там 13 лодок.'}]}
        editor = self.repair_runner(bad, bad)
        def cancelled():
            return bool(editor.last_metrics.get('usage'))
        with self.assertRaises(InterruptedError):
            editor.edit(self.rows, cancelled=cancelled)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)
        self.assertEqual(list((self.root/'cache').glob('*.json')), [])

    def test_incomplete_response_is_repaired_once_through_cli(self):
        fixed = {'items':[{'id':'a','text_ru':'Там 12 лодок.'}]}
        editor = self.repair_runner({'items':[]}, fixed)
        self.assertEqual(editor.edit(self.rows), fixed['items'])
        self.assertEqual(editor.last_metrics['repair_attempts'], 1)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        repair = json.loads(calls[1]['prompt'].split('UNTRUSTED CONTENT\n',1)[1])['repair']
        self.assertEqual(repair['validation_feedback'][0]['code'], 'incomplete_response')
        self.assertEqual(len(calls), 2)

    def test_response_schema_requires_exact_batch_without_character_timing_caps(self):
        from dubbing.editing_contract import response_schema
        rows = [self.rows[0], dict(self.rows[0], id='b', feedback='too long', target_chars=15)]
        items = response_schema(rows)['properties']['items']
        self.assertEqual(items['minItems'], 2)
        self.assertEqual(items['maxItems'], 2)
        variants = items['items']['anyOf']
        self.assertEqual([v['properties']['id']['enum'] for v in variants], [['a'], ['b']])
        self.assertEqual(variants[1]['properties']['text_ru']['maxLength'], 10000)

    def test_duplicate_response_ids_repair_but_character_estimates_do_not_reject_facts(self):
        rows = [self.rows[0], dict(self.rows[0], id='b')]
        fixed = {'items':[{'id':r['id'],'text_ru':'Там 12 лодок.'} for r in rows]}
        bad = {'items':[fixed['items'][0], fixed['items'][0]]}
        editor = self.repair_runner(bad, fixed)
        self.assertEqual(editor.edit(rows), fixed['items'])
        rows = [dict(self.rows[0], feedback='too long', target_chars=15)]
        fixed = {'items':[{'id':'a','text_ru':'12 лодок.'}]}
        editor = self.repair_runner({'items':[{'id':'a','text_ru':'Там находятся 12 лодок.'}]}, fixed)
        self.assertEqual(editor.edit(rows), [{'id':'a','text_ru':'Там находятся 12 лодок.'}])
        self.assertEqual(editor.last_metrics['repair_attempts'], 0)

    def test_mixed_decimal_and_grouping_candidates_match_each_fact(self):
        editor=self.runner({'items':[{'id':'a','text_ru':'Всего 1,200 лодок, рост 1,234 процента.'}]})
        rows=[dict(self.rows[0],source='1,200 boats; growth 1.234 percent.',draft='1200 лодок, рост 1,234 процента.')]
        self.assertEqual(editor.edit(rows)[0]['id'],'a')

    def test_corrupt_cache_revalidated(self):
        editor=self.runner();editor.edit(self.rows)
        next((self.root/'cache').glob('*.json')).write_text('{"items":[]}')
        self.assertEqual(editor.edit(self.rows)[0]['text_ru'], 'Там 12 лодок.')
        self.assertTrue(editor.last_metrics['cache_invalid'])
        self.assertEqual(len(list((self.root/'cache').glob('*.invalid'))), 1)
        self.assertEqual(len(self.calls.read_text().splitlines()), 2)

    def test_cancel_and_deadline_terminate_process(self):
        editor=self.runner(body='time.sleep(30)')
        started=time.monotonic()
        with self.assertRaises(InterruptedError):editor.edit(self.rows,cancelled=lambda:time.monotonic()-started>.15)
        self.assertLess(time.monotonic()-started,3)
        editor.timeout=.15
        with self.assertRaises(EditingError):editor.edit(self.rows)
        self.assertEqual(list((self.root/'cache').glob('*.json')),[])

    def test_output_and_input_limits(self):
        editor=self.runner(body='os.write(1,b"x"*3000000)\ntime.sleep(30)')
        with self.assertRaises(EditingError):editor.edit(self.rows)
        count=len(self.calls.read_text().splitlines())
        with self.assertRaises(ValueError):editor.edit([dict(self.rows[0],window_s=float('nan'))])
        with self.assertRaises(ValueError):editor.edit(self.rows*2)
        with self.assertRaises(InterruptedError):editor.edit(self.rows,cancelled=lambda:True)
        self.assertEqual(len(self.calls.read_text().splitlines()),count)

    def test_six_row_batches_restore_requested_order(self):
        editor=self.runner(body='data=json.loads(prompt.split("UNTRUSTED CONTENT\\n",1)[1])\nrows=data["items"]\nresult={"items":[{"id":r["id"],"text_ru":"Там 12 лодок."} for r in reversed(rows)]}\nPath(args[args.index("--output-last-message")+1]).write_text(json.dumps(result))')
        rows=[dict(self.rows[0],id=str(i)) for i in range(8)]
        result=editor.edit(rows)
        self.assertEqual([r['id'] for r in result],[str(i) for i in range(8)])
        self.assertEqual(len(self.calls.read_text().splitlines()),2)

    def test_character_target_is_advisory_even_after_measured_overshoot(self):
        rows=[dict(id='a',source='Absolutely, excited to be here. Thank you for having me.',
                   draft='Конечно, рад быть здесь. Спасибо, что пригласили.',window_s=2.64,
                   target_chars=34,feedback={'raw_audio_seconds':3.52,'attempt':2})]
        editor=self.runner({'items':[{'id':'a','text_ru':'Конечно, рад быть здесь. Спасибо, что пригласили.'}]})
        self.assertEqual(editor.edit(rows)[0]['text_ru'], rows[0]['draft'])
        rows=[dict(rows[0],draft='Рад быть здесь. Спасибо!')]
        editor=self.runner({'items':[{'id':'a','text_ru':'Рад быть здесь. Спасибо!'}]})
        self.assertEqual(editor.edit(rows)[0]['text_ru'],'Рад быть здесь. Спасибо!')
        call=json.loads(self.calls.read_text().splitlines()[-1])
        self.assertIn('NOT a maximum',call['prompt'])
        self.assertNotIn('HARD MAXIMUM',call['prompt'])

    def test_impossible_character_hint_cannot_authorize_missing_repeated_numbers(self):
        from dubbing.editing import NumericEditingError
        row=dict(id='a',source='3 years. 3 years.',draft='3 года. 3 года.',window_s=4.7,
                 target_chars=3,feedback={'raw_audio_seconds':5.76})
        full={'items':[{'id':'a','text_ru':'3 года. 3 года.'}]}
        self.assertEqual(CodexEditor._validate(full,[row]),full['items'])
        with self.assertRaises(NumericEditingError):
            CodexEditor._validate({'items':[{'id':'a','text_ru':'3 года.'}]},[row])

    def test_exact_lexical_quantifiers_allow_optional_digits_without_overcounting(self):
        from dubbing.editing import NumericEditingError
        # Real failure from job f1c2a458771300484eab62f5b30f3768 (s_366f6892f2b0ee26e4b7)
        src = ('Now the single bit instruction for up comes up only one eighth of the time, '
               'the two bits for down also come up only one eighth of the time, and the other '
               'two instructions each require three bits, which together happens 75% of the time.')
        out = ('«Вверх» — команда из 1 бита, выпадает в одном случае из восьми; «вниз» — из 2 бит, '
               'тоже в одном из восьми. Две другие команды требуют по 3 бита и вместе выпадают в 75% случаев.')
        self.assertEqual(CodexEditor._validate({'items':[{'id':'a','text_ru':out}]}, [dict(self.rows[0], source=src)])[0]['text_ru'], out)
        # Other exact lexical quantifiers work optionally (digits or words).
        for s_text, ru_text in [
            ('It runs twice as fast.', 'Работает в 2 раза быстрее.'),
            ('It runs twice as fast.', 'Работает вдвое быстрее.'),
            ('Double the memory and triple the cores.', 'Памяти в 2 раза больше, ядер в 3 раза больше.'),
            ('Check it once.', 'Проверь 1 раз.'),
            ('Not one single bit was lost.', 'Ни 1 бита не потеряно.'),
        ]:
            with self.subTest(source=s_text, ru=ru_text):
                self.assertEqual(CodexEditor._validate({'items':[{'id':'a','text_ru':ru_text}]}, [dict(self.rows[0], source=s_text)])[0]['text_ru'], ru_text)
        # Never overcount "one single" as two 1s, never match derived words ("singles", "doubled"), never allow invented values.
        for s_text, bad_ru in [
            ('Not one single bit was lost.', '1 бит и ещё 1 бит.'),
            ('The singles tournament starts soon.', 'Турнир 1 скоро начнётся.'),
            ('They doubled down.', 'Они поставили 2 раза.'),
            ('A single bit.', 'Команда из 2 бит.'),
        ]:
            with self.subTest(source=s_text, bad=bad_ru):
                with self.assertRaises(NumericEditingError):
                    CodexEditor._validate({'items':[{'id':'a','text_ru':bad_ru}]}, [dict(self.rows[0], source=s_text)])

    def test_adjacent_comma_grouped_numbers_do_not_merge_or_crash_decimal(self):
        from decimal import Decimal
        from dubbing.editing import _numbers
        src = "We've got over 3,000 300,000 uh GitHub stars."
        self.assertEqual(_numbers(src), {Decimal('3000'): 1, Decimal('300000'): 1})
        ru = 'У нас больше 3000 — 300 000 звёзд на Гитхабе.'
        self.assertEqual(CodexEditor._validate({'items':[{'id':'a','text_ru':ru}]}, [dict(self.rows[0], source=src)])[0]['text_ru'], ru)
        self.assertEqual(_numbers('Рост 3 000,5 процента.'), {Decimal('3000.5'): 1})
        self.assertEqual(_numbers('Maybe a 50/50 chance.'), {Decimal('50'): 2})
        self.assertEqual(CodexEditor._validate(
            {'items':[{'id':'a','text_ru':'Шанс 50 на 50.'}]},
            [dict(self.rows[0], source='Maybe a 50/50 chance.')]
        )[0]['text_ru'], 'Шанс 50 на 50.')

    def test_non_cyrillic_tokens_rejected_and_repaired_once(self):
        from dubbing.editing import ResponseEditingError
        rows = [dict(self.rows[0], source='Discord, Microsoft, GPT-4 API on quince.com and π.',
                     draft='Discord, Microsoft, GPT-4 API на quince.com и π.')]
        first = {'items':[{'id':'a','text_ru':'В Discord и Microsoft работает GPT-4 API на quince.com и π.'}]}
        second = {'items':[{'id':'a','text_ru':'В Дискорде и Майкрософт работает Джи-Пи-Ти-4 Эй-Пи-Ай на квинс точка ком и пи.'}]}
        with self.assertRaises(ResponseEditingError) as ctx:
            CodexEditor._validate(first, rows)
        self.assertEqual(ctx.exception.feedback, [{
            'code': 'non_cyrillic_speech',
            'id': 'a',
            'tokens': ['Discord', 'Microsoft', 'GPT-4', 'API', 'quince.com', 'π'],
        }])
        editor = self.repair_runner(first, second)
        self.assertEqual(editor.edit(rows), second['items'])
        self.assertEqual(editor.last_metrics['repair_attempts'], 1)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        repair = json.loads(calls[1]['prompt'].split('UNTRUSTED CONTENT\n', 1)[1])['repair']
        self.assertEqual(repair['validation_feedback'][0]['code'], 'non_cyrillic_speech')
        self.assertIn('Discord', repair['validation_feedback'][0]['tokens'])

    def test_non_cyrillic_cjk_and_persistent_latin_fail_closed(self):
        from dubbing.editing import NumericEditingError
        rows = [dict(self.rows[0], source='Support us on Discord for 5 dollars.', draft='Поддержите нас в Discord за 5 долларов.')]
        bad = {'items':[{'id':'a','text_ru':'Поддержите нас в Discord за 5 долларов支持.'}]}
        editor = self.repair_runner(bad, bad)
        with self.assertRaises(EditingError):
            editor.edit(rows)
        self.assertEqual(editor.last_metrics['repair_attempts'], 1)
        self.assertEqual(list((self.root/'cache').glob('*.json')), [])
        # Combined numeric + non-Cyrillic feedback is delivered together in one repair pass.
        both_bad = {'items':[{'id':'a','text_ru':'Поддержите нас в Discord за 6 долларов.'}]}
        with self.assertRaises(NumericEditingError) as ctx:
            CodexEditor._validate(both_bad, rows)
        codes = [entry.get('code') for entry in ctx.exception.feedback]
        self.assertIn('non_cyrillic_speech', codes)
        self.assertEqual(len(ctx.exception.feedback), 2)

    def test_real_world_numeric_false_positives_pass_while_guards_hold(self):
        from dubbing.editing import NumericEditingError
        # 1) Roman numerals in source (GTA VI -> ГТА 6, World War II -> Второй мировой / 2, GTA V -> ГТА 5)
        for src, ru in [
            ('Even GTA VI or something?', 'Даже в ГТА 6 или что-то такое?'),
            ('Even GTA VI or something?', 'Даже в ГТА шесть или что-то такое?'),
            ('Since World War II.', 'Со времён 2-й мировой.'),
            ('Play GTA V or Part I.', 'Сыграй в ГТА 5 или часть 1.'),
            # 2) Spoken Russian negative exponent ("минус 18", "минус 10")
            (
                'The small region for this word has a width of 2^-18, whereas the points on the number line '
                'that are representable with ten bits or fewer are spaced out by 2^-10.',
                'Область слова имеет ширину 2 в степени минус 18, а точки числовой прямой, '
                'представимые максимум десятью битами, отстоят на 2 в степени минус 10.',
            ),
            # 3) Russian ordinals for required source digits ("19th century, early 20th century")
            (
                'How much do you read in the earlier history of mathematics, say, 19th century, early 20th century?',
                'Много ли вы читаете по истории математики, скажем, девятнадцатого века, начала двадцатого века?',
            ),
            (
                'Welcome to the 21st century.',
                'Добро пожаловать в двадцать первый век.',
            ),
            # 4) Hyphenated phonetic Russian digits ("3b1b.co/talent" -> "три-би-один-би")
            (
                'You can try out puzzles at 3b1b.co/talent.',
                'Попробуйте задачи на три-би-один-би точка ко слэш талент.',
            ),
            # 5) YouTube caption glued fraction typo ("1/5if" -> "1/5")
            (
                'Maybe about 1/5if of researcher time is decent.',
                '1/5 времени исследователей — неплохо.',
            ),
        ]:
            with self.subTest(source=src, ru=ru):
                self.assertEqual(
                    CodexEditor._validate({'items': [{'id': 'a', 'text_ru': ru}]}, [dict(self.rows[0], source=src)])[0]['text_ru'],
                    ru,
                )

        # Negative guards: pronoun "I", platform "X", hallucinated minus sign, and extra digits must still fail closed.
        for src, bad_ru in [
            ('I think X is great.', 'Я думаю, 1 и 10 — отлично.'),
            ('Width is 18 meters.', 'Ширина минус 18 метров.'),
            ('Full diet rated out of 10.', 'Обычный рацион — 10 из 10.'),
        ]:
            with self.subTest(bad_source=src, bad_ru=bad_ru):
                with self.assertRaises(NumericEditingError) as ctx:
                    CodexEditor._validate({'items': [{'id': 'a', 'text_ru': bad_ru}]}, [dict(self.rows[0], source=src)])
                self.assertIn('hint', ctx.exception.feedback[0])

if __name__=='__main__':unittest.main()
