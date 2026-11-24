import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dubbing.jobs import JobManager
from dubbing.persistence import QuarantinedJobError

PAYLOAD = dict(video_id='abcdefghijk', source_language='en', target_language='ru',
               duration=5, cues=[dict(id='s', start=0, end=5, text='Hello')])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manager = JobManager(self.root, autostart=False)
        self.addCleanup(lambda: self.manager.close())
        self.jid = self.manager.create(PAYLOAD)['id']
        self.good = self.manager.create(dict(PAYLOAD, video_id='xxxxxxxxxxx'))['id']
        self.path = self.root/self.jid/'job.json'
        self.original = json.loads(self.path.read_text())
        self.wav = self.root/self.jid/'untouched.wav'
        self.wav.write_bytes(b'unchanged audio evidence')
        self.wav_hash = hashlib.sha256(self.wav.read_bytes()).hexdigest()
        self.manager.close()

    def reload(self):
        with self.assertLogs(level='WARNING'):
            self.manager = JobManager(self.root, autostart=False)

    def assert_preserved_and_isolated(self, before):
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(hashlib.sha256(self.wav.read_bytes()).hexdigest(), self.wav_hash)
        self.assertEqual(self.manager.summary(self.good)['id'], self.good)
        self.assertEqual(self.manager.health()['status'], 'degraded')
        self.assertEqual(self.manager.health()['quarantined_jobs'], 1)
        with self.assertRaises(QuarantinedJobError):self.manager.summary(self.jid)
        with self.assertRaises(QuarantinedJobError):self.manager.create(PAYLOAD)
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_json_does_not_block_other_jobs_or_allow_overwrite(self):
        self.path.write_text('{broken')
        before = self.path.read_bytes()
        self.reload()
        self.assert_preserved_and_isolated(before)

    def test_future_schema_is_quarantined_not_downgraded(self):
        self.original['schema_version'] = 999
        self.path.write_text(json.dumps(self.original))
        before = self.path.read_bytes()
        self.reload()
        self.assert_preserved_and_isolated(before)

    def test_missing_fields_and_wrong_identity_are_quarantined(self):
        for change in ({'id': 'wrong'}, {'segments': None}, {'cancelled': 'false'}, {'priority': float('nan')}):
            with self.subTest(change=change):
                self.manager.close()
                job = dict(self.original, **change)
                self.path.write_text(json.dumps(job))
                before = self.path.read_bytes()
                self.reload()
                self.assert_preserved_and_isolated(before)

    def test_reference_escape_is_rejected_before_reading_or_hashing(self):
        job = copy.deepcopy(self.original)
        job['segments'] = [dict(id='s', state='pending', start=0, end=5, text='Hello', text_ru='',
                                speaker='one', reference=str(self.root/'outside.wav'), reference_sha256='0'*64)]
        self.path.write_text(json.dumps(job))
        before = self.path.read_bytes()
        with patch('dubbing.jobs.digest', side_effect=AssertionError('must not read escaped reference')):
            self.reload()
        self.assert_preserved_and_isolated(before)

    def test_duplicate_json_keys_are_not_silently_accepted(self):
        self.path.write_text('{"id":"'+self.jid+'","id":"other"}')
        before = self.path.read_bytes()
        self.reload()
        self.assert_preserved_and_isolated(before)
