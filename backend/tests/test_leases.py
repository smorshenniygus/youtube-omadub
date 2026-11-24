import tempfile
import unittest
from unittest.mock import patch
from dubbing.jobs import JobManager
from dubbing.leases import ClientLeases, LEASE_SECONDS

PAYLOAD = dict(video_id='abcdefghijk', source_language='en', target_language='ru',
               duration=5, cues=[dict(id='s', start=0, end=5, text='Hello')])


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = JobManager(self.tmp.name, autostart=False)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: self.manager.close())

    def test_abandoned_client_expires_and_cannot_block_cancel(self):
        with patch('dubbing.jobs.time.time', return_value=1000):
            jid = self.manager.create(dict(PAYLOAD, client_id='old'))['id']
        with patch('dubbing.jobs.time.time', return_value=1000 + LEASE_SECONDS + 1):
            self.manager.create(dict(PAYLOAD, client_id='new'))
            result = self.manager.cancel(jid, 'new')
            self.assertEqual(result['active_clients'], 0)
            self.assertEqual(result['status'], 'cancelled')

    def test_heartbeat_renews_only_existing_live_client(self):
        with patch('dubbing.jobs.time.time', return_value=1000):
            jid = self.manager.create(dict(PAYLOAD, client_id='live'))['id']
        with patch('dubbing.jobs.time.time', return_value=1100):
            self.manager.heartbeat(jid, 'live')
        with patch('dubbing.jobs.time.time', return_value=1200):
            self.assertEqual(self.manager.summary(jid)['active_clients'], 1)
            with self.assertRaises(ValueError):self.manager.heartbeat(jid, 'unknown')
        with patch('dubbing.jobs.time.time', return_value=1400):
            with self.assertRaises(ValueError):self.manager.heartbeat(jid, 'live')
            self.assertEqual(self.manager.summary(jid)['status'], 'cancelled')
            self.assertNotEqual(self.manager.create(dict(PAYLOAD, client_id='live'))['status'], 'cancelled')

    def test_restart_does_not_extend_persisted_deadline(self):
        with patch('dubbing.jobs.time.time', return_value=1000):
            jid = self.manager.create(dict(PAYLOAD, client_id='live'))['id']
        self.manager.close()
        with patch('dubbing.jobs.time.time', return_value=1200):
            self.manager = JobManager(self.tmp.name, autostart=False)
            self.assertEqual(self.manager.summary(jid)['status'], 'cancelled')

    def test_legacy_ids_get_only_one_migration_grace_period(self):
        job = dict(clients=['legacy'], cancelled=False)
        ClientLeases.initialize(job, 1000)
        ClientLeases.initialize(job, 1100)
        self.assertTrue(ClientLeases.expire(job, 1000 + LEASE_SECONDS + 1))
        self.assertTrue(job['cancelled'])

    def test_legacy_unleased_jobs_are_not_spontaneously_cancelled(self):
        job = dict(clients=[], cancelled=False)
        self.assertFalse(ClientLeases.expire(job, 1000))
        self.assertFalse(job['cancelled'])

    def test_worker_failure_is_unhealthy_not_silent(self):
        def broken_worker():
            raise OSError('fixture failure')
        with self.assertLogs(level='ERROR'):
            self.manager._run_worker(broken_worker)
        health = self.manager.health()
        self.assertEqual(health['status'], 'unhealthy')
        self.assertEqual(health['failed_workers'], ['broken_worker'])
        self.assertNotIn('fixture failure', str(health))

    def test_duplicate_translation_batch_has_no_partial_effect(self):
        jid = self.manager.create(PAYLOAD)['id']
        segment = dict(id='s', start=0, end=5, text='Hello', text_ru='', speaker='one', state='needs_translation')
        self.manager.jobs[jid]['segments'] = [segment]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.manager.translations(jid, [dict(id='s', text_ru='Привет'), dict(id='s', text_ru='Здравствуйте')])
        self.assertEqual(segment['text_ru'], '')
        self.assertEqual(segment['state'], 'needs_translation')
