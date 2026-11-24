import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
import wave


def wav(path, seconds=.2):
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        f.writeframes(b''.join(struct.pack('<h', int(3000*math.sin(i*.1))) for i in range(int(seconds*8000))))

class Source:
    def prepare(self, video_id, cues, job_dir, progress=None):
        p = Path(job_dir)/'ref.wav'; wav(p)
        return {'segments':[dict(c, speaker='s1', reference=str(p)) for c in cues], 'metadata':{}, 'source_path':str(p)}

class Engine:
    def synthesize(self, segment, output_path):
        wav(output_path)

PAYLOAD = {'video_id':'abcdefghijk','duration':10,'source_language':'en','target_language':'ru', 'cues':[{'id':'a','start':1,'end':3,'text':'Hello'}, {'id':'b','start':5,'end':8,'text':'World'}]}

class QueueFixture:

    def setUp(self):
        from dubbing.jobs import JobManager
        self.tmp = tempfile.TemporaryDirectory(); self.Manager=JobManager
        self.m = JobManager(self.tmp.name, source_provider=Source(), engine_factory=Engine)
    def tearDown(self):
        self.m.close(); self.tmp.cleanup()
    def wait(self, jid, predicate):
        for _ in range(200):
            s=self.m.summary(jid)
            if predicate(s): return s
            time.sleep(.02)
        self.fail(str(s))
    def job(self):
        j=self.m.create(copy.deepcopy(PAYLOAD))['id']
        self.wait(j, lambda s: s['status']=='waiting_translation')
        return j

class QueueTests(QueueFixture, unittest.TestCase):
    def test_incremental_coverage_includes_silence_but_not_unready_speech(self):
        j=self.job(); self.m.translations(j,[{'id':'b','text_ru':'Мир'}])
        s=self.wait(j,lambda s:s['ready_segments']==1)
        self.assertEqual(s['ready_until'],1); self.assertFalse(s['ready_to_play'])
        self.m.translations(j,[{'id':'a','text_ru':'Привет'}])
        s=self.wait(j,lambda s:s['status']=='complete')
        self.assertEqual(s['ready_until'],10); self.assertTrue(s['ready_to_play'])
        self.assertEqual(s['ranges'],[[0,10]])
    def test_restart_resume_idempotency_and_changed_input(self):
        j=self.job(); self.m.translations(j,[{'id':'a','text_ru':'Привет'}])
        self.wait(j,lambda s:s['ready_segments']==1); self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine)
        self.assertEqual(self.m.create(copy.deepcopy(PAYLOAD))['id'],j)
        self.assertEqual(self.m.summary(j)['ready_segments'],1)
        p=copy.deepcopy(PAYLOAD);p['cues'][0]['text']='different'
        self.assertNotEqual(self.m.create(p)['id'],j)
    def test_cancel_and_explicit_create_resume(self):
        j=self.job();self.m.cancel(j);self.assertEqual(self.m.summary(j)['status'],'cancelled')
        self.m.create(copy.deepcopy(PAYLOAD));self.assertEqual(self.m.summary(j)['status'],'waiting_translation')
    def test_invalid_inputs_and_immutable_translation(self):
        for field,value in [('video_id','../../etc'),('duration',float('nan')),('cues',[])]:
            p=copy.deepcopy(PAYLOAD);p[field]=value
            with self.assertRaises(ValueError):self.m.create(p)
        j=self.job();self.m.translations(j,[{'id':'a','text_ru':'Привет'}])
        with self.assertRaises(ValueError):self.m.translations(j,[{'id':'a','text_ru':'Changed'}])
        with self.assertRaises(ValueError):self.m.translations(j,[{'id':'../x','text_ru':'x'}])
    def test_priority_selects_requested_pending_phrase(self):
        j=self.job(); self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine,autostart=False)
        self.m.translations(j,[{'id':'a','text_ru':'А'},{'id':'b','text_ru':'Б'}]);self.m.priority(j,5)
        self.assertEqual(self.m.next_segment()[1]['id'],'b')
    def test_priority_covers_containing_segment_first_and_computes_buffer_from_priority(self):
        self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine,autostart=False)
        p=dict(copy.deepcopy(PAYLOAD),duration=900,cues=[
            {'id':'s0','start':0,'end':200,'text':'A.'},
            {'id':'s1','start':200,'end':245,'text':'B.'},
            {'id':'s2','start':245,'end':560,'text':'C.'},
            {'id':'s3','start':560,'end':900,'text':'D.'},
        ])
        j=self.m.create(p)['id']
        # Prepare synchronously via _source_loop helper logic
        res=self.m.provider.prepare(p['video_id'],copy.deepcopy(p['cues']),self.m.root/j)
        from dubbing.jobs import digest
        with self.m.lock:
            job=self.m._get(j)
            job['segments']=[dict(s,reference=str(s['reference']),reference_sha256=digest(s['reference']),text_ru='Текст',state='pending') for s in res['segments']]
            job['prepared']=True
            self.m._save(job)
        # Priority at 244.5 is near end of s1 [200-245]; s1 MUST be picked before s2 [245-560]!
        self.m.priority(j,244.5)
        self.assertEqual(self.m.next_segment()[1]['id'],'s1')
        # Mark s1 and s2 ready while s0 [0-200] is still pending: ready_until stays 0,
        # but buffered_s from priority 244.5 is 560 - 244.5 = 315.5s >= 300s -> ready_to_play is True!
        with self.m.lock:
            job=self.m._get(j)
            for s in job['segments']:
                if s['id'] in ('s1','s2'):
                    s.update(state='ready',audio_duration=1.0,audio_sha256='0'*64)
        st=self.m.summary(j)
        self.assertEqual(st['ready_until'],0)
        self.assertAlmostEqual(st['buffered_s'],315.5,places=1)
        self.assertEqual(st['required_buffer_s'],300)
        self.assertTrue(st['ready_to_play'])
    def test_overflow_fails_without_ready_audio(self):
        class LongEngine:
            def synthesize(self,s,p):wav(p,10)
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=LongEngine)
        j=self.job();self.m.translations(j,[{'id':'a','text_ru':'Привет'}])
        s=self.wait(j,lambda s:s['status']=='failed')
        self.assertFalse(s['ready_to_play']);self.assertEqual(s['segments'][0]['state'],'error')

class HTTPTests(QueueFixture, unittest.TestCase):
    def test_auth_ranges_and_path_allowlist(self):
        from dubbing_service import make_server
        j=self.job();self.m.translations(j,[{'id':'a','text_ru':'Привет'}]);self.wait(j,lambda s:s['ready_segments']==1)
        server=make_server(self.m,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        def req(path, headers={}):return urllib.request.urlopen(urllib.request.Request(base+path,headers=headers))
        try:
            with req('/health') as r:self.assertEqual(r.status,200)
            with self.assertRaises(urllib.error.HTTPError) as e:req('/v1/jobs/'+j)
            self.assertEqual(e.exception.code,401);e.exception.close()
            h={'X-VoxDub-Token':self.m.token,'Range':'bytes=0-3'}
            with req('/v1/jobs/'+j+'/segments/a',h) as r:
                self.assertEqual(r.status,206);self.assertEqual(r.read(),b'RIFF')
            with self.assertRaises(urllib.error.HTTPError) as e:req('/v1/jobs/'+j+'/segments/../token',h)
            e.exception.close()
        finally:server.shutdown();server.server_close();thread.join()



class IntegrityTests(QueueFixture, unittest.TestCase):
    def test_changed_reference_invalidates_cached_audio_on_restart(self):
        j=self.job();self.m.translations(j,[{'id':'a','text_ru':'Привет'}]);self.wait(j,lambda s:s['ready_segments']==1)
        self.m.close();wav(Path(self.tmp.name)/j/'ref.wav',.3)
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine)
        self.assertEqual(self.m.summary(j)['status'],'failed')
        self.assertEqual(self.m.summary(j)['ready_segments'],0)
    def test_corrupt_audio_is_not_served_after_restart(self):
        j=self.job();self.m.translations(j,[{'id':'a','text_ru':'Привет'}]);self.wait(j,lambda s:s['ready_segments']==1)
        self.m.close();(Path(self.tmp.name)/j/'a.wav').write_bytes(b'broken')
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine,autostart=False)
        self.assertEqual(self.m.summary(j)['ready_segments'],0)
    def test_single_manager_per_data_directory(self):
        with self.assertRaises(RuntimeError):self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine)
    def test_normalized_authored_overlaps_pass_raw_caption_coverage_guard(self):
        from dubbing.source import SourceProvider
        class OverlapSource(Source):
            def prepare(self,video_id,cues,job_dir,progress=None):
                return super().prepare(video_id,SourceProvider._cues(cues),job_dir,progress)
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=OverlapSource(),engine_factory=Engine)
        payload=copy.deepcopy(PAYLOAD)
        payload['cues']=[dict(id='a',start=0,end=10,text='First.'),
                         dict(id='b',start=2,end=3,text='Second.'),
                         dict(id='c',start=2,end=4,text='Third.'),
                         dict(id='d',start=5,end=6,text='Fourth.')]
        j=self.m.create(payload)['id']
        result=self.wait(j,lambda s:s['status'] in ('failed','waiting_translation'))
        self.assertEqual(result['status'],'waiting_translation',result['error'])
        self.assertEqual([(s['start'],s['end']) for s in result['segments']],[(0,2),(2,5),(5,10)])

    def test_prepared_source_cannot_drop_caption_coverage(self):
        class DroppingSource(Source):
            def prepare(self,*args,**kwargs):
                result=super().prepare(*args,**kwargs);result['segments']=result['segments'][1:];return result
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=DroppingSource(),engine_factory=Engine)
        j=self.m.create(copy.deepcopy(PAYLOAD))['id']
        self.wait(j,lambda s:s['status']=='failed')

    def test_explicit_non_speech_coverage_is_accepted_without_a_fake_audio_segment(self):
        class EventSource(Source):
            def prepare(self,video_id,cues,job_dir,progress=None):
                result=super().prepare(video_id,cues[1:],job_dir,progress)
                result['metadata']['non_speech_cues']=[cues[0]]
                return result
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=EventSource(),engine_factory=Engine)
        payload=copy.deepcopy(PAYLOAD);payload['cues'][0]['text']='[laughter]'
        j=self.m.create(payload)['id']
        result=self.wait(j,lambda s:s['status'] in ('failed','waiting_translation'))
        self.assertEqual(result['status'],'waiting_translation',result['error'])
        self.assertEqual(len(result['segments']),1)
        self.assertEqual(result['ready_until'],5)

    def test_non_speech_metadata_cannot_exempt_spoken_words_or_expand_event_bounds(self):
        for text,expanded in [('Spoken [music] words.',False),('[laughter]',True)]:
            class BadEventSource(Source):
                def prepare(self,video_id,cues,job_dir,progress=None):
                    result=super().prepare(video_id,cues[1:],job_dir,progress)
                    event=dict(cues[0]);event['end']+=1 if expanded else 0
                    result['metadata']['non_speech_cues']=[event]
                    return result
            self.m.close();self.m=self.Manager(self.tmp.name,source_provider=BadEventSource(),engine_factory=Engine)
            payload=copy.deepcopy(PAYLOAD);payload['cues'][0]['text']=text
            j=self.m.create(payload)['id'];result=self.wait(j,lambda s:s['status']=='failed')
            self.assertFalse(result['ready_to_play'])

class CancelTests(QueueFixture, unittest.TestCase):
    def test_cancel_during_generation_prevents_next_phrase(self):
        entered=threading.Event();release=threading.Event();calls=[]
        class BlockingEngine:
            def synthesize(self,s,p):
                calls.append(s['id']);entered.set();release.wait(3);wav(p)
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=BlockingEngine)
        j=self.job();self.m.translations(j,[{'id':'a','text_ru':'А'},{'id':'b','text_ru':'Б'}])
        self.assertTrue(entered.wait(3));self.m.cancel(j);release.set()
        self.wait(j,lambda s:s['ready_segments']==1);time.sleep(.15)
        self.assertEqual(calls,['a']);self.assertEqual(self.m.summary(j)['status'],'cancelled')



class EngineLifetimeTests(QueueFixture, unittest.TestCase):
    def test_engine_is_reused_then_closed_after_idle(self):
        created=[];closed=threading.Event()
        class TrackingEngine(Engine):
            def __init__(self):created.append(self)
            def close(self):closed.set()
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=TrackingEngine,idle_timeout=.2)
        j=self.job();self.assertEqual(created,[])
        self.m.translations(j,[{'id':'a','text_ru':'А'},{'id':'b','text_ru':'Б'}])
        self.wait(j,lambda s:s['status']=='complete')
        self.assertEqual(len(created),1);self.assertTrue(closed.wait(2))

class RenderTests(unittest.TestCase):
    def test_float_waveform_tempo_fit_with_peak_headroom(self):
        from dubbing.render import fit_audio
        from array import array
        with tempfile.TemporaryDirectory() as tmp:
            import numpy as np
            import soundfile as sf
            raw=Path(tmp)/'raw.wav';out=Path(tmp)/'out.wav'
            t=np.arange(int(24000*2.2))/24000
            envelope=.15+.5*np.sin(2*np.pi*3*t)**2
            sf.write(raw,envelope*np.sin(2*np.pi*220*t),24000,subtype='FLOAT')
            duration=fit_audio(raw,out,2)
            self.assertLessEqual(duration,2);self.assertGreater(duration,1.8)
            with wave.open(str(out),'rb') as f:
                self.assertEqual(f.getframerate(),48000)
                values=array('h',f.readframes(f.getnframes()))
            self.assertLessEqual(max(abs(v) for v in values),round(.85*32767)+1)
            self.assertFalse(out.with_suffix('.fitting.wav').exists())

class ProgressTests(QueueFixture, unittest.TestCase):
    def test_source_progress_and_generation_metrics_are_persisted(self):
        class ReportingSource(Source):
            def prepare(self,*args,progress=None,**kwargs):
                progress({'phase':'analysis','seconds':3})
                result=super().prepare(*args,**kwargs);result['metadata']={'quality_flags':['approximate'],'references':{'s1':{'path':'private','start':0}}};return result
        class ReportingEngine(Engine):
            def synthesize(self,s,p):
                super().synthesize(s,p);return {'generation_seconds':.1,'peak_allocated_mib':123,'peak_reserved_mib':456}
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=ReportingSource(),engine_factory=ReportingEngine)
        j=self.job();self.assertEqual(self.m.summary(j)['source_progress']['phase'],'analysis')
        self.m.translations(j,[{'id':'a','text_ru':'А'}]);s=self.wait(j,lambda s:s['ready_segments']==1)
        self.assertEqual(s['segments'][0]['metrics']['peak_allocated_mib'],123)
        self.assertEqual(s['model_load_count'],1)
        self.assertEqual(s['source_metadata']['quality_flags'],['approximate'])
        self.assertNotIn('path',s['source_metadata']['references']['s1'])
        self.assertGreaterEqual(s['segments'][0]['metrics']['fit_seconds'],0)

if __name__=='__main__':unittest.main()

class LoudnessTests(unittest.TestCase):
    def test_quiet_and_loud_inputs_share_loudness_without_clipping(self):
        from dubbing.render import fit_audio
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            t=np.arange(24000*4)/24000;signal=np.sin(2*np.pi*220*t)*(.25+.5*np.sin(2*np.pi*3*t)**2)
            outputs=[]
            for i,level in enumerate((.03,.5)):
                raw=Path(tmp)/f'in{i}.wav';out=Path(tmp)/f'out{i}.wav'
                sf.write(raw,signal*level,24000,subtype='FLOAT');fit_audio(raw,out,5)
                audio,sr=sf.read(out);self.assertEqual(sr,48000);outputs.append(float(np.sqrt(np.mean(audio**2))))
                self.assertLessEqual(float(np.abs(audio).max()),.855)
            self.assertLess(abs(20*math.log10(outputs[0]/outputs[1])),.5)

class ExplicitRetryTests(QueueFixture, unittest.TestCase):
    def test_create_retries_failed_source_with_same_job_id(self):
        class FailOnceSource(Source):
            attempts=0
            def prepare(self,*args,**kwargs):
                self.attempts+=1
                if self.attempts==1:raise RuntimeError('temporary source failure')
                return super().prepare(*args,**kwargs)
        self.m.close();provider=FailOnceSource()
        self.m=self.Manager(self.tmp.name,source_provider=provider,engine_factory=Engine)
        j=self.m.create(copy.deepcopy(PAYLOAD))['id'];self.wait(j,lambda s:s['status']=='failed')
        time.sleep(.15);self.assertEqual(provider.attempts,1)
        self.assertEqual(self.m.create(copy.deepcopy(PAYLOAD))['id'],j)
        self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[{'id':'a','text_ru':'А'},{'id':'b','text_ru':'Б'}])
        self.wait(j,lambda s:s['status']=='complete')
    def test_create_retries_failed_engine_preserving_ready_audio(self):
        calls=[];instances=[]
        class FailOnceEngine(Engine):
            def __init__(self):instances.append(self)
            def synthesize(self,s,p):
                calls.append(s['id'])
                if s['id']=='b' and calls.count('b')==1:raise RuntimeError('temporary CUDA failure')
                return super().synthesize(s,p)
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=FailOnceEngine)
        j=self.job();self.m.translations(j,[{'id':'a','text_ru':'А'},{'id':'b','text_ru':'Б'}])
        self.wait(j,lambda s:s['status']=='failed');time.sleep(.15)
        self.assertEqual(calls,['a','b']);self.assertEqual(self.m.summary(j)['ready_segments'],1)
        self.assertEqual(self.m.create(copy.deepcopy(PAYLOAD))['id'],j)
        self.wait(j,lambda s:s['status']=='complete')
        self.assertEqual(calls,['a','b','b']);self.assertEqual(len(instances),2)

class ClientLeaseTests(QueueFixture, unittest.TestCase):
    def test_late_old_cancel_does_not_cancel_new_session_same_job(self):
        a=dict(copy.deepcopy(PAYLOAD),client_id='old-session')
        b=dict(copy.deepcopy(PAYLOAD),client_id='new-session')
        j=self.m.create(a)['id'];self.assertEqual(self.m.create(b)['id'],j)
        self.m.cancel(j,client_id='old-session');s=self.m.summary(j)
        self.assertNotEqual(s['status'],'cancelled');self.assertEqual(s['active_clients'],1)
        self.m.cancel(j,client_id='old-session');self.assertNotEqual(self.m.summary(j)['status'],'cancelled')
        self.m.cancel(j,client_id='new-session');self.assertEqual(self.m.summary(j)['status'],'cancelled')
    def test_client_leases_survive_restart_and_invalid_ids_rejected(self):
        p=dict(copy.deepcopy(PAYLOAD),client_id='session-a');j=self.m.create(p)['id']
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine)
        self.assertEqual(self.m.summary(j)['active_clients'],1)
        self.m.cancel(j);self.assertEqual(self.m.summary(j)['active_clients'],0)
        for ident in ('../x','x'*97,{},''):
            with self.assertRaises(ValueError):self.m.create(dict(copy.deepcopy(PAYLOAD),client_id=ident))


class TimingRecoveryTests(QueueFixture, unittest.TestCase):
    def test_actual_reported_15520ms_phrase_fits_12160ms_without_clipping(self):
        from dubbing.render import fit_audio
        from array import array
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw.wav';out=Path(tmp)/'out.wav';wav(raw,15.52)
            duration=fit_audio(raw,out,12.16)
            self.assertLessEqual(duration,12.16+1/24000)
            self.assertGreater(duration,12)
            with wave.open(str(out),'rb') as f:values=array('h',f.readframes(f.getnframes()))
            self.assertLessEqual(max(abs(v) for v in values),round(.70*32767))
            self.assertGreater(max(abs(v) for v in values[-2400:]),100)

    def test_mild_overflow_does_not_fail_job_or_discard_other_ready_clip(self):
        class MildOverflowEngine:
            def synthesize(self,s,p):wav(p,2.6 if s['id']=='a' else .2)
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=MildOverflowEngine)
        j=self.job();self.m.translations(j,[{'id':'b','text_ru':'Мир'}])
        self.wait(j,lambda s:s['ready_segments']==1)
        before=(Path(self.tmp.name)/j/'b.wav').read_bytes()
        self.m.translations(j,[{'id':'a','text_ru':'Привет'}])
        state=self.wait(j,lambda s:s['status'] in ('complete','failed'))
        self.assertEqual(state['status'],'complete',state['error'])
        self.assertTrue(state['ready_to_play'])
        self.assertEqual((Path(self.tmp.name)/j/'b.wav').read_bytes(),before)

    def test_maximum_tempo_boundary_has_room_for_filter_tail(self):
        from dubbing.render import fit_audio
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw.wav';out=Path(tmp)/'out.wav';wav(raw,4)
            duration=fit_audio(raw,out,2)
            self.assertLessEqual(duration,2+1/24000)
            self.assertGreater(duration,1.9)

    def test_next_real_20320ms_phrase_fits_13200ms(self):
        from dubbing.render import fit_audio
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw.wav';out=Path(tmp)/'out.wav';wav(raw,20.32)
            metrics={};duration=fit_audio(raw,out,13.2,metrics=metrics)
            self.assertLessEqual(duration,13.2+1/24000)
            self.assertGreater(duration,13)
            self.assertGreater(metrics['tempo'],1.5)
            self.assertTrue(metrics['accelerated'])
