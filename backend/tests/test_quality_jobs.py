import copy
import threading
import time
from pathlib import Path
import unittest

from test_dubbing import QueueFixture, Source, Engine, PAYLOAD, wav


class Editor:
    def __init__(self):self.calls=[];self.last_metrics={'model':'fixture','elapsed_seconds':.01}
    def edit(self,items,context=None,cancelled=None):
        self.calls.append((copy.deepcopy(items),copy.deepcopy(context)))
        return [dict(id=x['id'],text_ru='Кратко.' if x.get('feedback') else 'Понятный перевод.') for x in items]


class QualityTests(QueueFixture, unittest.TestCase):
    def quality(self,editor=None,engine=Engine,autostart=True):
        self.m.close();self.editor=editor or Editor()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=engine,
                            editor=self.editor,quality=True,autostart=autostart)

    def test_draft_is_edited_with_neighboring_context_before_synthesis(self):
        received=[]
        class Tracking(Engine):
            def synthesize(self,s,p):received.append(s['text_ru']);return super().synthesize(s,p)
        self.quality(engine=Tracking);j=self.job()
        self.m.translations(j,[{'id':'a','text_ru':'Плохая калька'}])
        state=self.wait(j,lambda s:s['ready_segments']==1)
        self.assertEqual(received,['Понятный перевод.'])
        self.assertEqual(state['segments'][0]['text_ru'],'Понятный перевод.')
        self.assertTrue(any(row['source']=='World' for row in self.editor.calls[0][1]))
        self.assertEqual(self.editor.calls[0][0][0]['draft'],'Плохая калька')

    def test_new_profile_preserves_old_audio_and_draft_idempotency(self):
        legacy=self.job();self.m.translations(legacy,[{'id':'a','text_ru':'Черновик'}])
        self.wait(legacy,lambda s:s['ready_segments']==1)
        old=(Path(self.tmp.name)/legacy/'a.wav').read_bytes()
        self.quality();quality=self.job();self.assertNotEqual(quality,legacy)
        self.m.translations(quality,[{'id':'a','text_ru':'Черновик'}])
        self.wait(quality,lambda s:s['ready_segments']==1)
        self.m.translations(quality,[{'id':'a','text_ru':'Черновик'}])
        with self.assertRaises(ValueError):self.m.translations(quality,[{'id':'a','text_ru':'Другое'}])
        self.assertEqual((Path(self.tmp.name)/legacy/'a.wav').read_bytes(),old)
        self.assertEqual(self.m.summary(legacy)['ready_segments'],1)

    def test_validated_speech_within_tempo_cap_is_not_sent_back_for_unnecessary_rewrite(self):
        calls=[]
        class FitsEngine:
            def synthesize(self,segment,path):
                calls.append(segment['text_ru']);wav(path,2.4)
                return {'raw_audio_seconds':2.4}
        self.quality(engine=FitsEngine);jid=self.job()
        self.m.translations(jid,[{'id':'a','text_ru':'Черновик'}])
        state=self.wait(jid,lambda value:value['ready_segments']==1 or value['status']=='failed')
        self.assertNotEqual(state['status'],'failed',state['error'])
        self.assertEqual(len(self.editor.calls),1)
        self.assertEqual(len(calls),1)
        self.assertEqual(state['segments'][0]['metrics']['rewrite_attempts'],0)
        self.assertLessEqual(state['segments'][0]['metrics']['tempo'],1.25)

    def test_measured_overflow_rewrites_before_retry_instead_of_twofold_tempo(self):
        received=[]
        class MeasuredEngine:
            def synthesize(self,s,p):
                received.append(s['text_ru']);seconds=1.6 if s['text_ru']=='Кратко.' else 3
                wav(p,seconds);return {'raw_audio_seconds':seconds}
        self.quality(engine=MeasuredEngine);j=self.job()
        self.m.translations(j,[{'id':'a','text_ru':'Исходный черновик'}])
        state=self.wait(j,lambda s:s['ready_segments']==1 or s['status']=='failed')
        self.assertNotEqual(state['status'],'failed',state['error'])
        self.assertEqual(received,['Понятный перевод.','Кратко.'])
        self.assertTrue(self.editor.calls[1][0][0]['feedback'])
        self.assertLessEqual(state['segments'][0]['metrics']['tempo'],1.25)
        self.assertEqual(state['segments'][0]['metrics']['rewrite_attempts'],1)

    def test_editor_failure_never_synthesizes_raw_draft(self):
        received=[]
        class BadEditor(Editor):
            def edit(self,*args,**kwargs):raise RuntimeError('editor unavailable')
        class Tracking(Engine):
            def synthesize(self,s,p):received.append(s);return super().synthesize(s,p)
        self.quality(editor=BadEditor(),engine=Tracking);j=self.job()
        self.m.translations(j,[{'id':'a','text_ru':'Черновик'}])
        state=self.wait(j,lambda s:s['status']=='failed')
        self.assertFalse(state['ready_to_play']);self.assertEqual(received,[])
        self.assertEqual(self.m.summary(j)['segments'][0]['text_ru'],'')

    def test_numeric_failure_retains_diagnostic_without_synthesizing(self):
        import json
        from dubbing.editing import NumericEditingError, REVISION
        class BadEditor(Editor):
            def edit(self, *args, **kwargs):
                error = NumericEditingError([{'id':'a','required_digits':[{'value':'20','count':1}]}])
                error.previous_output = {'items':[{'id':'a','text_ru':'Неверно.'}]}
                raise error
        self.quality(editor=BadEditor()); j=self.job()
        self.m.translations(j,[{'id':'a','text_ru':'Черновик'}])
        state=self.wait(j,lambda s:s['status']=='failed')
        self.assertEqual(state['ready_segments'],0)
        with self.m.lock:
            saved=json.loads((Path(self.tmp.name)/j/'job.json').read_text())
        diagnostic=saved['editing_failure']
        self.assertEqual(diagnostic['revision'],REVISION)
        self.assertEqual(diagnostic['items'][0]['id'],'a')
        self.assertEqual(diagnostic['output']['items'][0]['text_ru'],'Неверно.')
        self.assertEqual(diagnostic['feedback'][0]['required_digits'][0]['value'],'20')

    def test_cancelled_edit_cannot_submit_audio(self):
        entered=threading.Event();released=threading.Event();received=[]
        class Blocking(Editor):
            def edit(self,*args,cancelled=None,**kwargs):
                entered.set()
                while not released.wait(.01):
                    if cancelled():raise InterruptedError('cancelled')
                return super().edit(*args,**kwargs)
        class Tracking(Engine):
            def synthesize(self,s,p):received.append(s);return super().synthesize(s,p)
        self.quality(editor=Blocking(),engine=Tracking);j=self.job()
        self.m.translations(j,[{'id':'a','text_ru':'Черновик'}]);self.assertTrue(entered.wait(2))
        self.m.cancel(j);released.set();time.sleep(.15)
        self.assertEqual(self.m.summary(j)['status'],'cancelled');self.assertEqual(received,[])

    def test_excessive_speech_gets_only_two_rewrites_then_explicit_failure(self):
        attempts=[]
        class Long:
            def synthesize(self,s,p):attempts.append(s['text_ru']);wav(p,4);return {'raw_audio_seconds':4}
        self.quality(engine=Long)
        payload = dict(PAYLOAD, cues=PAYLOAD['cues'][:1])
        j = self.m.create(payload)['id']
        self.wait(j, lambda s: s['status'] == 'waiting_translation')
        self.m.translations(j,[{'id':'a','text_ru':'Черновик'}])
        state=self.wait(j,lambda s:s['status']=='failed')
        self.assertEqual(len(attempts),3);self.assertFalse(state['ready_to_play'])
        # Feedback carries measured budget and previous_attempts history.
        second_fb=self.editor.calls[1][0][0]['feedback']
        self.assertEqual(second_fb['max_raw_seconds'],2.5)
        self.assertEqual(second_fb['previous_attempts'][0]['text_ru'],'Понятный перевод.')
        # Explicit retry after timing error preserves previous_attempts instead of looping blindly.
        self.m.create(payload)
        state=self.wait(j,lambda s:s['status']=='failed' and len(attempts)==6)
        retry_call=self.editor.calls[3][0][0]
        self.assertEqual(retry_call['draft'],'Черновик')
        self.assertTrue(retry_call['feedback']['previous_attempts'])
        self.assertEqual(retry_call['feedback']['previous_attempts'][0]['text_ru'],'Понятный перевод.')

    def test_timing_failure_does_not_block_other_phrases_or_seeking(self):
        class MixedLength:
            def synthesize(self, segment, path):
                wav(path, 4 if segment['id'] == 'a' else .5)

        self.quality(engine=MixedLength)
        job = self.job()
        self.m.translations(job, [{'id':'a', 'text_ru':'Черновик'}])
        partial = self.wait(job, lambda s: s['segments'][0].get('error'))
        self.assertEqual(partial['status'], 'waiting_translation')
        self.assertIsNone(partial['error'])
        self.m.translations(job, [{'id':'b', 'text_ru':'Мир'}])
        result = self.wait(job, lambda s: s['ready_segments'] == 1 and s['segments'][0].get('error'))
        self.assertEqual(result['status'], 'complete')
        self.assertIsNone(result['error'])
        self.assertEqual(result['skipped_segments'], 1)
        self.assertEqual(result['segments'][0]['state'], 'skipped')
        self.assertNotIn('audio_url', result['segments'][0])
        self.assertEqual(result['ranges'], [[0, 10]])
        self.assertTrue(self.m.priority(job, 2)['ready_to_play'])
        self.assertTrue(self.m.priority(job, 6)['ready_to_play'])
        self.assertTrue(self.m.audio_path(job, 'b').is_file())

    def test_short_window_target_chars_not_clamped_up_to_twelve(self):
        self.quality()
        payload=copy.deepcopy(PAYLOAD)
        payload['cues']=[dict(id='a',start=0,end=0.65,text='A coin flip.')]
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[{'id':'a','text_ru':'Подбрасывание монеты.'}])
        self.wait(j,lambda s:s['ready_segments']==1 or s['status']=='failed')
        self.assertEqual(self.editor.calls[0][0][0]['target_chars'],8)

    def test_exact_tempo_cap_filter_tail_overflow_triggers_rewrite_instead_of_immediate_failure(self):
        received=[]
        class BoundaryEngine:
            def synthesize(self,s,p):
                received.append(s['text_ru'])
                # 2.400s for 1.920s is exactly 1.25x: bypasses pre-check, fails in ffmpeg atempo tail.
                seconds=1.76 if s['text_ru']=='Кратко.' else 2.4
                wav(p,seconds);return {'raw_audio_seconds':seconds}
        self.quality(engine=BoundaryEngine)
        payload=copy.deepcopy(PAYLOAD)
        payload['cues']=[dict(id='a',start=0,end=1.92,text='It is more messy and empirical.')]
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[{'id':'a','text_ru':'Черновик'}])
        state=self.wait(j,lambda s:s['ready_segments']==1 or s['status']=='failed')
        self.assertNotEqual(state['status'],'failed',state['error'])
        self.assertEqual(received,['Понятный перевод.','Кратко.'])
        self.assertEqual(state['segments'][0]['metrics']['rewrite_attempts'],1)
        self.assertLessEqual(state['segments'][0]['metrics']['tempo'],1.25)
        self.assertEqual(self.editor.calls[1][0][0]['feedback']['previous_attempts'][0]['text_ru'],'Понятный перевод.')

    def test_non_timing_render_error_does_not_trigger_rewrite(self):
        import wave as _wave
        class SilentEngine:
            def synthesize(self,s,p):
                with _wave.open(str(p),'wb') as f:
                    f.setparams((1,2,8000,0,'NONE','not compressed'))
                    f.writeframes(b'\x00'*8000)
                return {'raw_audio_seconds':0.5}
        self.quality(engine=SilentEngine);j=self.job()
        self.m.translations(j,[{'id':'a','text_ru':'Черновик'}])
        state=self.wait(j,lambda s:s['status']=='failed')
        self.assertIn('silent waveform',state['error'])
        self.assertEqual(len(self.editor.calls),1)

    def test_restart_recovers_interrupted_edit_without_losing_draft(self):
        self.quality();j=self.job();self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine,
                            editor=self.editor,quality=True,autostart=False)
        self.m.translations(j,[{'id':'a','text_ru':'Черновик'}])
        row=self.m.jobs[j]['segments'][0];row['state']='editing';self.m._save(self.m.jobs[j])
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine,
                                          editor=self.editor,quality=True)
        self.wait(j,lambda s:s['ready_segments']==1)
        self.assertEqual(self.editor.calls[-1][0][0]['draft'],'Черновик')

    def test_sparse_edit_batch_uses_only_local_bounded_context(self):
        from dubbing.editing import CodexEditor
        class CheckedEditor(Editor):
            def edit(self,items,context=None,**kwargs):
                CodexEditor._inputs(items,context)
                return super().edit(items,context=context,**kwargs)
        self.quality(editor=CheckedEditor())
        payload=dict(copy.deepcopy(PAYLOAD),duration=180,cues=[dict(id=str(i),start=i*3,end=i*3+2,text='Neighbor '+str(i)) for i in range(60)])
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine,editor=self.editor,quality=True,autostart=False)
        self.m.translations(j,[{'id':'0','text_ru':'Первый'},{'id':'50','text_ru':'Дальний'}])
        self.m.close();self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine,editor=self.editor,quality=True)
        state=self.wait(j,lambda s:s['ready_segments']==2 or s['status']=='failed')
        self.assertNotEqual(state['status'],'failed',state['error'])
        self.assertLessEqual(len(self.editor.calls[0][1]),8)


class QualityTempoCapTests(unittest.TestCase):
    def test_published_quality_audio_never_exceeds_requested_tempo_cap(self):
        import tempfile
        from dubbing.render import fit_audio
        for seconds in (2.49,2.5):
            with self.subTest(seconds=seconds),tempfile.TemporaryDirectory() as tmp:
                raw=Path(tmp)/'raw.wav';out=Path(tmp)/'out.wav';wav(raw,seconds);metrics={}
                try:fit_audio(raw,out,2,metrics=metrics,max_tempo=1.25)
                except ValueError:self.assertFalse(out.exists())
                else:self.assertLessEqual(metrics['tempo'],1.25)

class FlowQueueTests(QueueFixture, unittest.TestCase):
    def test_forced_continuation_publishes_stable_schedule_and_survives_restart(self):
        self.m.close()
        class FlowSource(Source):
            def prepare(self,*args,**kwargs):
                result=super().prepare(*args,**kwargs)
                result['segments'][1]['continues_previous']=True
                return result
        class FlowEngine:
            def synthesize(self,s,p):wav(p,2 if s['id']=='a' else .5)
        self.m=self.Manager(self.tmp.name,source_provider=FlowSource(),engine_factory=FlowEngine,editor=Editor(),quality=True)
        payload=copy.deepcopy(PAYLOAD)
        payload['cues']=[dict(id='a',start=1,end=6,text='First part of'),dict(id='b',start=6,end=8,text='the sentence.')]
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[dict(id='a',text_ru='Первая часть'),dict(id='b',text_ru='предложения.')])
        state=self.wait(j,lambda s:s['status'] in ('complete','failed'))
        self.assertEqual(state['status'],'complete',state['error'])
        first,second=state['segments']
        self.assertAlmostEqual(second['start'],first['start']+first['audio_duration']+.12)
        self.assertEqual(first['end'],second['start'])
        self.assertEqual(second['source_start'],6)
        self.assertEqual(self.m.jobs[j]['segments'][0]['end'],6)
        self.assertEqual(state['ready_until'],10)
        original_schedule=[(s['start'],s['end']) for s in state['segments']]
        self.m.close()
        self.m=self.Manager(self.tmp.name,autostart=False,quality=True)
        self.assertEqual([(s['start'],s['end']) for s in self.m.summary(j)['segments']],original_schedule)


class ShortWindowTests(QueueFixture, unittest.TestCase):
    def test_short_speech_uses_only_verified_previous_silence(self):
        from dubbing.jobs import speech_start
        prev=dict(start=44.559,end=56.239,state='ready',audio_duration=6.876)
        row=dict(start=56.239,end=56.96)
        self.assertAlmostEqual(speech_start([prev,row],row,1.12),55.989)
        prev['state']='pending';self.assertEqual(speech_start([prev,row],row,1.12),row['start'])
        prev.update(state='ready',audio_duration=11.68)
        self.assertEqual(speech_start([prev,row],row,1.12),row['start'])
        prev['audio_duration']=6.876
        self.assertEqual(speech_start([prev,row],row,2),row['start'])

    def test_real_short_window_publishes_nonoverlapping_audio_schedule(self):
        self.m.close()
        class FixedEngine:
            def synthesize(self,s,p):wav(p,.2 if s['id']=='a' else 1.12)
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=FixedEngine,editor=Editor(),quality=True)
        payload=copy.deepcopy(PAYLOAD);payload['cues']=[dict(id='a',start=1,end=3,text='Previous'),dict(id='b',start=3,end=3.721,text='Short')]
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[dict(id='a',text_ru='Предыдущая'),dict(id='b',text_ru='Исправим')])
        state=self.wait(j,lambda s:s['ready_segments']==2 or s['status']=='failed')
        self.assertEqual(state['ready_segments'],2,state['error'])
        a,b=state['segments'];self.assertEqual(a['end'],b['start']);self.assertEqual(b['start'],2.75)
        self.assertLessEqual(a['start']+a['audio_duration'],a['end'])
        self.assertLessEqual(b['audio_duration'],b['end']-b['start'])
        self.assertLessEqual(b['metrics']['tempo'],1.25)
        self.assertEqual(self.m.jobs[j]['segments'][0]['end'],3)
        self.assertEqual(self.m.jobs[j]['segments'][1]['start'],3)
        self.m.close()
        (Path(self.tmp.name)/j/'a.wav').write_bytes(b'invalidated')
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=FixedEngine,editor=Editor(),quality=True,autostart=False)
        self.assertEqual([s['state'] for s in self.m.jobs[j]['segments']],['pending','pending'])

    def test_micro_window_filler_recovers_after_rewrites_without_failing_job(self):
        class MicroEngine:
            def synthesize(self,s,p):wav(p,1.12);return {'raw_audio_seconds':1.12}
        self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=MicroEngine,editor=Editor(),quality=True)
        payload=copy.deepcopy(PAYLOAD)
        payload['cues']=[dict(id='a',start=1.0,end=1.23,text='Uh')]
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[dict(id='a',text_ru='Э-э.')])
        state=self.wait(j,lambda s:s['status'] in ('complete','failed'))
        self.assertEqual(state['status'],'complete',state['error'])
        self.assertEqual(state['ready_segments'],1)
        self.assertTrue(state['segments'][0]['metrics'].get('micro_window_passthrough'))
        self.assertLessEqual(state['segments'][0]['audio_duration'],0.23)

    def test_micro_window_with_dead_silence_trims_and_fits_real_speech(self):
        import math, wave as _wave
        from array import array as _array
        class PaddedMicroEngine:
            def synthesize(self,s,p):
                sr=48000
                sil=[0]*int(sr*0.35)
                tone=[int(12000*math.sin(2*math.pi*260*i/sr)) for i in range(int(sr*0.40))]
                samples=_array('h',sil+tone+sil)
                with _wave.open(str(p),'wb') as f:
                    f.setparams((1,2,sr,0,'NONE','not compressed'))
                    f.writeframes(samples.tobytes())
                return {'raw_audio_seconds':1.10}
        self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=PaddedMicroEngine,editor=Editor(),quality=True)
        payload=copy.deepcopy(PAYLOAD)
        payload['cues']=[dict(id='a',start=1.0,end=1.40,text='And I think')]
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[dict(id='a',text_ru='Думаю')])
        state=self.wait(j,lambda s:s['status'] in ('complete','failed'))
        self.assertEqual(state['status'],'complete',state['error'])
        m=state['segments'][0]['metrics']
        self.assertGreater(m.get('silence_trimmed_seconds',0),0.5)
        self.assertLessEqual(m['tempo'],1.25)
        self.assertEqual(m['rewrite_attempts'],0)
        self.assertLessEqual(state['segments'][0]['audio_duration'],0.40+1/48000)

    def test_dead_silence_trim_fits_under_125_without_unnecessary_rewrite(self):
        import math, wave as _wave
        from array import array as _array
        class PaddedEngine:
            def synthesize(self,s,p):
                sr=48000
                sil=[0]*int(sr*0.20)
                tone=[int(12000*math.sin(2*math.pi*220*i/sr)) for i in range(int(sr*1.68))]
                samples=_array('h',sil+tone+sil) # 2.08s raw for 1.463s window, but 1.72s trimmed -> 1.18x!
                with _wave.open(str(p),'wb') as f:
                    f.setparams((1,2,sr,0,'NONE','not compressed'))
                    f.writeframes(samples.tobytes())
                return {'raw_audio_seconds':2.08}
        self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=PaddedEngine,editor=Editor(),quality=True)
        payload=copy.deepcopy(PAYLOAD)
        payload['cues']=[dict(id='a',start=1.0,end=2.463,text='He said 36 months or')]
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[dict(id='a',text_ru='Сказал: 36 месяцев,')])
        state=self.wait(j,lambda s:s['status'] in ('complete','failed'))
        self.assertEqual(state['status'],'complete',state['error'])
        m=state['segments'][0]['metrics']
        self.assertEqual(m['rewrite_attempts'],0)
        self.assertLessEqual(m['tempo'],1.25)
        self.assertGreater(m.get('silence_trimmed_seconds',0),0.25)

    def test_mandatory_number_142x_recovers_after_two_rewrites_via_fallback_tempo(self):
        attempts=[]
        class TightNumberEngine:
            def synthesize(self,s,p):attempts.append(s['text_ru']);wav(p,2.08);return {'raw_audio_seconds':2.08}
        self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=TightNumberEngine,editor=Editor(),quality=True)
        payload=copy.deepcopy(PAYLOAD)
        payload['cues']=[dict(id='a',start=1.0,end=2.463,text='He said 36 months or')]
        j=self.m.create(payload)['id'];self.wait(j,lambda s:s['status']=='waiting_translation')
        self.m.translations(j,[dict(id='a',text_ru='Сказал: 36 месяцев,')])
        state=self.wait(j,lambda s:s['status'] in ('complete','failed'))
        self.assertEqual(state['status'],'complete',state['error'])
        self.assertEqual(len(attempts),3) # Tried 2 LLM rewrites first!
        m=state['segments'][0]['metrics']
        self.assertTrue(m.get('tempo_fallback'))
        self.assertLessEqual(m['tempo'],1.75)
        self.assertLessEqual(state['segments'][0]['audio_duration'],1.463+1/48000)
