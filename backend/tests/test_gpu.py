import time
import unittest
from pathlib import Path
from unittest.mock import patch

from dubbing.gpu import bounded_vae_decode, is_cuda_oom
from test_dubbing import QueueFixture, Source, Engine, PAYLOAD, wav


class DecodeTests(unittest.TestCase):
    def test_real_causal_decoder_matches_full_audio_and_restores_method(self):
        import torch
        from voxcpm.modules.audiovae.audio_vae_v2 import AudioVAE, AudioVAEConfig
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            torch.manual_seed(42)
            vae=AudioVAE(AudioVAEConfig(encoder_dim=4,encoder_rates=[2,2],latent_dim=4,
                decoder_dim=16,decoder_rates=[2,2],sr_bin_boundaries=None)).eval()
            original=vae.decode
            with torch.inference_mode():
                for frames in (1,4,5,17):
                    z=torch.randn(1,4,frames)
                    expected=original(z)
                    with bounded_vae_decode(vae,frames=4):actual=vae.decode(z)
                    self.assertEqual(actual.shape,expected.shape)
                    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-5)
                    self.assertEqual(vae.decode,original)
                with self.assertRaisesRegex(RuntimeError,'fixture'):
                    with bounded_vae_decode(vae):raise RuntimeError('fixture')
                self.assertEqual(vae.decode,original)
                forwards={name:module.forward for name,module in vae.decoder.named_modules()}
                calls=[]
                def fail_second(z,sr_cond=None):
                    calls.append(True)
                    if len(calls)==2:raise RuntimeError('CUDA out of memory. fixture')
                    return original(z,sr_cond)
                vae.decode=fail_second
                with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):
                    with bounded_vae_decode(vae,frames=4):vae.decode(torch.randn(1,4,9))
                self.assertIs(vae.decode,fail_second)
                self.assertEqual({name:module.forward for name,module in vae.decoder.named_modules()},forwards)
                vae.decode=original
        finally:torch.set_num_threads(threads)

    def test_only_cuda_memory_failure_is_retryable(self):
        self.assertTrue(is_cuda_oom(RuntimeError('CUDA out of memory. Tried to allocate')))
        self.assertFalse(is_cuda_oom(RuntimeError('CUDA device-side assert triggered')))
        self.assertFalse(is_cuda_oom(MemoryError('host memory exhausted')))


class GpuQueueTests(QueueFixture,unittest.TestCase):
    def configure(self,engine,probe,capacity_probe=None):
        self.m.close()
        self.patch=patch('dubbing.jobs.GPU_RETRY_SECONDS',.06);self.patch.start();self.addCleanup(self.patch.stop)
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=engine,memory_probe=probe,capacity_probe=capacity_probe)
        j=self.job();self.m.translations(j,[dict(id='a',text_ru='Понятный перевод.')]);return j

    def test_impossible_physical_budget_fails_without_loading_model(self):
        loads=[]
        class Tracked(Engine):
            def __init__(self):loads.append(True)
        jid=self.configure(Tracked,lambda:3000,capacity_probe=lambda:4000)
        state=self.wait(jid,lambda value:value['status']=='failed')
        self.assertEqual(loads,[])
        self.assertIn('ожидание не поможет',state['error'])
        self.assertIsNone(state['gpu_wait'])
        self.assertEqual(state['segments'][0]['text_ru'],'Понятный перевод.')

    def test_oom_cannot_increase_required_budget_beyond_physical_capacity_forever(self):
        calls=[]
        class Always(Engine):
            def synthesize(self,segment,path):
                calls.append(segment['id'])
                raise RuntimeError('CUDA out of memory. fixture')
        jid=self.configure(Always,lambda:6000,capacity_probe=lambda:6100)
        state=self.wait(jid,lambda value:value['status']=='failed')
        self.assertEqual(calls,['a'])
        self.assertIsNone(state['gpu_wait'])
        self.assertIn('объём GPU',state['error'])

    def test_preflight_waits_without_loading_model_then_resumes(self):
        memory=[1000];loads=[]
        class Tracked(Engine):
            def __init__(self):loads.append(True)
        j=self.configure(Tracked,lambda:memory[0])
        state=self.wait(j,lambda s:s['status']=='waiting_gpu')
        self.assertIsNone(state['error']);self.assertEqual(loads,[])
        self.assertEqual(state['gpu_wait']['available_mib'],1000)
        memory[0]=9000
        self.wait(j,lambda s:s['ready_segments']==1)
        self.assertEqual(len(loads),1)
        self.assertIsNone(self.m.summary(j)['gpu_wait'])

    def test_oom_releases_model_waits_for_more_memory_and_preserves_translation(self):
        memory=[6000];calls=[];closed=[]
        class Once(Engine):
            def synthesize(self,s,p):
                calls.append(s['text_ru'])
                if len(calls)==1:
                    wav(p)
                    raise RuntimeError('CUDA out of memory. fixture')
                return super().synthesize(s,p)
            def close(self):closed.append(True)
        j=self.configure(Once,lambda:memory[0])
        state=self.wait(j,lambda s:s['status']=='waiting_gpu')
        self.assertIsNone(state['error']);self.assertEqual(len(closed),1)
        self.assertFalse((Path(self.tmp.name)/j/'a.raw.wav').exists())
        time.sleep(.15);self.assertEqual(len(calls),1)
        memory[0]=9000
        self.wait(j,lambda s:s['ready_segments']==1)
        self.assertEqual(calls,['Понятный перевод.']*2)

    def test_repeated_oom_is_bounded_even_without_memory_telemetry(self):
        calls=[]
        class Always(Engine):
            def synthesize(self,s,p):
                calls.append(s['id']);raise RuntimeError('CUDA out of memory. fixture')
        j=self.configure(Always,lambda:None)
        state=self.wait(j,lambda s:s['status']=='failed')
        self.assertEqual(len(calls),3)
        self.assertIn('видеопамяти',state['error'])
        self.assertNotIn('Traceback',state['error'])
        self.assertEqual(state['segments'][0]['text_ru'],'Понятный перевод.')
        self.m.close()
        self.m=self.Manager(self.tmp.name,source_provider=Source(),engine_factory=Engine,autostart=False)
        self.m.create(PAYLOAD)
        self.assertEqual(self.m.jobs[j]['segments'][0]['state'],'pending')
        self.assertEqual(self.m.jobs[j]['segments'][0]['text_ru'],'Понятный перевод.')

    def test_cancel_during_resource_wait_prevents_late_generation(self):
        memory=[1000];calls=[]
        class Tracked(Engine):
            def synthesize(self,s,p):calls.append(s['id']);return super().synthesize(s,p)
        j=self.configure(Tracked,lambda:memory[0])
        self.wait(j,lambda s:s['status']=='waiting_gpu')
        self.m.cancel(j);memory[0]=9000;time.sleep(.2)
        self.assertEqual(calls,[]);self.assertEqual(self.m.summary(j)['status'],'cancelled')
