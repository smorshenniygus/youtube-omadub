"""Tests for LocDiT tail compaction policy, scatter/gather parity, and closest-length partner selection."""
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from dubbing.engine import VoxEngine
from dubbing.gpu import resolve_gpu_profile
from dubbing.jobs import JobManager, digest
from dubbing.vox_accel import choose_locdit_compact_batch


class LocDiTCompactionPolicyTests(unittest.TestCase):
    def test_ga106_width_policy(self):
        # B=8: 6..8 -> 8, 3..5 -> 5 (or 4 if B=4 warm and <=4 alive), 1..2 -> 2
        self.assertEqual(choose_locdit_compact_batch(8, 8, False), 8)
        self.assertEqual(choose_locdit_compact_batch(7, 8, False), 8)
        self.assertEqual(choose_locdit_compact_batch(6, 8, False), 8)
        self.assertEqual(choose_locdit_compact_batch(5, 8, False), 5)
        self.assertEqual(choose_locdit_compact_batch(5, 8, True), 5)
        self.assertEqual(choose_locdit_compact_batch(4, 8, False), 5)
        self.assertEqual(choose_locdit_compact_batch(4, 8, True), 4)
        self.assertEqual(choose_locdit_compact_batch(3, 8, False), 5)
        self.assertEqual(choose_locdit_compact_batch(3, 8, True), 4)
        self.assertEqual(choose_locdit_compact_batch(2, 8, True), 2)
        self.assertEqual(choose_locdit_compact_batch(1, 8, True), 2)

        # B=6: 6 -> 6, 5 -> 5, 3..4 -> 5 or warm 4, 1..2 -> 2
        self.assertEqual(choose_locdit_compact_batch(6, 6, False), 6)
        self.assertEqual(choose_locdit_compact_batch(5, 6, False), 5)
        self.assertEqual(choose_locdit_compact_batch(4, 6, True), 4)
        self.assertEqual(choose_locdit_compact_batch(2, 6, False), 2)

        # B=4: 3..4 stay 4, 1..2 -> 2
        self.assertEqual(choose_locdit_compact_batch(4, 4, False), 4)
        self.assertEqual(choose_locdit_compact_batch(3, 4, False), 4)
        self.assertEqual(choose_locdit_compact_batch(2, 4, False), 2)
        self.assertEqual(choose_locdit_compact_batch(1, 4, False), 2)

        # B=2 and B=1 never shrink further
        self.assertEqual(choose_locdit_compact_batch(1, 2, False), 2)
        self.assertEqual(choose_locdit_compact_batch(1, 1, False), 1)

    def test_gpu_profile_locdit_compact_defaults_and_overrides(self):
        p3060 = resolve_gpu_profile('NVIDIA GeForce RTX 3060', (8, 6), 11922, env={})
        self.assertTrue(p3060['locdit_tail_compaction'])

        p3060_off = resolve_gpu_profile(
            'NVIDIA GeForce RTX 3060', (8, 6), 11922, env={'VOXDUB_LOCDIT_COMPACT': '0'}
        )
        self.assertFalse(p3060_off['locdit_tail_compaction'])

        p4090 = resolve_gpu_profile('NVIDIA GeForce RTX 4090', (8, 9), 24564, env={})
        self.assertFalse(p4090['locdit_tail_compaction'])

        p4090_on = resolve_gpu_profile(
            'NVIDIA GeForce RTX 4090', (8, 9), 24564, env={'VOXDUB_LOCDIT_COMPACT': '1'}
        )
        self.assertTrue(p4090_on['locdit_tail_compaction'])

    def test_scatter_gather_middle_slots_finish_last_exact_parity(self):
        """Simultaneous stops + middle slots (3, 4) finishing last must match uncompacted output bit-for-bit."""
        # Planned stop steps (1-indexed patch count) for 8 slots:
        # slots 0,1,2 stop together at patch 4; slots 6,7 stop together at patch 6;
        # slot 5 stops at patch 8; middle slots 3 and 4 stop at patches 10 and 12!
        planned_stops = [4, 4, 4, 10, 12, 8, 6, 6]
        bsz, in_ch, patch_sz, h_dim = 8, 4, 2, 8
        observed_solver_widths = []

        class FakeKVCache:
            def __init__(self):
                self.kv_cache = torch.zeros((2, 1, bsz, 1, 64, 4), dtype=torch.float32)
                self.current_length = 0

            def step(self):
                pos = self.current_length
                self.current_length += 1
                return pos

        class FakeDecoder:
            in_channels = in_ch
            _cuda_graphs = {}

            def solve_euler(self, x, t_span, mu, cond, cfg_value, use_cfg_zero_star):
                observed_solver_widths.append(x.size(0))
                # Purely per-slot deterministic function of x, mu, cond
                m_bias = mu[:, :in_ch].unsqueeze(-1)
                return x * 0.5 + cond * 0.25 + m_bias * 0.1

        tts = types.SimpleNamespace(
            device='cpu',
            _dtype=lambda: torch.float32,
            patch_size=patch_sz,
            base_lm=types.SimpleNamespace(kv_cache=FakeKVCache()),
            residual_lm=types.SimpleNamespace(kv_cache=FakeKVCache()),
            feat_decoder=FakeDecoder(),
            lm_to_dit_proj=lambda h: h[:, :in_ch],
            res_to_dit_proj=lambda h: h[:, :in_ch],
            stop_proj=lambda h: h,
            stop_actn=lambda h: h,
            # Encode step counter in h[:, 0]; trigger stop when step >= planned_stops[slot] - 1
            stop_head=lambda h: torch.stack(
                [
                    torch.zeros(h.size(0)),
                    (h[:, 0] >= torch.tensor([s - 1 for s in planned_stops], dtype=torch.float32)).float() * 2.0 - 0.5,
                ],
                dim=-1,
            ),
            feat_encoder=lambda x: x,
            enc_to_lm_proj=lambda x: x,
            _enc_graph_disabled=True,
        )

        def fake_enc(model, pred_feat):
            # Preserve per-slot signal cleanly
            return pred_feat.mean(dim=1).unsqueeze(1).repeat(1, 1, h_dim // in_ch)

        def fake_lm(model, curr_embed_2d, pos, slot_groups=None):
            # max_pre = max(6 + g for g in 0..3) = 9, so step 0 -> pos=9 -> next step_idx=1
            step_idx = float(pos - 9 + 1)
            lm_h = curr_embed_2d * 0.5
            lm_h[:, 0] = step_idx
            res_h = curr_embed_2d * 0.25
            return lm_h, res_h

        def make_items():
            items = []
            for g in range(4):
                l_i = 6 + g
                lm0 = torch.zeros((1, h_dim))
                lm0[0, 1:] = g + 1.0
                items.append({
                    'prefill_state': {
                        'prefill_len': l_i,
                        'lm_hidden': lm0,
                        'residual_hidden': torch.ones((1, h_dim)) * (g + 0.5),
                        'prefix_feat_cond': torch.ones((1, patch_sz, in_ch)) * (g + 0.2),
                        'context_patches': None,
                        'target_text_length': 20,
                    },
                    'base_kv': torch.zeros((2, 1, 1, 1, l_i, 4)),
                    'res_kv': torch.zeros((2, 1, 1, 1, l_i, 4)),
                    'seeds': [100 + 2 * g, 101 + 2 * g],
                })
            return items

        model = types.SimpleNamespace(tts_model=tts)
        eng = VoxEngine(model=model)

        with mock.patch('dubbing.engine._run_enc_step_graphed', side_effect=fake_enc), \
             mock.patch('dubbing.engine._run_lm_step_fast', side_effect=fake_lm):
            eng._locdit_tail_compaction = False
            observed_solver_widths.clear()
            ref_out = eng._run_autoregressive_multi(make_items(), steps=10, cfg=2.0, max_len_cap=30)
            ref_hist = dict(eng.last_alive_hist)

            eng._locdit_tail_compaction = True
            observed_solver_widths.clear()
            cmp_out = eng._run_autoregressive_multi(make_items(), steps=10, cfg=2.0, max_len_cap=30)
            cmp_hist = dict(eng.last_alive_hist)

        self.assertEqual(ref_hist, cmp_hist)
        # Alive counts across patches 0..11:
        # i=0..3: 8 alive -> B=8
        # i=4..5: 5 alive (0,1,2 stopped on i=3) -> B=5
        # i=6..7: 3 alive (6,7 stopped on i=5) -> B=5 (no B=4 graph warm)
        # i=8..9: 2 alive (5 stopped on i=7) -> B=2
        # i=10..11: 1 alive (3 stopped on i=9; middle slot 4 finishes at i=11) -> B=2
        self.assertEqual(observed_solver_widths, [8, 8, 8, 8, 5, 5, 5, 5, 2, 2, 2, 2])

        flat_ref = [x for item in ref_out for x in item]
        flat_cmp = [x for item in cmp_out for x in item]
        for slot_i, ((rf, rseq, rbad), (cf, cseq, cbad)) in enumerate(zip(flat_ref, flat_cmp)):
            self.assertEqual(rseq.shape[0], planned_stops[slot_i], f'slot {slot_i} stop step mismatch')
            self.assertEqual(cseq.shape[0], planned_stops[slot_i], f'slot {slot_i} compacted stop step mismatch')
            self.assertEqual(rbad, cbad)
            self.assertTrue(torch.equal(rf, cf), f'slot {slot_i} output tensor diverged')


class ClosestPartnerSelectionTests(unittest.TestCase):
    def test_closest_partners_prefers_length_match_over_first_passing(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'ref.wav'
            ref.write_bytes(b'RIFF0000WAVE')
            ref_sha = digest(ref)
            mgr = JobManager(td, autostart=False, engine_factory=lambda: None)
            try:
                mgr.engine = types.SimpleNamespace(
                    max_batch_size=8,
                    needs_anchor=lambda view: False,
                    _should_rerank=lambda view, text: (True, 4.0),
                )
                # Anchor segment: 50 chars, past startup buffer (start=100s, priority=0)
                s0 = {
                    'id': 's0', 'state': 'generating', 'start': 100.0, 'end': 104.0,
                    'speaker': 'spk0', 'reference': str(ref), 'reference_sha256': ref_sha,
                    'text_ru': 'a' * 50,
                }
                # First 3 candidates pass ratio<=1.35 (len=66 -> ratio=1.32),
                # next 3 candidates are exact matches (len=50 -> ratio=1.00).
                cands = []
                for idx, length in enumerate([66, 66, 66, 50, 50, 50], start=1):
                    cands.append({
                        'id': f's{idx}', 'state': 'pending', 'start': 104.0 + idx * 4.0, 'end': 108.0 + idx * 4.0,
                        'speaker': 'spk0', 'reference': str(ref), 'reference_sha256': ref_sha,
                        'text_ru': 'b' * length,
                    })
                job = {'id': 'j1', 'cancelled': False, 'error': None, 'priority': 0.0, 'segments': [s0] + cands}

                # With closest_partners=True -> picks exact-length s4, s5, s6
                mgr.closest_partners = True
                chosen_closest, _ = mgr._select_batch_partners(job, s0, dict(s0, job_id='j1'), {}, threading.Lock())
                self.assertEqual({c['id'] for c in chosen_closest}, {'s4', 's5', 's6'})

                # Reset states and verify closest_partners=False -> picks first-passing s1, s2, s3
                for c in cands:
                    c['state'] = 'pending'
                mgr.closest_partners = False
                chosen_first, _ = mgr._select_batch_partners(job, s0, dict(s0, job_id='j1'), {}, threading.Lock())
                self.assertEqual([c['id'] for c in chosen_first], ['s1', 's2', 's3'])
            finally:
                mgr.close()


if __name__ == '__main__':
    unittest.main()
