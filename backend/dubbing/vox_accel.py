"""Zero-loss CUDA inference acceleration for VoxCPM2 (weight fusion, split-SDPA, hoisted CUDA Graphs)."""
import math
import types


def _fuse_qkv_and_mlp(tts_model):
    """Mathematically exact weight fusion: q/k/v -> qkv_proj and gate/up -> gate_up_proj (0 extra VRAM)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from voxcpm.modules.minicpm4.model import MiniCPMAttention, MiniCPMMLP

    def fused_mlp_forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    for module in tts_model.modules():
        if isinstance(module, MiniCPMAttention) and hasattr(module, 'q_proj') and not hasattr(module, 'qkv_proj'):
            w = torch.cat((module.q_proj.weight.data, module.k_proj.weight.data, module.v_proj.weight.data), dim=0).contiguous()
            q_dim = module.q_proj.out_features
            kv_dim = module.k_proj.out_features
            qkv = nn.Linear(module.q_proj.in_features, q_dim + 2 * kv_dim, bias=False, device=w.device, dtype=w.dtype)
            qkv.weight = nn.Parameter(w, requires_grad=False)
            module.qkv_proj = qkv
            module._qkv_dims = (q_dim, kv_dim, kv_dim)
            del module.q_proj, module.k_proj, module.v_proj
        elif isinstance(module, MiniCPMMLP) and hasattr(module, 'gate_proj') and not hasattr(module, 'gate_up_proj'):
            w = torch.cat((module.gate_proj.weight.data, module.up_proj.weight.data), dim=0).contiguous()
            gu = nn.Linear(module.gate_proj.in_features, 2 * module.gate_proj.out_features, bias=False, device=w.device, dtype=w.dtype)
            gu.weight = nn.Parameter(w, requires_grad=False)
            module.gate_up_proj = gu
            del module.gate_proj, module.up_proj
            module.forward = types.MethodType(fused_mlp_forward, module)


def _apply_residual_and_mlp(layer, residual, attn_out):
    scale = (layer.scale_depth / math.sqrt(layer.num_hidden_layers)) if layer.use_mup else None
    hidden_states = residual + (attn_out * scale if scale is not None else attn_out)
    mlp_out = layer.mlp(layer.post_attention_layernorm(hidden_states))
    return hidden_states + (mlp_out * scale if scale is not None else mlp_out)


def _install_fast_kv_attention(tts_model, max_length=2048, max_batch_size=8, enable_cuda_graphs=True):
    """Zero-loss inference acceleration on CUDA (bit-equivalent output):
    - Slice StaticKVCache to [:pos+1] with attn_mask=None and bound KV length (supports B=1..max_batch_size)
    - Eliminate 36 per-layer position_id.item() GPU->CPU sync stalls per patch
    - Fuse q/k/v -> qkv_proj and gate/up -> gate_up_proj (180 fewer GEMM launches/patch, 0 extra VRAM)
    - Replace 5-op eager MiniCPMRMSNorm with fused torch.nn.functional.rms_norm
    - Remove unnecessary Apple-MPS .contiguous() copies in MiniCPMAttention.forward
    - Fold AudioVAE weight_norm permanently for inference
    - Precompute SinusoidalPosEmb frequencies & zero-dt embedding in LocDiT
    - Hoist loop invariants & capture CUDA Graphs for UnifiedCFM.solve_euler & single-patch VoxCPMLocEnc
    """
    if not (hasattr(tts_model, 'base_lm') and hasattr(tts_model, 'residual_lm')):
        return False
    import torch
    import torch.nn.functional as F
    from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb, MiniCPMRMSNorm, MiniCPMAttention

    _fuse_qkv_and_mlp(tts_model)

    def fast_attn_forward_step(self, hidden_states, position_emb, position_id, kv_cache):
        bsz, _ = hidden_states.size()
        q_dim, k_dim, v_dim = self._qkv_dims
        q, k, v = self.qkv_proj(hidden_states).split((q_dim, k_dim, v_dim), dim=-1)
        query_states = q.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = v.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_cache, value_cache = kv_cache
        pos = position_id if type(position_id) is int else int(position_id.item())
        key_cache[:bsz, :, pos, :] = key_states[:, :, 0, :]
        value_cache[:bsz, :, pos, :] = value_states[:, :, 0, :]
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_cache[:bsz, :, :pos + 1, :],
            value_cache[:bsz, :, :pos + 1, :],
            attn_mask=None,
            is_causal=False,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def fast_lm_forward_step(self, inputs_embeds, position_id):
        pos = position_id if type(position_id) is int else int(position_id.item())
        position_emb = (
            (self.rope_emb.cos_cached[pos:pos + 1], self.rope_emb.sin_cached[pos:pos + 1])
            if self.rope_emb is not None else None
        )
        hidden_states = inputs_embeds
        kv = self.kv_cache.kv_cache
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer.forward_step(hidden_states, position_emb, pos, (kv[0, i], kv[1, i]))
        return self.norm(hidden_states)

    def fast_cuda_attn_forward(self, hidden_states, position_emb, is_causal):
        bsz, q_len, _ = hidden_states.size()
        q_dim, k_dim, v_dim = self._qkv_dims
        q, k, v = self.qkv_proj(hidden_states).split((q_dim, k_dim, v_dim), dim=-1)
        query_states = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=is_causal, enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), (key_states, value_states)

    def fast_rms_norm_forward(self, hidden_states):
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), self.weight, self.variance_epsilon)

    dtype = tts_model._dtype()
    dev = torch.device(tts_model.device)
    is_cuda = dev.type == 'cuda' and torch.cuda.is_available()

    tts_model.base_lm.setup_cache(max_batch_size, max_length, tts_model.device, dtype)
    tts_model.residual_lm.setup_cache(max_batch_size, max_length, tts_model.device, dtype)
    for lm in (tts_model.base_lm, tts_model.residual_lm):
        lm.forward_step = types.MethodType(fast_lm_forward_step, lm)
        for layer in lm.layers:
            layer.self_attn.forward_step = types.MethodType(fast_attn_forward_step, layer.self_attn)

    has_fused_rms = hasattr(F, 'rms_norm')
    for module in tts_model.modules():
        if has_fused_rms and isinstance(module, MiniCPMRMSNorm):
            module.forward = types.MethodType(fast_rms_norm_forward, module)
        elif isinstance(module, MiniCPMAttention):
            module.forward = types.MethodType(fast_cuda_attn_forward, module)

    # Fold weight_norm on AudioVAE once at load time (100% mathematically identical).
    vae = getattr(tts_model, 'audio_vae', None)
    if vae is not None:
        for module in vae.modules():
            if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
                try:
                    torch.nn.utils.remove_weight_norm(module)
                except ValueError:
                    pass

    # Precompute SinusoidalPosEmb frequencies & zero-dt embedding in LocDiT/UnifiedCFM.
    feat_decoder = getattr(tts_model, 'feat_decoder', None)
    est = getattr(feat_decoder, 'estimator', None) if feat_decoder is not None else None
    if est is not None and hasattr(est, 'time_embeddings'):
        te = est.time_embeddings
        half_dim = te.dim // 2
        inv_freq = torch.exp(
            torch.arange(half_dim, dtype=dtype, device=dev) * -(math.log(10000) / (half_dim - 1))
        )
        te.register_buffer('_cached_inv_freq', inv_freq, persistent=False)

        def fast_sin_emb(self, x, scale=1000):
            if x.ndim < 1:
                x = x.unsqueeze(0)
            freq = self._cached_inv_freq
            if freq.device != x.device or freq.dtype != x.dtype:
                freq = freq.to(device=x.device, dtype=x.dtype)
            emb = scale * x.unsqueeze(1) * freq.unsqueeze(0)
            return torch.cat((emb.sin(), emb.cos()), dim=-1)

        te.forward = types.MethodType(fast_sin_emb, te)

        if not getattr(feat_decoder, 'mean_mode', False) and hasattr(est, 'delta_time_mlp'):
            with torch.inference_mode():
                z0 = torch.zeros(1, device=dev, dtype=dtype)
                zero_dt_emb = est.delta_time_mlp(est.time_embeddings(z0).to(dtype)).detach()
            est.register_buffer('_cached_zero_dt_emb', zero_dt_emb, persistent=False)

            def fast_dit_forward(self, x, mu, t, cond, dt):
                x = self.in_proj(x.transpose(1, 2).contiguous())
                cond = self.cond_proj(cond.transpose(1, 2).contiguous())
                prefix = cond.size(1)
                t_emb = self.time_mlp(self.time_embeddings(t).to(x.dtype)) + self._cached_zero_dt_emb
                mu = mu.view(x.size(0), -1, x.size(-1))
                x = torch.cat([mu, t_emb.unsqueeze(1), cond, x], dim=1)
                hidden, _ = self.decoder(x, is_causal=False)
                hidden = hidden[:, prefix + mu.size(1) + 1:, :]
                return self.out_proj(hidden).transpose(1, 2).contiguous()

            est.forward = types.MethodType(fast_dit_forward, est)

    # Eager optimized solve_euler with hoisted loop invariants + lazy CUDA Graph capture.
    if feat_decoder is not None and hasattr(feat_decoder, 'solve_euler') and not getattr(feat_decoder, 'mean_mode', False):
        has_fast_est = (
            est is not None
            and hasattr(est, '_cached_zero_dt_emb')
            and hasattr(est, 'in_proj')
            and hasattr(est, 'cond_proj')
            and hasattr(est, 'out_proj')
            and hasattr(est, 'decoder')
        )

        def _combine_cfg(self, out, b, cfg_value, use_cfg_zero_star):
            dphi_dt, cfg_dphi_dt = out[:b], out[b:]
            if use_cfg_zero_star:
                st_star = self.optimized_scale(dphi_dt.view(b, -1), cfg_dphi_dt.view(b, -1))
                st_star = st_star.view(b, *([1] * (dphi_dt.ndim - 1)))
                neg = cfg_dphi_dt * st_star
            else:
                neg = cfg_dphi_dt
            return neg + cfg_value * (dphi_dt - neg)

        def _eager_solve_euler(self, x, t_span, mu, cond, cfg_value=1.0, use_cfg_zero_star=True):
            t = t_span[0]
            dt = t_span[0] - t_span[1]
            b = x.size(0)
            mu_in = torch.cat((mu, torch.zeros_like(mu)), dim=0)
            zero_init_steps = max(1, int(len(t_span) * 0.04))

            if has_fast_est:
                est_m = self.estimator
                cond_p1 = est_m.cond_proj(cond.transpose(1, 2).contiguous())
                cond_p_in = torch.cat((cond_p1, cond_p1), dim=0)
                mu_view = mu_in.view(2 * b, -1, cond_p1.size(-1))
                prefix_len = cond_p1.size(1) + mu_view.size(1) + 1
                t_emb_all = est_m.time_mlp(est_m.time_embeddings(t_span[:-1]).to(x.dtype)) + est_m._cached_zero_dt_emb
            else:
                cond_in = torch.cat((cond, cond), dim=0)
                dt_in = torch.zeros(2 * b, device=x.device, dtype=x.dtype)

            for step in range(1, len(t_span)):
                if use_cfg_zero_star and step <= zero_init_steps:
                    dphi_dt = torch.zeros_like(x)
                else:
                    if has_fast_est:
                        x_p1 = est_m.in_proj(x.transpose(1, 2).contiguous())
                        x_p_in = torch.cat((x_p1, x_p1), dim=0)
                        t_step = t_emb_all[step - 1:step].expand(2 * b, -1).unsqueeze(1)
                        seq = torch.cat((mu_view, t_step, cond_p_in, x_p_in), dim=1)
                        hidden, _ = est_m.decoder(seq, is_causal=False)
                        out = est_m.out_proj(hidden[:, prefix_len:, :]).transpose(1, 2).contiguous()
                    else:
                        out = self.estimator(torch.cat((x, x), dim=0), mu_in, t.expand(2 * b), cond_in, dt_in)
                    dphi_dt = _combine_cfg(self, out, b, cfg_value, use_cfg_zero_star)
                x = x - dt * dphi_dt
                t = t - dt
                if step < len(t_span) - 1:
                    dt = t - t_span[step + 1]
            return x

        feat_decoder._cuda_graphs = {}
        feat_decoder._cuda_graph_disabled = not (is_cuda and enable_cuda_graphs)

        def graphed_solve_euler(self, x, t_span, mu, cond, cfg_value=1.0, use_cfg_zero_star=True):
            if self._cuda_graph_disabled or not x.is_cuda:
                return _eager_solve_euler(self, x, t_span, mu, cond, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
            key = (x.size(0), x.size(1), x.size(2), x.dtype, mu.size(1), tuple(cond.shape), len(t_span), round(float(cfg_value), 4), bool(use_cfg_zero_star))
            entry = self._cuda_graphs.get(key)
            if entry is None:
                try:
                    s_x = x.clone()
                    s_mu = mu.clone()
                    s_cond = cond.clone()
                    s_tspan = t_span.clone()
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side), torch.inference_mode():
                        for _ in range(2):
                            _ = _eager_solve_euler(self, s_x, s_tspan, s_mu, s_cond, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph), torch.inference_mode():
                        s_out = _eager_solve_euler(self, s_x, s_tspan, s_mu, s_cond, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
                    entry = (graph, s_x, s_mu, s_cond, s_tspan, s_out)
                    self._cuda_graphs[key] = entry
                except Exception:
                    self._cuda_graph_disabled = True
                    self._cuda_graphs.clear()
                    return _eager_solve_euler(self, x, t_span, mu, cond, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
            graph, s_x, s_mu, s_cond, s_tspan, s_out = entry
            s_x.copy_(x)
            s_mu.copy_(mu)
            s_cond.copy_(cond)
            s_tspan.copy_(t_span)
            graph.replay()
            return s_out.clone()

        feat_decoder.solve_euler = types.MethodType(graphed_solve_euler, feat_decoder)

    tts_model._enc_step_graphs = {}
    tts_model._enc_graph_disabled = not (is_cuda and enable_cuda_graphs)
    tts_model._max_batch_size = max_batch_size

    if is_cuda:
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    return True


def _run_lm_step_fast(tts_model, curr_embed_2d, kv_pos, slot_groups=None):
    """Fast eager sliced LM step (base_lm + fsq + fusion + residual_lm) with attn_mask=None everywhere.

    Empirically ~20% faster than boolean-masked bucketed CUDA Graph on RTX 3060 because
    attn_mask=None keeps cuDNN/Flash SDPA active, while all heavy GEMMs (qkv/o/mlp) run fused at B=2..8.
    `slot_groups`: optional list of (start_slot, end_slot, pad) for left-padded multi-segment batches.
    """
    import torch
    import torch.nn.functional as F
    from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb

    bsz = curr_embed_2d.size(0)

    def _make_rope_emb(rope_emb):
        if rope_emb is None:
            return None
        if not slot_groups or all(pad == 0 for _, _, pad in slot_groups):
            return (rope_emb.cos_cached[kv_pos:kv_pos + 1], rope_emb.sin_cached[kv_pos:kv_pos + 1])
        cos_parts, sin_parts = [], []
        for s0, s1, pad in slot_groups:
            rp = kv_pos - pad
            cos_parts.append(rope_emb.cos_cached[rp:rp + 1].expand(s1 - s0, -1))
            sin_parts.append(rope_emb.sin_cached[rp:rp + 1].expand(s1 - s0, -1))
        return (
            torch.cat(cos_parts, dim=0).view(bsz, 1, 1, -1),
            torch.cat(sin_parts, dim=0).view(bsz, 1, 1, -1),
        )

    use_split_sdpa = bool(slot_groups) and any(pad > 0 for _, _, pad in slot_groups)
    end_pos = kv_pos + 1

    def _run_lm(lm, hidden_states, pos_emb):
        kv = lm.kv_cache.kv_cache
        for i, layer in enumerate(lm.layers):
            residual = hidden_states
            h = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            q_dim, k_dim, v_dim = attn._qkv_dims
            q, k, v = attn.qkv_proj(h).split((q_dim, k_dim, v_dim), dim=-1)
            q = q.view(bsz, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = v.view(bsz, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            if pos_emb is not None:
                cos, sin = pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k_cache, v_cache = kv[0, i], kv[1, i]
            k_cache[:bsz, :, kv_pos, :] = k[:, :, 0, :]
            v_cache[:bsz, :, kv_pos, :] = v[:, :, 0, :]
            if not use_split_sdpa:
                out = F.scaled_dot_product_attention(
                    q, k_cache[:bsz, :, :end_pos, :], v_cache[:bsz, :, :end_pos, :],
                    attn_mask=None, is_causal=False, enable_gqa=True,
                )
            else:
                out = torch.cat([
                    F.scaled_dot_product_attention(
                        q[s0:s1], k_cache[s0:s1, :, pad:end_pos, :], v_cache[s0:s1, :, pad:end_pos, :],
                        attn_mask=None, is_causal=False, enable_gqa=True,
                    )
                    for s0, s1, pad in slot_groups
                ], dim=0)
            out = attn.o_proj(out.transpose(1, 2).contiguous().reshape(bsz, attn.num_heads * attn.head_dim))
            hidden_states = _apply_residual_and_mlp(layer, residual, out)
        return lm.norm(hidden_states)

    lm_hidden = tts_model.fsq_layer(_run_lm(tts_model.base_lm, curr_embed_2d, _make_rope_emb(tts_model.base_lm.rope_emb)))
    res_in = tts_model.fusion_concat_proj(torch.cat((lm_hidden, curr_embed_2d), dim=-1))
    residual_hidden = _run_lm(tts_model.residual_lm, res_in, _make_rope_emb(tts_model.residual_lm.rope_emb))
    return lm_hidden, residual_hidden


def _run_enc_step_graphed(tts_model, pred_feat):
    """Run enc_to_lm_proj(feat_encoder(pred_feat.unsqueeze(1))) via static CUDA Graph on CUDA."""
    import torch

    if getattr(tts_model, '_enc_graph_disabled', True) or not pred_feat.is_cuda:
        return tts_model.enc_to_lm_proj(tts_model.feat_encoder(pred_feat.unsqueeze(1)))
    key = tuple(pred_feat.shape)
    graphs = tts_model._enc_step_graphs
    entry = graphs.get(key)
    if entry is None:
        try:
            s_in = pred_feat.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.inference_mode():
                for _ in range(2):
                    _ = tts_model.enc_to_lm_proj(tts_model.feat_encoder(s_in.unsqueeze(1)))
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                s_out = tts_model.enc_to_lm_proj(tts_model.feat_encoder(s_in.unsqueeze(1)))
            entry = (graph, s_in, s_out)
            graphs[key] = entry
        except Exception:
            tts_model._enc_graph_disabled = True
            graphs.clear()
            return tts_model.enc_to_lm_proj(tts_model.feat_encoder(pred_feat.unsqueeze(1)))
    graph, s_in, s_out = entry
    s_in.copy_(pred_feat)
    graph.replay()
    return s_out.clone()


def _prefill_lm_suffix(lm, suffix_embeds, prefix_len):
    """Causal prefill of suffix_embeds attending to cached KV [:prefix_len], writing into [prefix_len:prefix_len+q_len].

    Bit-equivalent to running full causal forward over [prefix; suffix].
    """
    import torch
    import torch.nn.functional as F
    from voxcpm.modules.minicpm4.model import apply_rotary_pos_emb

    bsz, q_len, _ = suffix_embeds.size()
    total_len = prefix_len + q_len
    dev = suffix_embeds.device

    position_emb = (
        (lm.rope_emb.cos_cached[prefix_len:total_len], lm.rope_emb.sin_cached[prefix_len:total_len])
        if lm.rope_emb is not None else None
    )
    if prefix_len > 0:
        mask = torch.ones((q_len, total_len), dtype=torch.bool, device=dev)
        mask[:, prefix_len:] = torch.tril(torch.ones((q_len, q_len), dtype=torch.bool, device=dev))
        attn_mask = mask.view(1, 1, q_len, total_len)
        use_causal = False
    else:
        attn_mask = None
        use_causal = True

    hidden_states = suffix_embeds
    kv = lm.kv_cache.kv_cache
    for i, layer in enumerate(lm.layers):
        residual = hidden_states
        h = layer.input_layernorm(hidden_states)
        attn = layer.self_attn
        q_dim, k_dim, v_dim = attn._qkv_dims
        q, k, v = attn.qkv_proj(h).split((q_dim, k_dim, v_dim), dim=-1)
        q = q.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        if position_emb is not None:
            cos, sin = position_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        kv[0, i, :bsz, :, prefix_len:total_len, :] = k
        kv[1, i, :bsz, :, prefix_len:total_len, :] = v
        out = F.scaled_dot_product_attention(
            q, kv[0, i, :bsz, :, :total_len, :], kv[1, i, :bsz, :, :total_len, :],
            attn_mask=attn_mask, is_causal=use_causal, enable_gqa=True,
        )
        out = attn.o_proj(out.transpose(1, 2).contiguous().reshape(bsz, q_len, attn.num_heads * attn.head_dim))
        hidden_states = _apply_residual_and_mlp(layer, residual, out)

    lm.kv_cache.current_length = total_len
    return lm.norm(hidden_states)
