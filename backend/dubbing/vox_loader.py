"""Fast meta-device checkpoint loader for VoxCPM2 (direct safetensors -> CUDA, ~2.9s cold start)."""
from pathlib import Path
import types


def _load_voxcpm2_fast(model_dir, device='cuda', max_length=2048, dtype_override=None):
    """Load VoxCPM2 via meta-device init + direct safetensors->CUDA (2-3s vs 10-19s, lower peak RAM/VRAM)."""
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaTokenizerFast
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.model.utils import get_dtype
    from voxcpm.modules.audiovae import AudioVAEV2
    from voxcpm.modules.minicpm4.model import MiniCPMLongRoPE

    path = Path(model_dir)
    safetensors_path = path / 'model.safetensors'
    if not safetensors_path.is_file():
        raise FileNotFoundError(str(safetensors_path))

    config = VoxCPMConfig.model_validate_json((path / 'config.json').read_text(encoding='utf-8'))
    config.max_length = max_length
    if dtype_override:
        config.dtype = dtype_override
    tokenizer = LlamaTokenizerFast.from_pretrained(str(path))
    audio_vae_config = getattr(config, 'audio_vae_config', None)
    # AudioVAE is small (~360MB) and uses weight_norm; construct on CPU cleanly, then load weights on GPU.
    audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()

    with torch.device('meta'):
        tts_model = VoxCPM2Model(config, tokenizer, audio_vae, device=device)

    lm_dtype = get_dtype(tts_model.config.dtype)
    cuda_dev = torch.device(device)

    # Load transformer weights directly onto GPU (no CPU copy, no random FP32 initialization).
    model_sd = load_file(str(safetensors_path), device=device)
    for k, v in list(model_sd.items()):
        if v.is_floating_point() and v.dtype != lm_dtype:
            model_sd[k] = v.to(lm_dtype)

    # Load AudioVAE weights directly onto GPU in FP32.
    vae_st = path / 'audiovae.safetensors'
    vae_pth = path / 'audiovae.pth'
    if vae_st.is_file():
        vae_sd = load_file(str(vae_st), device=device)
    elif vae_pth.is_file():
        ckpt = torch.load(str(vae_pth), map_location=device, weights_only=True)
        vae_sd = ckpt.get('state_dict', ckpt)
    else:
        raise FileNotFoundError('AudioVAE checkpoint not found')
    for k, v in vae_sd.items():
        if v.is_floating_point() and v.dtype != torch.float32:
            v = v.to(torch.float32)
        model_sd[f'audio_vae.{k}'] = v

    tts_model.load_state_dict(model_sd, strict=False, assign=True)
    del model_sd, vae_sd

    # Re-materialize AudioVAE weight_norm computed `.weight` attributes on CUDA cleanly.
    tts_model.audio_vae.to(cuda_dev)

    # Re-materialize non-persistent RoPE buffers (the only non-persistent buffers in VoxCPM2Model).
    for module in tts_model.modules():
        if isinstance(module, MiniCPMLongRoPE):
            inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2, device=cuda_dev).float() / module.dim))
            module.register_buffer('inv_freq', inv_freq, persistent=False)
            module._set_cos_sin_cache(
                seq_len=module.max_position_embeddings,
                device=cuda_dev,
                dtype=torch.float32,
            )

    # Sanity check: ensure no parameter or buffer remained on meta device.
    for name, p in tts_model.named_parameters():
        if p.is_meta:
            raise RuntimeError(f'Unmaterialized meta parameter: {name}')
    for name, b in tts_model.named_buffers():
        if b.is_meta:
            raise RuntimeError(f'Unmaterialized meta buffer: {name}')

    tts_model.eval()
    return types.SimpleNamespace(tts_model=tts_model)
