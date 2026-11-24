"""Bounded VAE decoding and resource-pressure helpers; no foreign processes killed."""
from contextlib import contextmanager
import gc
import os
import subprocess
import sys
import threading

# Must precede CUDA initialization (jobs imports engine before source preparation).
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
DECODE_FRAMES = 4
DECODE_FRAMES_FAST = 32
DECODE_FRAMES_MEDIUM = 16
MIN_GPU_MIB = 5500
GPU_RETRY_SECONDS = 15
MAX_OOM_RETRIES = 3

_NVML = None
_NVML_FAILED = False
_NVML_LOCK = threading.Lock()


def _nvml_memory_info():
    """Query (free_mib, total_mib) via libnvidia-ml.so.1 in ~0.05ms (no subprocess)."""
    global _NVML, _NVML_FAILED
    if _NVML_FAILED:
        return None
    import ctypes

    class _NvmlMemory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

    try:
        if _NVML is None:
            with _NVML_LOCK:
                if _NVML_FAILED:
                    return None
                if _NVML is None:
                    lib = ctypes.CDLL('libnvidia-ml.so.1')
                    lib.nvmlInit_v2.argtypes = []
                    lib.nvmlInit_v2.restype = ctypes.c_int
                    lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
                    lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
                    lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
                    if lib.nvmlInit_v2() != 0:
                        _NVML_FAILED = True
                        return None
                    handle = ctypes.c_void_p()
                    if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                        _NVML_FAILED = True
                        return None
                    _NVML = (lib, handle)
        lib, handle = _NVML
        mem = _NvmlMemory()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            return None
        return (mem.free // 2**20, mem.total // 2**20)
    except (OSError, AttributeError):
        _NVML_FAILED = True
        return None


def choose_decode_frames(default_frames=DECODE_FRAMES):
    """Pick larger causal VAE chunk size when ample free VRAM is available (bit-exact output)."""
    torch = sys.modules.get('torch')
    free_mib = None
    if torch is not None and torch.cuda.is_initialized():
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_mib = free_bytes // 2**20
        except Exception:
            free_mib = None
    if free_mib is None:
        info = _nvml_memory_info()
        if info is not None:
            free_mib = info[0]
    if free_mib is None:
        return default_frames
    if free_mib >= 2400:
        return DECODE_FRAMES_FAST
    if free_mib >= 1200:
        return DECODE_FRAMES_MEDIUM
    return default_frames


@contextmanager
def bounded_vae_decode(vae, frames=DECODE_FRAMES):
    """Keep complete TTS generation/retry semantics, stream only causal VAE decode.

    Upstream streaming_decode patches causal convolution state. Call the saved
    decode method directly: decode_chunk would recurse into our temporary wrapper.
    Wave chunks go straight to CPU, so activation memory does not grow with speech
    duration. The upstream consumer already crops context and transfers audio to CPU.
    """
    if type(frames) is not int or frames < 1:
        raise ValueError('invalid VAE chunk bound')
    original = vae.decode
    def decode(z, sr_cond=None):
        import torch
        if z.shape[-1] <= frames:
            return original(z, sr_cond)
        chunks = []
        with vae.streaming_decode():
            for offset in range(0, z.shape[-1], frames):
                chunks.append(original(z[..., offset:offset+frames], sr_cond).cpu())
        return torch.cat(chunks, dim=-1)
    vae.decode = decode
    try:
        yield
    finally:
        vae.decode = original


def is_cuda_oom(error):
    return isinstance(error, RuntimeError) and 'cuda out of memory' in str(error).lower()


def clear_cuda_memory():
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def gpu_capacity_mib():
    """Physical capacity distinguishes impossible budgets from temporary pressure."""
    info = _nvml_memory_info()
    if info is not None and info[1] > 0:
        return int(info[1])
    try:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        total = int(result.stdout.strip())
        return total if total > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def gpu_memory_mib():
    """Budget available to our process, including its reusable PyTorch reservation."""
    torch = sys.modules.get('torch')
    reserved = torch.cuda.memory_reserved() // 2**20 if torch is not None and torch.cuda.is_initialized() else 0
    if torch is not None and torch.cuda.is_initialized():
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            return int(free_bytes // 2**20) + reserved
        except Exception:
            pass
    info = _nvml_memory_info()
    if info is not None:
        return int(info[0]) + reserved
    try:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
        free = int(result.stdout.strip())
        return free + reserved
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _arch_family(major, minor):
    if major >= 12:
        return 'blackwell'
    if major >= 10:
        return 'blackwell'
    if major == 9:
        return 'hopper'
    if major == 8 and minor >= 9:
        return 'ada'
    if major == 8:
        return 'ampere'
    if major == 7:
        return 'turing'
    return 'other'


def resolve_gpu_profile(name, compute_capability, total_vram_mib, env=None):
    """Derive safe portable acceleration settings from GPU architecture and VRAM.

    Portable optimizations (QKV/MLP weight fusion, fused RMSNorm, bounded KV cache,
    split-SDPA, AudioVAE weight_norm fold) use 0 extra VRAM and stay on for all CUDA GPUs.
    CUDA Graphs and larger super-batches (B=4/6/8) enable automatically on modern
    architectures (Ampere SM 8.0+, Ada SM 8.9, Hopper SM 9.0, Blackwell SM 10/12)
    with enough VRAM headroom, and fall back safely on older or 8 GiB cards.
    """
    env = os.environ if env is None else env
    if not compute_capability or total_vram_mib is None or total_vram_mib <= 0:
        return {
            'available': False,
            'name': name or 'CPU / No CUDA GPU',
            'compute_capability': None,
            'arch_family': 'cpu',
            'total_vram_mib': 0,
            'dtype': 'float32',
            'cuda_graphs': False,
            'max_batch_size': 1,
            'weight_fusion': False,
        }

    major, minor = int(compute_capability[0]), int(compute_capability[1])
    total_vram_mib = int(total_vram_mib)
    dtype_env = str(env.get('VOXDUB_DTYPE', 'auto')).strip().lower()
    if dtype_env in ('bfloat16', 'bf16'):
        dtype = 'bfloat16'
    elif dtype_env in ('float16', 'fp16'):
        dtype = 'float16'
    else:
        dtype = 'bfloat16' if major >= 8 else 'float16'

    graphs_env = str(env.get('VOXDUB_CUDA_GRAPHS', 'auto')).strip().lower()
    if graphs_env in ('1', 'true', 'yes', 'on'):
        cuda_graphs = True
    elif graphs_env in ('0', 'false', 'no', 'off'):
        cuda_graphs = False
    else:
        cuda_graphs = major >= 8 and total_vram_mib >= 9500

    batch_env = str(env.get('VOXDUB_MAX_BATCH', 'auto')).strip().lower()
    if batch_env.isdigit() and int(batch_env) in (1, 2, 4, 6, 8):
        max_batch_size = int(batch_env)
    else:
        # Measured B=8 + CUDA Graphs peak is ~7.2 GiB (leaving ~4.5 GiB headroom on 12 GiB cards).
        # Engine still caps chosen_batch dynamically by actual post-load free VRAM.
        if total_vram_mib >= 11000:
            max_batch_size = 8
        elif total_vram_mib >= 9500:
            max_batch_size = 6
        else:
            max_batch_size = 2

    return {
        'available': True,
        'name': str(name or 'CUDA GPU'),
        'compute_capability': [major, minor],
        'arch_family': _arch_family(major, minor),
        'total_vram_mib': total_vram_mib,
        'dtype': dtype,
        'cuda_graphs': cuda_graphs,
        'max_batch_size': max_batch_size,
        'weight_fusion': True,
    }


def detect_gpu_profile(env=None):
    """Inspect local CUDA device without forcing heavy PyTorch init when only probing."""
    torch = sys.modules.get('torch')
    if torch is not None and getattr(torch, 'cuda', None) is not None:
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return resolve_gpu_profile(
                    props.name,
                    (props.major, props.minor),
                    props.total_memory // 2**20,
                    env=env,
                )
        except Exception:
            pass
    try:
        result = subprocess.run(
            ['nvidia-smi', '--id=0', '--query-gpu=name,compute_cap,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3, check=True,
        )
        parts = [p.strip() for p in result.stdout.strip().splitlines()[0].split(',')]
        if len(parts) >= 3:
            cc_parts = parts[1].split('.')
            cc = (int(cc_parts[0]), int(cc_parts[1]) if len(cc_parts) > 1 else 0)
            return resolve_gpu_profile(parts[0], cc, int(float(parts[2])), env=env)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return resolve_gpu_profile(None, None, 0, env=env)
