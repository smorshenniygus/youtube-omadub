#!/usr/bin/env python3
"""Download required local weights for VoxDub (VoxCPM2 + optional CPU reranker models)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
RERANK_DIR = MODELS_DIR / 'rerank'

GIGAAM_URL = (
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
    'sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16.tar.bz2'
)
WESPEAKER_URL = (
    'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/'
    'wespeaker_en_voxceleb_resnet34.onnx'
)


def download_voxcpm2(target_dir: Path, revision: str | None = None) -> None:
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {'local_dir': str(target_dir), 'max_workers': 2}
    if revision:
        kwargs['revision'] = revision
    print(f'Downloading openbmb/VoxCPM2 -> {target_dir} ...', flush=True)
    snapshot_download('openbmb/VoxCPM2', **kwargs)
    print('VoxCPM2 download complete.', flush=True)


def download_reranker(rerank_dir: Path) -> None:
    rerank_dir.mkdir(parents=True, exist_ok=True)
    giga_dir = rerank_dir / 'sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16'
    if not (giga_dir / 'model.int8.onnx').is_file():
        archive = rerank_dir / 'gigaam.tar.bz2'
        print(f'Downloading GigaAM v3 Russian CTC -> {giga_dir} ...', flush=True)
        urllib.request.urlretrieve(GIGAAM_URL, archive)
        with tarfile.open(archive, 'r:bz2') as tar:
            tar.extractall(rerank_dir)
        archive.unlink(missing_ok=True)
    else:
        print('GigaAM v3 already present.', flush=True)

    wespeaker = rerank_dir / 'wespeaker_en_voxceleb_resnet34.onnx'
    if not wespeaker.is_file():
        print(f'Downloading WeSpeaker ResNet34 -> {wespeaker} ...', flush=True)
        urllib.request.urlretrieve(WESPEAKER_URL, wespeaker)
    else:
        print('WeSpeaker ResNet34 already present.', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-rerank', action='store_true', help='Skip optional CPU reranker models')
    parser.add_argument('--revision', default=None, help='Optional HuggingFace commit SHA for openbmb/VoxCPM2')
    args = parser.parse_args()

    download_voxcpm2(MODELS_DIR / 'VoxCPM2', revision=args.revision)
    if not args.skip_rerank:
        download_reranker(RERANK_DIR)
    print('All requested models are ready.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
