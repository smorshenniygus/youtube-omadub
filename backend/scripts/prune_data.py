#!/usr/bin/env python3
"""Safe, dry-run-by-default disk retention utility for VoxDub backend data.

Usage:
  .venv/bin/python scripts/prune_data.py                  # Dry-run report only
  .venv/bin/python scripts/prune_data.py --apply          # Apply default retention
  .venv/bin/python scripts/prune_data.py --apply --source-days 7 --job-days 3
  .venv/bin/python scripts/prune_data.py --apply --include-maintenance
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time


ROOT = Path(__file__).resolve().parents[1]


def dir_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def fmt_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024.0 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{num_bytes} B'


def latest_mtime(path: Path) -> float:
    try:
        newest = path.stat().st_mtime
    except OSError:
        return 0.0
    if path.is_dir():
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    newest = max(newest, (Path(root) / name).stat().st_mtime)
                except OSError:
                    pass
    return newest


def collect_candidates(
    root: Path,
    *,
    source_days: float,
    job_days: float,
    include_maintenance: bool,
    now: float | None = None,
) -> list[tuple[str, Path, int]]:
    now = time.time() if now is None else now
    candidates: list[tuple[str, Path, int]] = []

    source_cache = root / 'data' / 'source-cache'
    if source_cache.is_dir():
        source_cutoff = now - source_days * 86400.0
        for vid_dir in sorted(source_cache.iterdir()):
            if not vid_dir.is_dir():
                continue
            if source_days >= 0 and latest_mtime(vid_dir) < source_cutoff:
                candidates.append(('old-source-cache', vid_dir, dir_size(vid_dir)))
                continue
            # Even for fresh cache entries, remove redundant original.* if source.wav exists.
            if (vid_dir / 'source.wav').exists():
                for orig in sorted(vid_dir.glob('original.*')):
                    if orig.is_file():
                        candidates.append(('redundant-original', orig, dir_size(orig)))

    jobs_dir = root / 'data' / 'jobs'
    if jobs_dir.is_dir() and job_days >= 0:
        job_cutoff = now - job_days * 86400.0
        for job_dir in sorted(jobs_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            if latest_mtime(job_dir) < job_cutoff:
                candidates.append(('old-job', job_dir, dir_size(job_dir)))

    if include_maintenance:
        maint_dir = root / 'maintenance'
        if maint_dir.is_dir():
            for entry in sorted(maint_dir.iterdir()):
                candidates.append(('maintenance', entry, dir_size(entry)))

    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description='Inspect or prune cached dubbing data.')
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root directory')
    parser.add_argument('--source-days', type=float, default=14.0, help='Prune source-cache entries older than N days (-1 to keep)')
    parser.add_argument('--job-days', type=float, default=7.0, help='Prune job directories older than N days (-1 to keep)')
    parser.add_argument('--include-maintenance', action='store_true', help='Also include maintenance/ experiment archives')
    parser.add_argument('--apply', action='store_true', help='Actually delete listed paths (default is dry-run)')
    args = parser.parse_args()

    candidates = collect_candidates(
        args.root,
        source_days=args.source_days,
        job_days=args.job_days,
        include_maintenance=args.include_maintenance,
    )
    total = sum(size for _, _, size in candidates)
    mode = 'DELETE' if args.apply else 'DRY-RUN'
    if not candidates:
        print(f'[{mode}] Nothing to prune.')
        return 0

    for kind, path, size in candidates:
        rel = path.relative_to(args.root)
        print(f'[{mode}] {kind:18s} {fmt_size(size):>10s}  {rel}')
        if args.apply:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    action = 'Freed' if args.apply else 'Reclaimable'
    print(f'[{mode}] {action}: {fmt_size(total)} across {len(candidates)} items.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
