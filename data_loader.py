#!/usr/bin/env python3
"""Download the active_matter dataset from Hugging Face into a local data folder.

Designed for HPC use cases where you want the raw HDF5 files laid out as:

    data/
      train/
      valid/
      test/

Examples
--------
Download everything into ./data:
    python data_loader.py

Download only train split into /scratch/$USER/project/data:
    python data_loader.py --root /scratch/$USER/project/data --split train

Download train + valid using 8 workers:
    python data_loader.py --root /scratch/$USER/data --split train valid --workers 8

Notes
-----
- The Hugging Face repo currently stores the dataset as raw HDF5 files under
  data/train, data/valid, and data/test.
- The full repository is about 55.8 GB, so using /scratch on HPC is recommended.
- You may need to run `huggingface-cli login` first if your environment requires it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is not installed. Install it with: pip install -U huggingface_hub"
    ) from exc


REPO_ID = "polymathic-ai/active_matter"
VALID_SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the polymathic-ai/active_matter dataset into a local data folder."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Local directory where the dataset should be stored. Default: ./data",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
        help="One or more splits to download. Default: train valid test",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {REPO_ID}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum number of concurrent download workers. Default: 4",
    )
    parser.add_argument(
        "--local-dir-use-symlinks",
        action="store_true",
        help="Allow symlinks in the local dir when supported by the Hub client.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of files even if they already exist locally.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. If omitted, your cached login is used.",
    )
    return parser.parse_args()


def ensure_disk_space(path: Path, min_free_gb: int = 65) -> None:
    """Warn if the target filesystem looks too small for the full dataset."""
    parent = path.resolve().parent
    usage = shutil.disk_usage(parent)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        print(
            f"[warning] Only {free_gb:.1f} GB free in {parent}. "
            f"The full dataset is ~55.8 GB, so downloads may fail.",
            file=sys.stderr,
        )


def build_allow_patterns(splits: Iterable[str]) -> List[str]:
    patterns = [f"data/{split}/*" for split in splits]
    patterns.extend(["README.md", "active_matter.yaml", "stats.yaml"])
    return patterns


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    ensure_disk_space(root)

    allow_patterns = build_allow_patterns(args.split)

    print(f"[info] Downloading repo: {args.repo_id}")
    print(f"[info] Target directory: {root}")
    print(f"[info] Splits: {', '.join(args.split)}")
    print(f"[info] Allow patterns: {allow_patterns}")

    snapshot_kwargs = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "local_dir": str(root),
        "allow_patterns": allow_patterns,
        "max_workers": args.workers,
        "token": args.hf_token,
        "force_download": args.force,
    }
    if args.local_dir_use_symlinks:
        snapshot_kwargs["local_dir_use_symlinks"] = True

    try:
        snapshot_download(**snapshot_kwargs)
    except TypeError:
        snapshot_kwargs.pop("local_dir_use_symlinks", None)
        snapshot_download(**snapshot_kwargs)

    print("[done] Download complete.")
    print("[done] Expected structure:")
    for split in args.split:
        split_path = root / "data" / split
        n_files = len(list(split_path.glob("*.hdf5"))) if split_path.exists() else 0
        print(f"  - {split_path} ({n_files} .hdf5 files)")


if __name__ == "__main__":
    main()
