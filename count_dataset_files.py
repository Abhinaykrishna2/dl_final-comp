#!/usr/bin/env python3
"""Print sample counts for dataset splits under data/.

Each split contains HDF5 files where the first axis of every field
dataset is the number of simulation trajectories in that file.
This script counts:
  - files   : number of .hdf5 files on disk
  - sims    : total simulation trajectories (first axis across all files)
  - single_clips : one deterministic 16-frame clip per simulation
  - windows      : total 32-frame sliding windows extractable from those sims
                   (each simulation has 81 time steps, so 81 - 32 + 1 = 50 windows)
                   The 32-frame window follows the JEPA training setup:
                   16 context frames + 16 target frames, stride = 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py


# Field used to read the number of simulations per file.
_PROBE_FIELD = "t0_fields/concentration"
# Single evaluation clip = 16 encoder frames.
_CLIP = 16
# Full JEPA window = context (16) + target (16).
_WINDOW = 32


def count_split(path: Path) -> tuple[int, int, int, int]:
    """Return (n_files, n_sims, n_single_clips, n_windows) for a split directory."""
    if not path.exists():
        return 0, 0, 0, 0

    hdf5_files = sorted(path.glob("*.hdf5"))
    n_files = len(hdf5_files)
    n_sims = 0
    n_single_clips = 0
    n_windows = 0

    for f in hdf5_files:
        with h5py.File(f, "r") as h:
            shape = h[_PROBE_FIELD].shape  # (sims, steps, H, W)
            sims, steps = shape[0], shape[1]
        n_sims += sims
        n_single_clips += sims * int(steps >= _CLIP)
        n_windows += sims * max(0, steps - _WINDOW + 1)

    return n_files, n_sims, n_single_clips, n_windows


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

    valid_dir = root / "valid"
    if not valid_dir.exists():
        valid_dir = root / "val"

    splits = [
        ("train", root / "train"),
        ("test",  root / "test"),
        ("val",   valid_dir),
    ]

    print(f"root: {root.resolve()}")
    print(f"{'split':<6}  {'files':>6}  {'sims':>6}  {'single clips':>14}  {'sliding windows':>18}")
    print("-" * 62)
    for name, path in splits:
        n_files, n_sims, n_single_clips, n_windows = count_split(path)
        print(f"{name:<6}  {n_files:>6}  {n_sims:>6}  {n_single_clips:>14}  {n_windows:>18}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
