#!/usr/bin/env python3
"""Print file counts for dataset splits under data/."""

from __future__ import annotations

import sys
from pathlib import Path


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

    train_dir = root / "train"
    test_dir = root / "test"
    valid_dir = root / "valid"
    if not valid_dir.exists():
        valid_dir = root / "val"

    print(f"root: {root.resolve()}")
    print(f"train: {count_files(train_dir)}")
    print(f"test: {count_files(test_dir)}")
    print(f"val: {count_files(valid_dir)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
