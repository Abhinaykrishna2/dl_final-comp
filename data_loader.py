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

Download train + valid using 12 workers:
    python data_loader.py --root /scratch/$USER/data --split train valid --workers 12

Preview the exact transfer before downloading:
    python data_loader.py --dry-run

Notes
-----
- The Hugging Face repo currently stores the dataset as raw HDF5 files under
  data/train, data/valid, and data/test.
- The full repository is about 55.8 GB, so using /scratch on HPC is recommended.
- This script uses the real Hugging Face cache and then materializes files into
  the target folder, which makes reruns resumable and avoids weak local-dir-only
  caching behavior.
- If `hf_xet` is installed, high-performance Xet mode is enabled by default.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import errno
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ID = "polymathic-ai/active_matter"
VALID_SPLITS = ("train", "valid", "test")
METADATA_FILES = ("README.md", "active_matter.yaml", "stats.yaml")
STATE_FILENAME = ".download_state.json"


@dataclass(frozen=True)
class PlannedFile:
    repo_filename: str
    dest_rel: Path
    file_size: int
    commit_hash: str
    is_cached: bool
    will_download: bool


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 8
    return max(8, min(16, cpu_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the polymathic-ai/active_matter dataset into a local data folder."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Final dataset directory. Default: ./data",
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
        "--revision",
        default=None,
        help="Optional branch, tag, or commit to pin the dataset revision.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Maximum number of concurrent download workers. Default: auto (8-16)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for Hugging Face blobs. Default: <root-parent>/.hf-cache-active_matter",
    )
    parser.add_argument(
        "--xet-cache-dir",
        type=Path,
        default=None,
        help="Cache directory for hf_xet chunk data. Default: <cache-dir>/xet",
    )
    parser.add_argument(
        "--disable-xet-high-performance",
        action="store_true",
        help="Disable HF_XET_HIGH_PERFORMANCE even when hf_xet is installed.",
    )
    parser.add_argument(
        "--hf-progress-bars",
        action="store_true",
        help="Keep Hugging Face progress bars enabled instead of using compact script logging.",
    )
    parser.add_argument(
        "--etag-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds when resolving remote ETags. Default: 30",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not hit the network. Use only already-cached files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of files even if they already exist in cache.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. If omitted, your cached login is used.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.etag_timeout <= 0:
        parser.error("--etag-timeout must be > 0")
    return args


def resolve_cache_dir(root: Path, cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir.expanduser().resolve()
    return (root.parent / ".hf-cache-active_matter").resolve()


def resolve_xet_cache_dir(cache_dir: Path, xet_cache_dir: Path | None) -> Path:
    if xet_cache_dir is not None:
        return xet_cache_dir.expanduser().resolve()
    return (cache_dir / "xet").resolve()


def configure_hf_environment(args: argparse.Namespace, xet_cache_dir: Path) -> None:
    if args.xet_cache_dir is not None:
        os.environ["HF_XET_CACHE"] = str(xet_cache_dir)
    else:
        os.environ.setdefault("HF_XET_CACHE", str(xet_cache_dir))

    if args.disable_xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
    else:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    if args.hf_progress_bars:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    else:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def import_hf_clients() -> tuple[Any, Any]:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is not installed. Install it with: pip install -U huggingface_hub hf_xet"
        ) from exc

    return hf_hub_download, snapshot_download


def build_allow_patterns(splits: Iterable[str]) -> list[str]:
    patterns: list[str] = []
    for split in splits:
        patterns.extend([f"data/{split}/*", f"data/{split}/**"])
    patterns.extend(METADATA_FILES)
    return patterns


def repo_path_to_dest(repo_filename: str) -> Path:
    parts = Path(repo_filename).parts
    if parts[:1] == ("data",) and len(parts) > 1:
        return Path(*parts[1:])
    return Path(*parts)


def format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}

    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"[warning] Ignoring unreadable state file at {state_path}: {exc}",
            file=sys.stderr,
        )
        return {}


def save_state(
    state_path: Path,
    args: argparse.Namespace,
    resolved_commit: str,
    plan: list[PlannedFile],
) -> None:
    payload = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_commit": resolved_commit,
        "splits": list(args.split),
        "files": {
            item.dest_rel.as_posix(): {
                "repo_filename": item.repo_filename,
                "size": item.file_size,
            }
            for item in plan
        },
    }
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def migrate_legacy_layout(root: Path, splits: Iterable[str]) -> None:
    legacy_root = root / "data"
    if not legacy_root.is_dir():
        return

    moved_any = False
    for split in splits:
        old_path = legacy_root / split
        new_path = root / split
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            moved_any = True

    if moved_any:
        print("[info] Migrated legacy nested data/ layout into the target root.")

    try:
        legacy_root.rmdir()
    except OSError:
        pass


def build_download_plan(
    snapshot_download: Any,
    args: argparse.Namespace,
    cache_dir: Path,
) -> list[PlannedFile]:
    allow_patterns = build_allow_patterns(args.split)
    try:
        dry_run_infos = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            max_workers=args.workers,
            token=args.hf_token,
            force_download=args.force,
            etag_timeout=args.etag_timeout,
            local_files_only=args.local_files_only,
            dry_run=True,
        )
    except TypeError as exc:
        raise SystemExit(
            "This script requires a recent huggingface_hub with dry-run support. "
            "Upgrade it with: pip install -U huggingface_hub hf_xet"
        ) from exc

    plan = [
        PlannedFile(
            repo_filename=info.filename,
            dest_rel=repo_path_to_dest(info.filename),
            file_size=info.file_size,
            commit_hash=info.commit_hash,
            is_cached=info.is_cached,
            will_download=info.will_download,
        )
        for info in sorted(dry_run_infos, key=lambda item: item.filename)
    ]

    if not plan:
        raise SystemExit("No files matched the requested splits.")

    return plan


def resolved_commit_for(plan: list[PlannedFile]) -> str:
    commits = {item.commit_hash for item in plan}
    if len(commits) != 1:
        raise SystemExit(
            f"Expected a single resolved commit for the download plan, got: {sorted(commits)}"
        )
    return next(iter(commits))


def is_current_file(
    item: PlannedFile,
    root: Path,
    state: dict[str, Any],
    resolved_commit: str,
    force: bool,
) -> bool:
    if force:
        return False

    dest_path = root / item.dest_rel
    if not dest_path.is_file():
        return False

    try:
        dest_size = dest_path.stat().st_size
    except OSError:
        return False

    if dest_size != item.file_size:
        return False

    state_commit = state.get("resolved_commit")
    state_files = state.get("files", {})
    entry = state_files.get(item.dest_rel.as_posix())

    if state_commit == resolved_commit and isinstance(entry, dict):
        return (
            entry.get("repo_filename") == item.repo_filename
            and entry.get("size") == item.file_size
        )

    return state_commit is None


def report_disk_space(
    root: Path,
    cache_dir: Path,
    bytes_to_download: int,
    bytes_to_materialize: int,
) -> None:
    root_usage = shutil.disk_usage(root)
    cache_usage = shutil.disk_usage(cache_dir)
    same_filesystem = root.stat().st_dev == cache_dir.stat().st_dev

    if same_filesystem:
        required = bytes_to_download + bytes_to_materialize
        free = root_usage.free
        if free < required:
            print(
                f"[warning] Only {format_bytes(free)} free on {root}. "
                f"Estimated additional space needed is about {format_bytes(required)}.",
                file=sys.stderr,
            )
        return

    if cache_usage.free < bytes_to_download:
        print(
            f"[warning] Only {format_bytes(cache_usage.free)} free on cache filesystem {cache_dir}. "
            f"Expected download footprint is about {format_bytes(bytes_to_download)}.",
            file=sys.stderr,
        )
    if root_usage.free < bytes_to_materialize:
        print(
            f"[warning] Only {format_bytes(root_usage.free)} free on target filesystem {root}. "
            f"Expected materialized footprint is about {format_bytes(bytes_to_materialize)}.",
            file=sys.stderr,
        )


def same_source_file(source_path: Path, dest_path: Path) -> bool:
    try:
        source_stat = source_path.stat()
        dest_stat = dest_path.stat()
    except OSError:
        return False

    return (
        source_stat.st_dev == dest_stat.st_dev
        and source_stat.st_ino == dest_stat.st_ino
    )


def materialize_cached_file(source_path: Path, dest_path: Path) -> str:
    source_blob = source_path.resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        if same_source_file(source_blob, dest_path):
            return "existing"

    temp_path = dest_path.with_name(f"{dest_path.name}.partial")
    if temp_path.exists():
        temp_path.unlink()

    mode = "copy"
    try:
        os.link(source_blob, temp_path)
        mode = "hardlink"
    except OSError as exc:
        if exc.errno not in {
            errno.EACCES,
            errno.EEXIST,
            errno.EPERM,
            errno.EXDEV,
            errno.EMLINK,
            errno.ENOTSUP,
        }:
            raise
        shutil.copy2(source_blob, temp_path)

    os.replace(temp_path, dest_path)
    return mode


def should_report_progress(completed: int, total: int) -> bool:
    if total <= 10:
        return True
    step = max(1, total // 10)
    return completed == total or completed % step == 0


def count_hdf5_files(split_path: Path) -> int:
    if not split_path.exists():
        return 0
    return sum(1 for path in split_path.rglob("*.hdf5") if path.is_file())


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    cache_dir = resolve_cache_dir(root, args.cache_dir)
    xet_cache_dir = resolve_xet_cache_dir(cache_dir, args.xet_cache_dir)

    root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    xet_cache_dir.mkdir(parents=True, exist_ok=True)

    configure_hf_environment(args, xet_cache_dir)
    hf_hub_download, snapshot_download = import_hf_clients()

    migrate_legacy_layout(root, args.split)

    plan = build_download_plan(snapshot_download, args, cache_dir)
    resolved_commit = resolved_commit_for(plan)
    state_path = root / STATE_FILENAME
    state = load_state(state_path)
    files_to_process = [
        item
        for item in plan
        if not is_current_file(item, root, state, resolved_commit, args.force)
    ]

    total_bytes = sum(item.file_size for item in plan)
    cached_bytes = sum(item.file_size for item in plan if item.is_cached and not args.force)
    download_bytes = sum(
        item.file_size for item in files_to_process if item.will_download or args.force
    )
    same_filesystem = root.stat().st_dev == cache_dir.stat().st_dev
    materialize_bytes = 0 if same_filesystem else sum(item.file_size for item in files_to_process)

    print(f"[info] Downloading repo: {args.repo_id}")
    print(f"[info] Resolved commit: {resolved_commit}")
    print(f"[info] Target directory: {root}")
    print(f"[info] Cache directory: {cache_dir}")
    print(f"[info] Xet cache directory: {xet_cache_dir}")
    print(f"[info] Splits: {', '.join(args.split)}")
    print(
        f"[info] Files matched: {len(plan)} "
        f"({format_bytes(total_bytes)} total, {format_bytes(cached_bytes)} already cached)"
    )

    if importlib.util.find_spec("hf_xet") is not None:
        xet_mode = os.environ.get("HF_XET_HIGH_PERFORMANCE", "0")
        print(f"[info] hf_xet detected. HF_XET_HIGH_PERFORMANCE={xet_mode}")

    if args.dry_run:
        print(
            f"[done] Dry run only. {len(files_to_process)} files still need work "
            f"({format_bytes(download_bytes)} to fetch, {format_bytes(materialize_bytes)} to materialize)."
        )
        return

    if not files_to_process:
        save_state(state_path, args, resolved_commit, plan)
        print("[done] Dataset is already present for the requested revision.")
        return

    report_disk_space(root, cache_dir, download_bytes, materialize_bytes)

    total_process_bytes = sum(item.file_size for item in files_to_process)
    completed_files = 0
    completed_bytes = 0
    mode_counts: dict[str, int] = {}

    def worker(item: PlannedFile) -> tuple[str, int]:
        cached_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=item.repo_filename,
            repo_type="dataset",
            revision=resolved_commit,
            cache_dir=str(cache_dir),
            force_download=args.force,
            etag_timeout=args.etag_timeout,
            token=args.hf_token,
            local_files_only=args.local_files_only,
        )
        mode = materialize_cached_file(Path(cached_path), root / item.dest_rel)
        return mode, item.file_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {
            executor.submit(worker, item): item
            for item in files_to_process
        }
        for future in concurrent.futures.as_completed(future_to_item):
            mode, file_size = future.result()
            completed_files += 1
            completed_bytes += file_size
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

            if should_report_progress(completed_files, len(files_to_process)):
                print(
                    f"[progress] {completed_files}/{len(files_to_process)} files ready "
                    f"({format_bytes(completed_bytes)} / {format_bytes(total_process_bytes)})"
                )

    save_state(state_path, args, resolved_commit, plan)

    print("[done] Download complete.")
    print(
        "[done] Materialization summary: "
        + ", ".join(f"{count} {mode}" for mode, count in sorted(mode_counts.items()))
    )
    print("[done] Expected structure:")
    for split in args.split:
        split_path = root / split
        print(f"  - {split_path} ({count_hdf5_files(split_path)} .hdf5 files)")


if __name__ == "__main__":
    main()
