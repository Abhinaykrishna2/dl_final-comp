from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import weakref

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import LABEL_NAMES


FIELD_GROUPS = ("t0_fields", "t1_fields", "t2_fields")
VALID_SPLITS = {"train", "valid", "val", "test"}
VALID_INDEX_MODES = {"single_clip", "sliding_window"}
VALID_CLIP_SELECTIONS = {"start", "center", "end"}


@dataclass(frozen=True)
class FieldSpec:
    path: str
    channels: int


@dataclass(frozen=True)
class FileInfo:
    path: Path
    n_sims: int
    n_steps: int
    labels: np.ndarray


def canonical_split(split: str) -> str:
    split = split.lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return "valid" if split == "val" else split


def resolve_split_dir(root: str | Path, split: str) -> Path:
    split = canonical_split(split)
    root = Path(root).expanduser()
    candidates = [root / split, root / "data" / split]
    if split == "valid":
        candidates.extend([root / "val", root / "data" / "val"])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    return candidates[0].resolve()


class ActiveMatterWindowDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        context_frames: int = 16,
        target_frames: int = 16,
        stride: int = 1,
        resolution: Optional[int] = 224,
        max_samples: Optional[int] = None,
        max_open_files: int = 4,
        index_mode: str = "sliding_window",
        clip_selection: str = "center",
        include_labels: bool = True,
    ) -> None:
        if context_frames < 1:
            raise ValueError("context_frames must be >= 1")
        if target_frames < 0:
            raise ValueError("target_frames must be >= 0")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if index_mode not in VALID_INDEX_MODES:
            raise ValueError(f"unsupported index_mode: {index_mode}")
        if clip_selection not in VALID_CLIP_SELECTIONS:
            raise ValueError(f"unsupported clip_selection: {clip_selection}")

        self.split = canonical_split(split)
        self.root = Path(root).expanduser().resolve()
        self.split_dir = resolve_split_dir(self.root, self.split)
        self.context_frames = context_frames
        self.target_frames = target_frames
        self.total_frames = context_frames + target_frames
        self.stride = stride
        self.resolution = resolution
        self.max_samples = max_samples
        self.max_open_files = max_open_files
        self.index_mode = index_mode
        self.clip_selection = clip_selection
        self.include_labels = include_labels
        self._open_files: OrderedDict[str, h5py.File] | None = None

        self.files = self._discover_files()
        self.field_specs = self._discover_fields(self.files[0])
        self.file_infos = self._scan_files()
        self.index = self._build_index()

        if not self.index:
            raise ValueError(
                f"No samples found in {self.split_dir}. "
                "Check the split path, frame counts, and HDF5 file contents."
            )

    def _discover_files(self) -> list[Path]:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"split directory does not exist: {self.split_dir}")

        files = sorted(self.split_dir.glob("*.hdf5")) + sorted(self.split_dir.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"no HDF5 files found in {self.split_dir}")
        return files

    def _discover_fields(self, sample_file: Path) -> list[FieldSpec]:
        field_specs: list[FieldSpec] = []
        with h5py.File(sample_file, "r") as h5:
            for group_name in FIELD_GROUPS:
                if group_name not in h5:
                    continue
                for dataset_name in sorted(h5[group_name].keys()):
                    ds = h5[f"{group_name}/{dataset_name}"]
                    comp_shape = ds.shape[4:]
                    channels = int(np.prod(comp_shape)) if comp_shape else 1
                    field_specs.append(FieldSpec(path=f"{group_name}/{dataset_name}", channels=channels))

        if not field_specs:
            raise ValueError(f"no field datasets found in {sample_file}")

        return field_specs

    def _scan_files(self) -> list[FileInfo]:
        file_infos: list[FileInfo] = []
        for path in self.files:
            with h5py.File(path, "r") as h5:
                probe = h5[self.field_specs[0].path]
                n_sims = int(probe.shape[0])
                n_steps = int(probe.shape[1])
                if self.include_labels:
                    labels = self._extract_label_table(h5, n_sims)
                else:
                    labels = np.zeros((n_sims, len(LABEL_NAMES)), dtype=np.float32)
            file_infos.append(FileInfo(path=path, n_sims=n_sims, n_steps=n_steps, labels=labels))
        return file_infos

    def _build_index(self) -> list[tuple[int, int, int]]:
        entries: list[tuple[int, int, int]] = []
        needed_frames = self.total_frames if self.target_frames > 0 else self.context_frames
        for file_idx, file_info in enumerate(self.file_infos):
            max_start = file_info.n_steps - needed_frames
            if max_start < 0:
                continue
            for sim_idx in range(file_info.n_sims):
                if self.index_mode == "single_clip":
                    start = self._single_clip_start(max_start)
                    entries.append((file_idx, sim_idx, start))
                    if self.max_samples is not None and len(entries) >= self.max_samples:
                        return entries
                    continue

                for start in range(0, max_start + 1, self.stride):
                    entries.append((file_idx, sim_idx, start))
                    if self.max_samples is not None and len(entries) >= self.max_samples:
                        return entries
        return entries

    def _single_clip_start(self, max_start: int) -> int:
        if self.clip_selection == "start":
            return 0
        if self.clip_selection == "end":
            return int(max_start)
        return int(max_start // 2)

    def _extract_label_table(self, h5: h5py.File, n_sims: int) -> np.ndarray:
        if "scalars" not in h5:
            raise KeyError(f"missing scalars group in {h5.filename}")

        group = h5["scalars"]
        ordered_keys = [key for key in LABEL_NAMES if key in group]
        if len(ordered_keys) != len(LABEL_NAMES):
            fallback = [key for key in sorted(group.keys()) if key != "L"]
            if len(fallback) < len(LABEL_NAMES):
                raise KeyError(
                    f"could not find alpha/zeta labels in {h5.filename}; "
                    f"available scalar keys: {sorted(group.keys())}"
                )
            ordered_keys = fallback[: len(LABEL_NAMES)]

        table = np.empty((n_sims, len(ordered_keys)), dtype=np.float32)
        for idx, key in enumerate(ordered_keys):
            values = np.asarray(group[key][()], dtype=np.float32)
            if values.ndim == 0:
                table[:, idx] = float(values)
                continue
            flat = values.reshape(values.shape[0], -1)
            if flat.shape[0] == n_sims:
                table[:, idx] = flat[:, 0]
                continue
            transposed = values.reshape(-1, values.shape[-1]) if values.ndim > 1 else values.reshape(1, -1)
            if transposed.shape[-1] == n_sims:
                table[:, idx] = transposed.reshape(-1, n_sims)[0]
                continue
            if flat.size == 1:
                table[:, idx] = float(flat.reshape(-1)[0])
                continue
            raise ValueError(
                f"unsupported scalar layout for {key} in {h5.filename}: {values.shape}"
            )

        return table

    def _ensure_cache(self) -> None:
        if self._open_files is None:
            self._open_files = OrderedDict()
            weakref.finalize(self, self._close_all_files)

    def _close_all_files(self) -> None:
        if self._open_files is None:
            return
        for h5 in self._open_files.values():
            try:
                h5.close()
            except Exception:
                pass
        self._open_files.clear()

    def _get_file(self, path: Path) -> h5py.File:
        self._ensure_cache()
        assert self._open_files is not None
        key = str(path)
        if key in self._open_files:
            h5 = self._open_files.pop(key)
            self._open_files[key] = h5
            return h5

        while len(self._open_files) >= self.max_open_files:
            _, old_h5 = self._open_files.popitem(last=False)
            old_h5.close()

        h5 = h5py.File(path, "r", libver="latest", swmr=True)
        self._open_files[key] = h5
        return h5

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_open_files"] = None
        return state

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, sim_idx, start = self.index[index]
        info = self.file_infos[file_idx]
        h5 = self._get_file(info.path)

        first_ds = h5[self.field_specs[0].path]
        height, width = int(first_ds.shape[2]), int(first_ds.shape[3])
        total_channels = sum(field.channels for field in self.field_specs)

        needed_frames = self.total_frames if self.target_frames > 0 else self.context_frames
        window = np.empty((needed_frames, height, width, total_channels), dtype=np.float32)

        offset = 0
        stop = start + needed_frames
        for field in self.field_specs:
            arr = np.asarray(h5[field.path][sim_idx, start:stop], dtype=np.float32)
            arr = arr.reshape(needed_frames, height, width, field.channels)
            window[..., offset : offset + field.channels] = arr
            offset += field.channels

        context = torch.from_numpy(window[: self.context_frames]).permute(3, 0, 1, 2).contiguous()

        if self.target_frames > 0:
            target_np = window[self.context_frames :]
            target = torch.from_numpy(target_np).permute(3, 0, 1, 2).contiguous()
        else:
            target = torch.empty((0,), dtype=torch.float32)

        if self.resolution is not None and tuple(context.shape[-2:]) != (self.resolution, self.resolution):
            context = F.interpolate(
                context,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )
            if self.target_frames > 0:
                target = F.interpolate(
                    target,
                    size=(self.resolution, self.resolution),
                    mode="bilinear",
                    align_corners=False,
                )

        label = torch.from_numpy(info.labels[sim_idx].copy()).float()
        return {
            "context": context,
            "target": target,
            "label": label,
        }


def collect_split_labels(
    root: str | Path,
    split: str,
    *,
    max_samples: Optional[int] = None,
    index_mode: str = "single_clip",
    clip_selection: str = "center",
) -> np.ndarray:
    dataset = ActiveMatterWindowDataset(
        root=root,
        split=split,
        context_frames=16,
        target_frames=0,
        stride=1,
        resolution=224,
        max_samples=max_samples,
        index_mode=index_mode,
        clip_selection=clip_selection,
    )
    labels = [dataset.file_infos[file_idx].labels[sim_idx] for file_idx, sim_idx, _ in dataset.index]
    return np.asarray(labels, dtype=np.float32)
