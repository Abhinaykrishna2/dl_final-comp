from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import LABEL_NAMES


def configure_torch_runtime(*, deterministic: bool = False, allow_tf32: bool = True) -> None:
    if deterministic:
        allow_tf32 = False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


class LabelNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, labels: np.ndarray) -> "LabelNormalizer":
        mean = labels.mean(axis=0)
        std = labels.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelNormalizer":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float32), std=np.asarray(payload["std"], dtype=np.float32))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_names": list(LABEL_NAMES),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def transform(self, labels: np.ndarray) -> np.ndarray:
        return (labels - self.mean) / self.std

    def inverse_transform(self, labels: np.ndarray) -> np.ndarray:
        return labels * self.std + self.mean


def mse_report(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mse = ((pred - target) ** 2).mean(axis=0)
    report = {f"{name}_mse": float(value) for name, value in zip(LABEL_NAMES, mse)}
    report["mean_mse"] = float(mse.mean())
    return report


def pool_features(features: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "avg":
        return features.mean(dim=(-1, -2))
    if pool == "flatten":
        return features.flatten(1)
    if pool == "avgmax":
        avg = features.mean(dim=(-1, -2))
        maxv = features.amax(dim=(-1, -2))
        return torch.cat([avg, maxv], dim=1)
    raise ValueError(f"unsupported pool mode: {pool}")


def normalize_feature_splits(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    test_x: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if mode == "none":
        return train_x, valid_x, test_x, {"mode": "none"}

    if mode == "zscore":
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (
            (train_x - mean) / std,
            (valid_x - mean) / std,
            (test_x - mean) / std,
            {"mode": "zscore", "mean": mean.tolist(), "std": std.tolist()},
        )

    if mode == "zscore_l2":
        z_train, z_valid, z_test, stats = normalize_feature_splits(train_x, valid_x, test_x, "zscore")
        l2_train, l2_valid, l2_test, _ = normalize_feature_splits(z_train, z_valid, z_test, "l2")
        stats["mode"] = "zscore_l2"
        return l2_train, l2_valid, l2_test, stats

    if mode == "l2":
        def _l2(x: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms = np.where(norms < 1e-6, 1.0, norms)
            return x / norms

        return _l2(train_x), _l2(valid_x), _l2(test_x), {"mode": "l2"}

    raise ValueError(f"unsupported feature normalization mode: {mode}")
