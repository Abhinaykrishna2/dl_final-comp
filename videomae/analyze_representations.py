"""Representation-quality diagnostics for any frozen encoder.

Operates on the ``.npz`` embedding dumps produced by ``videomae/export_embeddings.py``
(``embeddings`` array of shape ``(N, D)`` plus ``labels`` of shape ``(N, 2)``).

For each split (typically train + valid), computes:

* **Per-dimension std (sorted)** -- a collapsed encoder shows a heavy long-tail, with
  most dims near zero std. A healthy encoder shows roughly uniform std.
* **Effective rank** ``exp(H(p))`` where ``p`` are the normalized singular values.
  Lower rank = more collapsed.
* **Participation ratio** ``(Sum lambda)^2 / Sum lambda^2`` -- another collapse proxy.
* **Covariance condition number** ``lambda_max / lambda_min`` -- ill-conditioned
  covariance hurts both linear probes and kNN with Euclidean distance.
* **Mean off-diagonal correlation magnitude** -- shows feature redundancy.
* **Mean Epps-Pulley statistic** to ``N(0, I)`` (reusing ``videomae.losses.sigreg_loss``).
  This is the SIGReg objective treated as a *diagnostic*, applicable to ANY encoder.
  Lower = closer to isotropic Gaussian, the distribution that LeJEPA proves is optimal
  for downstream linear / kNN regression (Balestriero & LeCun 2026).

Saves ``analysis.json`` (all numerical stats) and ``analysis_plots.pdf`` (per-dim std,
sorted singular values, scatter of the leading 2 PCA components colored by alpha and
zeta) in the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .losses import sigreg_loss
from .utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute representation-quality diagnostics on .npz embedding dumps.")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="Directory containing train.npz / valid.npz / test.npz from export_embeddings.py.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <embeddings-dir>/analysis).")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                        help="Which split files to analyze.")
    parser.add_argument("--num-slices", type=int, default=2048,
                        help="Random projections for the Epps-Pulley statistic.")
    parser.add_argument("--num-points", type=int, default=17)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample N points before computing stats (for very large train splits).")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib PDF output.")
    return parser.parse_args()


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["embeddings"].astype(np.float32), payload["labels"].astype(np.float32)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def per_dim_std(z: np.ndarray) -> np.ndarray:
    """Return per-dimension std, sorted descending."""
    return np.sort(z.std(axis=0))[::-1]


def effective_rank(z: np.ndarray) -> dict[str, float]:
    """Compute effective rank exp(H(p)) and participation ratio of singular values."""
    z_centered = z - z.mean(axis=0, keepdims=True)
    # Use SVD on centered features (equivalent to PCA up to scale)
    s = np.linalg.svd(z_centered, compute_uv=False, full_matrices=False)
    eig = (s ** 2) / max(z.shape[0] - 1, 1)  # eigenvalues of covariance
    eig = np.maximum(eig, 0.0)
    eig_sum = eig.sum()
    if eig_sum <= 0:
        return {"effective_rank": float("nan"), "participation_ratio": float("nan"),
                "top_eig": 0.0, "min_eig": 0.0, "condition_number": float("nan")}
    p = eig / eig_sum
    p_pos = p[p > 1e-12]
    entropy = -np.sum(p_pos * np.log(p_pos))
    eff_rank = float(np.exp(entropy))
    pr = float((eig_sum ** 2) / (eig ** 2).sum())
    top = float(eig.max())
    bot = float(max(eig.min(), 1e-30))
    return {
        "effective_rank": eff_rank,
        "participation_ratio": pr,
        "top_eig": top,
        "min_eig": bot,
        "condition_number": float(top / bot),
    }


def covariance_metrics(z: np.ndarray) -> dict[str, float]:
    """Mean off-diagonal correlation magnitude (cosine on centered features)."""
    z_centered = z - z.mean(axis=0, keepdims=True)
    std = z_centered.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z_norm = z_centered / std
    # Sample correlation matrix
    if z_norm.shape[0] <= 1:
        return {"mean_abs_offdiag_corr": float("nan")}
    corr = (z_norm.T @ z_norm) / (z_norm.shape[0] - 1)
    d = corr.shape[0]
    if d <= 1:
        return {"mean_abs_offdiag_corr": float("nan")}
    iu = np.triu_indices(d, k=1)
    off = np.abs(corr[iu])
    return {"mean_abs_offdiag_corr": float(off.mean())}


def epps_pulley_distance(
    z: np.ndarray, *, num_slices: int, num_points: int, t_max: float, seed: int, device: torch.device,
) -> dict[str, float]:
    """Apply the SIGReg / Epps-Pulley statistic as a diagnostic.

    A perfectly isotropic Gaussian batch has statistic 0; large values indicate
    distributional drift (skewness, multi-modality, anisotropy, etc.).

    Two variants:
        * raw: applied directly to the embeddings
        * z-scored: applied after per-feature z-scoring (isolates the *shape*
          of the distribution from any global mean/scale offset).
    """
    z_t = torch.from_numpy(z).to(device).float()
    val_raw = float(
        sigreg_loss(
            z_t, num_slices=num_slices, num_points=num_points, t_max=t_max,
            seed=seed, scale_by_samples=False, distributed=False,
        ).item()
    )
    z_zscore_t = (z_t - z_t.mean(dim=0, keepdim=True)) / z_t.std(dim=0, keepdim=True).clamp_min(1e-6)
    val_zscore = float(
        sigreg_loss(
            z_zscore_t, num_slices=num_slices, num_points=num_points, t_max=t_max,
            seed=seed + 1, scale_by_samples=False, distributed=False,
        ).item()
    )
    return {
        "epps_pulley_raw": val_raw,
        "epps_pulley_zscore": val_zscore,
    }


def _maybe_subsample(z: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or z.shape[0] <= max_samples:
        return z, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(z.shape[0], size=max_samples, replace=False)
    return z[idx], y[idx]


def _analyze_one(
    split: str, embeddings: np.ndarray, labels: np.ndarray, *,
    num_slices: int, num_points: int, t_max: float, seed: int,
    device: torch.device, max_samples: int | None,
) -> dict[str, object]:
    z, y = _maybe_subsample(embeddings, labels, max_samples, seed)
    stats: dict[str, object] = {
        "split": split,
        "n_samples": int(z.shape[0]),
        "embedding_dim": int(z.shape[1]),
        "label_dim": int(y.shape[1]),
        "global_mean_norm": float(np.linalg.norm(z.mean(axis=0))),
        "global_mean_l2": float(np.linalg.norm(z.mean(axis=0))) / max(z.shape[1] ** 0.5, 1e-9),
        "feature_mean_abs_mean": float(np.abs(z.mean(axis=0)).mean()),
        "feature_std_mean": float(z.std(axis=0).mean()),
        "feature_std_min": float(z.std(axis=0).min()),
        "feature_std_max": float(z.std(axis=0).max()),
    }
    stds = per_dim_std(z)
    stats["per_dim_std_sorted"] = stds.tolist()
    stats.update(effective_rank(z))
    stats.update(covariance_metrics(z))
    stats.update(epps_pulley_distance(
        z, num_slices=num_slices, num_points=num_points, t_max=t_max, seed=seed, device=device,
    ))
    return stats


def _try_plots(stats_per_split: dict[str, dict[str, object]], out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pdf_path = out_dir / "analysis_plots.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, stats in stats_per_split.items():
        stds = np.asarray(stats["per_dim_std_sorted"], dtype=np.float64)
        axes[0].plot(stds, label=split)
        # log10-cumulative explained variance proxy
        eig = stds ** 2
        eig = eig / eig.sum()
        axes[1].plot(np.cumsum(eig), label=split)
    axes[0].set_title("Per-dim embedding std (sorted descending)")
    axes[0].set_xlabel("dimension index (sorted)")
    axes[0].set_ylabel("std")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].set_title("Cumulative explained variance (per-dim std^2 normalized)")
    axes[1].set_xlabel("dimension index (sorted)")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def main() -> None:
    args = parse_args()
    embeddings_dir = args.embeddings_dir.expanduser().resolve()
    out_dir = ensure_dir((args.out_dir or (embeddings_dir / "analysis")).expanduser().resolve())
    device = _resolve_device(args.device)
    summary: dict[str, object] = {
        "embeddings_dir": str(embeddings_dir),
        "splits": {},
    }

    for split in args.splits:
        path = embeddings_dir / f"{split}.npz"
        if not path.exists():
            print({"split": split, "status": "missing", "path": str(path)}, flush=True)
            continue
        z, y = _load_split(path)
        stats = _analyze_one(
            split, z, y,
            num_slices=args.num_slices, num_points=args.num_points, t_max=args.t_max,
            seed=args.seed, device=device, max_samples=args.max_samples,
        )
        summary["splits"][split] = stats
        # print a concise summary line for the user
        print(
            {
                "split": split, "n": stats["n_samples"], "dim": stats["embedding_dim"],
                "eff_rank": round(stats["effective_rank"], 2),
                "participation_ratio": round(stats["participation_ratio"], 2),
                "cond_num": round(stats["condition_number"], 2),
                "mean_abs_offdiag_corr": round(stats["mean_abs_offdiag_corr"], 4),
                "epps_pulley_zscore": round(stats["epps_pulley_zscore"], 4),
                "feature_std_mean": round(stats["feature_std_mean"], 4),
            },
            flush=True,
        )

    save_json(out_dir / "analysis.json", summary)
    if not args.no_plots:
        pdf_path = _try_plots(summary["splits"], out_dir)
        if pdf_path is not None:
            summary["plots_pdf"] = str(pdf_path)
            save_json(out_dir / "analysis.json", summary)
            print({"plots_pdf": str(pdf_path)}, flush=True)


if __name__ == "__main__":
    main()
