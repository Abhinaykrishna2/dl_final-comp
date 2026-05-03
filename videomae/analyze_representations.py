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
    """
    Parameters
    ----------
    (No input parameters; reads from ``sys.argv`` via ``argparse``.)

    Output
    ------
    Output returned: An ``argparse.Namespace`` with attributes ``embeddings_dir`` (input dir), ``out_dir`` (output dir; default ``<embeddings-dir>/analysis``), ``splits`` (list of split names), ``num_slices``/``num_points``/``t_max`` (Epps-Pulley hyperparameters), ``seed``, ``max_samples``, ``device``, ``no_plots``.

    Purpose
    -------
    Define the CLI for ``python -m videomae.analyze_representations``. Provides knobs for the number of random projections (``--num-slices``) and quadrature density (``--num-points``) used by the Epps-Pulley diagnostic.

    Assumptions
    -----------
    Designed to be called once at the start of ``main``. ``--embeddings-dir`` must contain ``train.npz`` / ``valid.npz`` / ``test.npz`` (produced by ``export_embeddings.py``).

    Notes
    -----
    ``--max-samples`` is provided so the train split (which can be 10K+ samples) doesn't dominate runtime; the diagnostics are stable at ~5K samples.
    """
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
    """
    Parameters
    ----------
    **Input parameter 1:** z - Embedding matrix of shape ``(n_samples, embedding_dim)``.

    Output
    ------
    Output returned: A 1-D ``np.ndarray`` of length ``embedding_dim`` containing the per-dimension std, sorted in descending order.

    Purpose
    -------
    Compute and return the per-dimension std of an embedding matrix, sorted descending so the long-tail collapse pattern is visually obvious in plots. A healthy encoder shows roughly uniform std across dimensions; a collapsed encoder shows a heavy long tail with most dims near zero.

    Assumptions
    -----------
    Designed for ``np.float32`` or ``np.float64`` embeddings. ``n_samples`` should be at least 2 for the std to be meaningful.

    Notes
    -----
    Std is computed with the default biased estimator (``ddof=0``); using the unbiased estimator changes nothing qualitatively at our sample sizes.
    """
    return np.sort(z.std(axis=0))[::-1]


def effective_rank(z: np.ndarray) -> dict[str, float]:
    """
    Parameters
    ----------
    **Input parameter 1:** z - Embedding matrix of shape ``(n_samples, embedding_dim)``.

    Output
    ------
    Output returned: A dict with five floats: ``effective_rank`` (``exp(H(p))``), ``participation_ratio`` (``(sum lambda)^2 / sum lambda^2``), ``top_eig``, ``min_eig``, ``condition_number`` (``top_eig / min_eig``).

    Purpose
    -------
    Compute three complementary spectrum-based collapse / dimensionality proxies from the singular values of the centered embedding matrix. ``effective_rank`` and ``participation_ratio`` are different ways of quantifying "how many directions does the representation actually use"; ``condition_number`` quantifies how ill-conditioned the covariance is, which directly affects linear-probe and Euclidean-kNN performance.

    Assumptions
    -----------
    Designed for ``np.float32`` or ``np.float64`` embeddings with ``n_samples > 1`` and ``embedding_dim`` not larger than ``n_samples`` (otherwise the SVD is rank-deficient by construction). Returns ``NaN`` for ``effective_rank`` / ``participation_ratio`` / ``condition_number`` if all eigenvalues are zero.

    Notes
    -----
    Uses ``np.linalg.svd`` on centered features rather than ``np.linalg.eigh`` on the covariance because SVD is numerically more stable when the embedding dim and sample count are similar.
    """
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
    """
    Parameters
    ----------
    **Input parameter 1:** z - Embedding matrix of shape ``(n_samples, embedding_dim)``.

    Output
    ------
    Output returned: A dict with one float: ``mean_abs_offdiag_corr`` (the mean absolute value of off-diagonal entries of the sample correlation matrix).

    Purpose
    -------
    Quantify feature redundancy. A high mean ``|off-diag corr|`` means many embedding dimensions are linearly correlated, indicating effective rank is much lower than the embedding dim. Combined with ``effective_rank`` this lets us distinguish "narrow representation" (low rank) from "redundant representation" (full rank but highly correlated).

    Assumptions
    -----------
    Designed for ``np.float32`` or ``np.float64`` embeddings with ``n_samples > 1`` and ``embedding_dim > 1``. Returns ``NaN`` for degenerate inputs.

    Notes
    -----
    Uses upper-triangular indices (``k=1``) to avoid double-counting symmetric correlations and to skip the diagonal (which is identically 1 by construction).
    """
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
    """
    Parameters
    ----------
    **Input parameter 1:** z - Embedding matrix of shape ``(n_samples, embedding_dim)`` as a NumPy array.
    **Input parameter 2:** num_slices - Number of random unit-vector projections used by the SIGReg statistic.
    **Input parameter 3:** num_points - Number of quadrature points along the characteristic-function axis.
    **Input parameter 4:** t_max - Upper bound of the characteristic-function quadrature.
    **Input parameter 5:** seed - RNG seed for the random projections (paired with ``seed + 1`` for the z-scored variant so the two are independent).
    **Input parameter 6:** device - PyTorch device on which to run the SIGReg loss kernel.

    Output
    ------
    Output returned: A dict with two floats: ``epps_pulley_raw`` (statistic on raw embeddings) and ``epps_pulley_zscore`` (statistic after per-feature z-scoring).

    Purpose
    -------
    Apply the SIGReg / Epps-Pulley statistic as a representation-quality diagnostic. A perfectly isotropic Gaussian batch has statistic 0; large values indicate distributional drift (skewness, multi-modality, anisotropy). Closer to 0 means closer to the isotropic Gaussian distribution that LeJEPA (Balestriero & LeCun 2026) proves is optimal for downstream linear / kNN regression.

    Assumptions
    -----------
    Designed for embeddings with at least ~1K samples (the statistic has a known sampling-variance floor below that). Internally re-uses ``videomae.losses.sigreg_loss`` so the diagnostic is exactly the loss used as a training objective by Person A's SIGReg-JEPA, just applied post-hoc.

    Notes
    -----
    The two variants serve different purposes: ``raw`` measures total drift (mean offset + scale + shape), while ``zscore`` isolates the *shape* of the distribution from any global mean/scale offset. The z-scored variant is the more honest comparison across encoders with different output scales.
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
    """
    Parameters
    ----------
    (No input parameters; CLI args come from ``parse_args``.)

    Output
    ------
    Output returned: ``None``. Side effect: writes ``analysis.json`` (numerical stats per split) and optionally ``analysis_plots.pdf`` (per-dim std + cumulative explained variance) to the output directory; prints a one-line concise summary per split to stdout.

    Purpose
    -------
    For every split in ``--splits`` that has a corresponding ``.npz`` under ``--embeddings-dir``, compute the full diagnostic suite (per-dim std, effective rank, participation ratio, condition number, mean off-diag correlation, Epps-Pulley distance to N(0, I)), serialize the results to JSON, and optionally render a PDF summary.

    Assumptions
    -----------
    Designed for ``.npz`` embedding dumps produced by ``export_embeddings.py``. The output directory is created if it does not exist; existing files are overwritten.

    Notes
    -----
    Stats are serialized with ``save_json`` which uses ``sort_keys=True`` so successive runs produce stable JSON diffs. The optional plots reuse the same data and are skipped if ``--no-plots`` is set or matplotlib is unavailable.
    """
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
