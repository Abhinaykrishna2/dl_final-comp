"""Centered-Kernel-Alignment (CKA) similarity between encoders.

Computes pairwise CKA between two or more frozen encoders' representations on
the same input set (typically the validation split). High CKA (~1) means the
encoders learned essentially the same representation up to a linear transform;
low CKA means they learned fundamentally different features.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019, https://arxiv.org/abs/1905.00414.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """
    Parameters
    ----------
    (No input parameters; reads from ``sys.argv`` via ``argparse``.)

    Output
    ------
    Output returned: An ``argparse.Namespace`` with attributes ``runs`` (list of "path:label" specs), ``out_dir``, ``split``, ``max_samples``, ``seed``.

    Purpose
    -------
    Define the CLI for ``python -m videomae.cka_compare``. Each ``--runs`` entry takes the form ``<path>:<label>`` so labels appear in the heatmap directly.

    Assumptions
    -----------
    Designed to be called once at the start of ``main``. The path component of each ``--runs`` entry must contain ``embeddings/<split>.npz`` (produced by ``export_embeddings.py``).

    Notes
    -----
    ``--max-samples`` is useful when the train split is too large to materialize a dense Gram matrix in memory; for our 224x224 dataset and ~10 encoders, the val split (5K samples, 5K x 5K Gram) fits comfortably.
    """
    parser = argparse.ArgumentParser(description="Pairwise CKA between encoder embeddings.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more 'path:label' pairs pointing at run output directories.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_runs(runs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in runs:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, Path(spec).name
        out.append((label, Path(path).expanduser().resolve()))
    return out


def _load_split(run_dir: Path, split: str) -> np.ndarray | None:
    p = run_dir / "embeddings" / f"{split}.npz"
    if not p.exists():
        return None
    payload = np.load(p)
    return payload["embeddings"].astype(np.float64)


def _gram_linear(x: np.ndarray) -> np.ndarray:
    return x @ x.T


def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """
    Parameters
    ----------
    **Input parameter 1:** x - First embedding matrix of shape ``(n_samples, d_x)``.
    **Input parameter 2:** y - Second embedding matrix of shape ``(n_samples, d_y)``. ``d_x`` and ``d_y`` may differ.

    Output
    ------
    Output returned: A float in ``[0, 1]`` (or ``NaN`` for degenerate inputs) measuring linear similarity between the two representations.

    Purpose
    -------
    Compute the linear-kernel CKA between two embedding sets sharing the same n_samples. Implemented as ``HSIC(K_x, K_y) / sqrt(HSIC(K_x, K_x) * HSIC(K_y, K_y))`` with linear (uncentered) Gram matrices ``K_x = X X^T``, ``K_y = Y Y^T`` and centering matrix ``H``. Reference: Kornblith et al., ICML 2019, arXiv:1905.00414.

    Assumptions
    -----------
    Designed for ``np.float64`` inputs (the caller in ``main`` casts to float64 before invoking). Both inputs must have the same number of samples in the same order; rows of ``x`` and ``y`` are assumed to correspond to the same data points.

    Notes
    -----
    Returns ``NaN`` when one of the inputs is degenerate (zero variance after centering), which is preferable to dividing by zero. Linear CKA is bounded in ``[0, 1]`` and equals 1 if and only if ``y = a * x * R + b`` for some scalar ``a``, orthogonal ``R``, and constant ``b``.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires the same number of samples")
    Kx = _gram_linear(x)
    Ky = _gram_linear(y)
    Kxc = _center_gram(Kx)
    Kyc = _center_gram(Ky)
    num = (Kxc * Kyc).sum()
    denom = np.sqrt((Kxc * Kxc).sum() * (Kyc * Kyc).sum())
    if denom <= 0:
        return float("nan")
    return float(num / denom)


def main() -> None:
    """
    Parameters
    ----------
    (No input parameters; CLI args come from ``parse_args``.)

    Output
    ------
    Output returned: ``None``. Side effect: writes ``cka_<split>.json`` (full pairwise CKA matrix) and ``cka_<split>.pdf`` (a labeled heatmap with annotated cell values) to the output directory.

    Purpose
    -------
    Load embeddings for every ``--runs`` entry, trim them to a common subset (same indices in the same order across encoders), compute every pairwise linear CKA, and render the result as a heatmap.

    Assumptions
    -----------
    Designed for embeddings produced by ``export_embeddings.py``. Missing splits are silently skipped per encoder. The minimum-common-subset trim is required because CKA needs identical sample counts and orderings; we use a fixed RNG-seeded permutation so re-running with the same args produces an identical figure.

    Notes
    -----
    The heatmap annotates each cell with its CKA value to two decimal places; viridis colormap with ``vmin=0, vmax=1`` for direct visual comparison across runs.
    """
    args = parse_args()
    runs = _parse_runs(args.runs)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    embeddings: dict[str, np.ndarray] = {}
    sample_count: int | None = None
    for label, run_dir in runs:
        z = _load_split(run_dir, args.split)
        if z is None:
            print({"label": label, "status": "missing", "split": args.split}, flush=True)
            continue
        if args.max_samples is not None and z.shape[0] > args.max_samples:
            idx = rng.choice(z.shape[0], size=args.max_samples, replace=False)
            z = z[idx]
        if sample_count is None:
            sample_count = int(z.shape[0])
        elif z.shape[0] != sample_count:
            # Subsample to the smallest common size
            sample_count = min(sample_count, int(z.shape[0]))
        embeddings[label] = z
    # Trim everything to the common size, with a fixed permutation.
    if sample_count is None:
        raise SystemExit("no embeddings loaded")
    common_idx = rng.permutation(min(z.shape[0] for z in embeddings.values()))[:sample_count]
    for label in list(embeddings.keys()):
        embeddings[label] = embeddings[label][:sample_count][common_idx]

    labels = list(embeddings.keys())
    n = len(labels)
    cka = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = linear_cka(embeddings[labels[i]], embeddings[labels[j]])
            cka[i, j] = cka[j, i] = v
    print({"labels": labels, "cka": cka.tolist()}, flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    payload = {
        "split": args.split,
        "n_samples": sample_count,
        "labels": labels,
        "cka_matrix": cka.tolist(),
    }
    with open(out_dir / f"cka_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(0.6 * n + 3, 0.6 * n + 2.5))
        im = ax.imshow(cka, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize="small")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize="small")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cka[i, j]:.2f}", ha="center", va="center",
                        color="white" if cka[i, j] < 0.5 else "black", fontsize="small")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"Linear CKA between encoders (split={args.split})")
        fig.tight_layout()
        pdf_path = out_dir / f"cka_{args.split}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print({"pdf": str(pdf_path)}, flush=True)


if __name__ == "__main__":
    main()
