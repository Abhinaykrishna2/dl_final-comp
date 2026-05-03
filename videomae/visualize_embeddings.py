"""PCA and t-SNE visualization of frozen-encoder embeddings, colored by alpha and zeta.

Reads ``videomae/artifacts/<run>/embeddings/{train,valid,test}.npz`` for one or
more encoders and produces a multi-panel figure where each row is one encoder
and each column is one coloring (PCA-2D colored by alpha, by zeta; t-SNE-2D
colored by alpha, by zeta).

Visually demonstrates the "rank vs alignment" trade-off: a supervised encoder
collapses to ~2 directions perfectly aligned with (alpha, zeta), while VideoMAE
embeds along many more directions but with little label structure visible in
the leading components.
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
    Output returned: An ``argparse.Namespace`` with attributes ``runs`` (list of "path:label" specs), ``out_dir``, ``split``, ``no_tsne``, ``max_samples``, ``seed``.

    Purpose
    -------
    Define the CLI for ``python -m videomae.visualize_embeddings``. Each ``--runs`` entry takes the form ``<path>:<label>`` so labels appear directly in figure titles.

    Assumptions
    -----------
    Designed to be called once at the start of ``main``. The path component of each ``--runs`` entry must contain ``embeddings/<split>.npz`` (produced by ``export_embeddings.py``).

    Notes
    -----
    ``--no-tsne`` skips t-SNE entirely (saves ~5x time on small splits, ~20x on large ones). ``--max-samples`` is useful for the train split where t-SNE is the bottleneck.
    """
    parser = argparse.ArgumentParser(description="PCA / t-SNE visualization of embeddings.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more 'path:label' pairs pointing at run output directories.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid",
                        help="Which split to visualize. Default: valid.")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE (faster).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample N points before PCA / t-SNE.")
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


def _load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = run_dir / "embeddings" / f"{split}.npz"
    if not p.exists():
        return None
    payload = np.load(p)
    return payload["embeddings"].astype(np.float32), payload["labels"].astype(np.float32)


def main() -> None:
    """
    Parameters
    ----------
    (No input parameters; CLI args come from ``parse_args``.)

    Output
    ------
    Output returned: ``None``. Side effect: writes ``embeddings_<split>.pdf`` (multi-panel figure) and ``embeddings_<split>_summary.json`` (per-run PCA explained-variance ratio) to the output directory.

    Purpose
    -------
    Build a multi-panel figure where each row is one encoder and each column is one coloring scheme: PCA-2D colored by alpha, PCA-2D colored by zeta, [t-SNE-2D colored by alpha, t-SNE-2D colored by zeta]. Used to visually demonstrate the rank-vs-alignment trade-off that the report's PC-alignment analysis quantifies.

    Assumptions
    -----------
    Designed for embeddings produced by ``export_embeddings.py`` (a ``.npz`` per split with ``embeddings`` and ``labels`` keys). Encoders missing the requested split are silently skipped with a printed status. Requires ``matplotlib`` and ``scikit-learn``.

    Notes
    -----
    t-SNE perplexity is set to ``max(5, min(30, (n - 1) / 3))`` so it scales reasonably with the number of samples; PCA uses ``random_state=args.seed`` so two runs with the same arguments produce identical figures.
    """
    args = parse_args()
    runs = _parse_runs(args.runs)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")
    from sklearn.decomposition import PCA
    if not args.no_tsne:
        from sklearn.manifold import TSNE

    rng = np.random.default_rng(args.seed)
    all_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, run_dir in runs:
        out = _load_split(run_dir, args.split)
        if out is None:
            print({"label": label, "status": "missing", "split": args.split, "run_dir": str(run_dir)}, flush=True)
            continue
        z, y = out
        if args.max_samples is not None and z.shape[0] > args.max_samples:
            idx = rng.choice(z.shape[0], size=args.max_samples, replace=False)
            z, y = z[idx], y[idx]
        all_data.append((label, z, y))

    n_runs = len(all_data)
    if n_runs == 0:
        raise SystemExit("no embeddings found; check --runs paths")

    n_cols = 2 + (0 if args.no_tsne else 2)  # PCA-alpha, PCA-zeta, [t-SNE-alpha, t-SNE-zeta]
    fig, axes = plt.subplots(n_runs, n_cols, figsize=(4.0 * n_cols, 3.5 * n_runs))
    if n_runs == 1:
        axes = axes[None, :]
    summary: list[dict] = []
    for r, (label, z, y) in enumerate(all_data):
        # PCA
        pca = PCA(n_components=2, random_state=args.seed)
        z_pca = pca.fit_transform(z)
        explained = pca.explained_variance_ratio_.tolist()
        for c, name in enumerate(["alpha", "zeta"]):
            ax = axes[r, c]
            sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=y[:, c], cmap="viridis", s=12, alpha=0.85)
            plt.colorbar(sc, ax=ax, fraction=0.046)
            ax.set_title(f"{label} - PCA, color={name}\n(explained {explained[0]*100:.1f}%, {explained[1]*100:.1f}%)")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        if not args.no_tsne:
            n = z.shape[0]
            perp = max(5.0, min(30.0, (n - 1) / 3.0))
            tsne = TSNE(n_components=2, perplexity=perp, init="pca", random_state=args.seed)
            z_tsne = tsne.fit_transform(z)
            for c, name in enumerate(["alpha", "zeta"]):
                ax = axes[r, 2 + c]
                sc = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y[:, c], cmap="viridis", s=12, alpha=0.85)
                plt.colorbar(sc, ax=ax, fraction=0.046)
                ax.set_title(f"{label} - t-SNE, color={name}\n(perplexity {perp:.1f})")
                ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        summary.append({
            "label": label,
            "n_samples": int(z.shape[0]),
            "embedding_dim": int(z.shape[1]),
            "pca_explained_ratio": explained,
            "split": args.split,
        })

    fig.tight_layout()
    pdf_path = out_dir / f"embeddings_{args.split}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    with open(out_dir / f"embeddings_{args.split}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": summary, "split": args.split, "pdf": str(pdf_path)}, f, indent=2)
    print({"pdf": str(pdf_path), "n_runs": n_runs}, flush=True)


if __name__ == "__main__":
    main()
