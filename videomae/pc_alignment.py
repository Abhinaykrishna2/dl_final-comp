"""Top-PC alignment analysis: how well do the leading principal components
of an encoder's frozen embeddings predict (alpha, zeta)?

For each encoder, fits a linear probe on JUST the top-K principal components
of the embedding (K in {1, 2, 4, 8, 16}) instead of the full 256-d. If the
encoder's leading directions are aligned with (alpha, zeta), the K=2 probe
should already give good MSE; if the encoder is rank-rich but
target-misaligned, K=2 will be much worse than K=full.

This is the strongest single quantitative test for the
"rank-vs-alignment" claim in the report.

Reads existing .npz embedding dumps; no new training required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .utils import (
    LabelNormalizer,
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    mse_report,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-PC alignment analysis.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more 'path:label' pointing at run directories with embeddings/{train,valid,test}.npz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 256],
                        help="Numbers of top PCs to keep.")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-wd", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
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


def _fit_probe(train_x: np.ndarray, train_y_n: np.ndarray, valid_x: np.ndarray, valid_y_n: np.ndarray,
               test_x: np.ndarray, test_y_n: np.ndarray,
               *, lr: float, wd: float, epochs: int, device: torch.device, seed: int) -> dict:
    """Closed-form ridge regression with grid-searched lambda for the probe.

    For top-K PC features this is far more reliable than SGD because the
    closed form finds the global optimum for MSE+lambda*||W||^2 without
    relying on optimizer hyperparameters or sufficient epochs. Lambda is
    selected by validation MSE.
    """
    del lr, epochs, seed  # unused (kept for argparse compat); see docstring
    # Augment with a bias column
    def aug(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    Xtr = aug(train_x.astype(np.float64))
    Xva = aug(valid_x.astype(np.float64))
    Xte = aug(test_x.astype(np.float64))
    Ytr = train_y_n.astype(np.float64)
    in_dim = Xtr.shape[1]
    XtX = Xtr.T @ Xtr
    XtY = Xtr.T @ Ytr
    eye = np.eye(in_dim, dtype=np.float64)
    eye[-1, -1] = 0.0  # do not regularize the bias term
    best_lam: float = 0.0
    best_v: float = float("inf")
    best_W: np.ndarray | None = None
    lambdas = [float(wd)] if wd > 0 else [0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0, 1e4]
    if wd > 0 and 0.0 not in lambdas:
        lambdas = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0, 1e4]
    for lam in lambdas:
        try:
            W = np.linalg.solve(XtX + lam * eye, XtY)
        except np.linalg.LinAlgError:
            continue
        v = float(((Xva @ W - valid_y_n.astype(np.float64)) ** 2).mean())
        if v < best_v:
            best_v = v
            best_lam = lam
            best_W = W
    if best_W is None:
        raise RuntimeError("ridge regression failed for all lambdas")
    Yte_pred = Xte @ best_W
    Yva_pred = Xva @ best_W
    return {
        "valid_mse": mse_report(Yva_pred.astype(np.float32), valid_y_n)["mean_mse"],
        "test_mse": mse_report(Yte_pred.astype(np.float32), test_y_n)["mean_mse"],
        "test_alpha_mse": mse_report(Yte_pred.astype(np.float32), test_y_n)["alpha_mse"],
        "test_zeta_mse": mse_report(Yte_pred.astype(np.float32), test_y_n)["zeta_mse"],
        "best_ridge_lambda": best_lam,
    }


def _zscore(train_x: np.ndarray, valid_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (valid_x - mean) / std, (test_x - mean) / std


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_torch_runtime(deterministic=False)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    runs = _parse_runs(args.runs)

    summary: dict[str, dict] = {"runs": {}}
    for label, run_dir in runs:
        train = _load_split(run_dir, "train")
        valid = _load_split(run_dir, "valid")
        test = _load_split(run_dir, "test")
        if train is None or valid is None or test is None:
            print({"label": label, "status": "missing embeddings"})
            continue
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test
        # Z-score features using train stats
        tr_z, va_z, te_z = _zscore(train_x, valid_x, test_x)
        # Label normalization on train labels
        label_norm = LabelNormalizer.fit(train_y)
        tr_y_n = label_norm.transform(train_y)
        va_y_n = label_norm.transform(valid_y)
        te_y_n = label_norm.transform(test_y)

        # Fit PCA on train embeddings
        # Use SVD on centered features (centering is implicit since z-score makes mean zero)
        u, s, vt = np.linalg.svd(tr_z, full_matrices=False)
        tot = (s ** 2).sum()
        explained = (s ** 2) / tot
        # Fit linear probes on top-K projections for several K
        records: list[dict] = []
        max_k = min(int(tr_z.shape[1]), int(tr_z.shape[0] - 1))
        for k in args.ks:
            k_eff = min(k, max_k)
            tr_pc = tr_z @ vt.T[:, :k_eff]
            va_pc = va_z @ vt.T[:, :k_eff]
            te_pc = te_z @ vt.T[:, :k_eff]
            m = _fit_probe(
                tr_pc, tr_y_n, va_pc, va_y_n, te_pc, te_y_n,
                lr=args.probe_lr, wd=args.probe_wd, epochs=args.probe_epochs,
                device=device, seed=args.seed + k_eff,
            )
            records.append({
                "k_requested": int(k),
                "k_effective": int(k_eff),
                "explained_variance": float(explained[:k_eff].sum()),
                **m,
            })
            print(f"[pc-align] {label}: K={k_eff:>3} explains {records[-1]['explained_variance']*100:.1f}% var, test_MSE={m['test_mse']:.4f}", flush=True)
        # Correlation of leading PCs with alpha and zeta
        pc_scores = tr_z @ vt.T[:, : min(8, max_k)]
        corrs: list[dict] = []
        for k in range(min(8, max_k)):
            ca = float(np.corrcoef(pc_scores[:, k], train_y[:, 0])[0, 1])
            cz = float(np.corrcoef(pc_scores[:, k], train_y[:, 1])[0, 1])
            corrs.append({"pc": k + 1, "corr_with_alpha": ca, "corr_with_zeta": cz, "max_abs_corr": max(abs(ca), abs(cz))})
        summary["runs"][label] = {
            "run_dir": str(run_dir),
            "n_train": int(tr_z.shape[0]),
            "embedding_dim": int(tr_z.shape[1]),
            "topk_probe": records,
            "pc_corrs_with_targets": corrs,
        }
    save_json(out_dir / "pc_alignment.json", summary)

    # Plot K vs test MSE
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None and summary["runs"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, payload in summary["runs"].items():
            ks = [r["k_effective"] for r in payload["topk_probe"]]
            ms = [r["test_mse"] for r in payload["topk_probe"]]
            ax.plot(ks, ms, "o-", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("K = number of leading PCs used as features")
        ax.set_ylabel("test normalized MSE (linear probe on top-K PCs)")
        ax.set_title("How many leading PCs does each encoder need?")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf_path = out_dir / "pc_alignment.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print({"pdf": str(pdf_path)}, flush=True)


if __name__ == "__main__":
    main()
