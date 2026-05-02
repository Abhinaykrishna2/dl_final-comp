"""Per-class separability analysis: how well-separated are the 45 unique
(alpha, zeta) classes in each encoder's frozen embedding space?

Computes:
  * Class centroid for each unique (alpha, zeta) combination on the train split.
  * Mean intra-class distance (sample to its class centroid).
  * Mean inter-class distance (centroid to centroid).
  * Fisher-style separability ratio = inter / intra.
  * On valid+test: nearest-centroid classification accuracy (informational
    only -- the assignment evaluates with linear/kNN regression, this is a
    diagnostic).

Reads existing .npz embedding dumps; no new training required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class separability of (alpha, zeta) combos.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more 'path:label' pairs.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-norm", choices=["none", "zscore", "l2", "zscore_l2"], default="zscore")
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
    return payload["embeddings"].astype(np.float64), payload["labels"].astype(np.float32)


def _norm(train_x: np.ndarray, valid_x: np.ndarray, test_x: np.ndarray, mode: str):
    if mode == "none":
        return train_x, valid_x, test_x
    if mode == "zscore":
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (train_x - mean) / std, (valid_x - mean) / std, (test_x - mean) / std
    if mode == "l2":
        def f(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            return x / np.where(n < 1e-9, 1.0, n)
        return f(train_x), f(valid_x), f(test_x)
    if mode == "zscore_l2":
        a, b, c = _norm(train_x, valid_x, test_x, "zscore")
        return _norm(a, b, c, "l2")
    raise ValueError(mode)


def _class_key(y: np.ndarray) -> np.ndarray:
    return np.array([f"a={a:.4f}|z={z:.4f}" for (a, z) in y])


def _classify_nearest_centroid(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Returns predicted index in centroids
    d = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=-1)
    return d.argmin(axis=1)


def main() -> None:
    args = parse_args()
    runs = _parse_runs(args.runs)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {"feature_norm": args.feature_norm, "runs": {}}
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
        train_x, valid_x, test_x = _norm(train_x, valid_x, test_x, args.feature_norm)

        keys_train = _class_key(train_y)
        unique_keys = sorted(set(keys_train.tolist()))
        n_classes = len(unique_keys)
        # Class centroids
        centroids = np.zeros((n_classes, train_x.shape[1]), dtype=np.float64)
        intra_distances: list[float] = []
        for i, k in enumerate(unique_keys):
            mask = keys_train == k
            members = train_x[mask]
            centroids[i] = members.mean(axis=0)
            d = np.linalg.norm(members - centroids[i], axis=1)
            intra_distances.append(float(d.mean()))
        mean_intra = float(np.mean(intra_distances))

        # Inter-centroid distances (pairwise)
        cd = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
        iu = np.triu_indices(n_classes, k=1)
        mean_inter = float(cd[iu].mean())

        fisher = mean_inter / max(mean_intra, 1e-12)

        # Nearest-centroid accuracy on valid + test (informational only)
        keys_valid = _class_key(valid_y)
        valid_pred = _classify_nearest_centroid(valid_x, centroids)
        valid_pred_keys = np.array([unique_keys[i] for i in valid_pred])
        valid_acc = float((valid_pred_keys == keys_valid).mean())

        keys_test = _class_key(test_y)
        test_pred = _classify_nearest_centroid(test_x, centroids)
        test_pred_keys = np.array([unique_keys[i] for i in test_pred])
        test_acc = float((test_pred_keys == keys_test).mean())

        record = {
            "run_dir": str(run_dir),
            "n_classes": n_classes,
            "mean_intra_class_distance": mean_intra,
            "mean_inter_centroid_distance": mean_inter,
            "fisher_separability": fisher,
            "valid_nearest_centroid_acc": valid_acc,
            "test_nearest_centroid_acc": test_acc,
            "n_train": int(train_x.shape[0]),
            "n_valid": int(valid_x.shape[0]),
            "n_test": int(test_x.shape[0]),
            "embedding_dim": int(train_x.shape[1]),
        }
        summary["runs"][label] = record
        print(f"[separability] {label}: classes={n_classes} | intra={mean_intra:.4f} | "
              f"inter={mean_inter:.4f} | fisher={fisher:.4f} | "
              f"NC valid_acc={valid_acc:.4f}, test_acc={test_acc:.4f}", flush=True)

    out_path = out_dir / "class_separability.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print({"json": str(out_path)})

    # Bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None and summary["runs"]:
        labels_ = list(summary["runs"].keys())
        fishers = [summary["runs"][l]["fisher_separability"] for l in labels_]
        valid_accs = [summary["runs"][l]["valid_nearest_centroid_acc"] for l in labels_]
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].bar(np.arange(len(labels_)), fishers)
        ax[0].set_xticks(np.arange(len(labels_)))
        ax[0].set_xticklabels(labels_, rotation=20, ha="right", fontsize="small")
        ax[0].set_ylabel("Fisher separability (inter / intra centroid distance)")
        ax[0].set_title("Per-class separability")
        ax[1].bar(np.arange(len(labels_)), valid_accs)
        ax[1].set_xticks(np.arange(len(labels_)))
        ax[1].set_xticklabels(labels_, rotation=20, ha="right", fontsize="small")
        ax[1].set_ylabel("Nearest-centroid accuracy on validation")
        ax[1].set_title("Diagnostic NC classification (45 classes)")
        fig.tight_layout()
        pdf_path = out_dir / "class_separability.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print({"pdf": str(pdf_path)}, flush=True)


if __name__ == "__main__":
    main()
