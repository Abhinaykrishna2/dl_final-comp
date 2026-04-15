from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .utils import LabelNormalizer, ensure_dir, mse_report, normalize_feature_splits, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run kNN regression on frozen embeddings.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/knn"))
    parser.add_argument("--neighbors", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--weights", nargs="+", choices=["uniform", "distance"], default=["uniform", "distance"])
    parser.add_argument("--metric", nargs="+", choices=["euclidean", "manhattan", "cosine"], default=["cosine", "euclidean"])
    parser.add_argument("--feature-norm", choices=["none", "zscore", "l2", "zscore_l2"], default="zscore_l2")
    return parser.parse_args()


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["embeddings"].astype(np.float32), payload["labels"].astype(np.float32)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    train_x, train_y = _load_split(args.train_file)
    valid_x, valid_y = _load_split(args.valid_file)
    test_x, test_y = _load_split(args.test_file)

    train_x, valid_x, test_x, feature_stats = normalize_feature_splits(
        train_x, valid_x, test_x, args.feature_norm
    )
    label_norm = LabelNormalizer.fit(train_y)
    train_y_n = label_norm.transform(train_y)
    valid_y_n = label_norm.transform(valid_y)
    test_y_n = label_norm.transform(test_y)

    best: dict[str, object] | None = None
    search_results: list[dict[str, object]] = []

    for n_neighbors in args.neighbors:
        for weights in args.weights:
            for metric in args.metric:
                model = KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    algorithm="brute" if metric == "cosine" else "auto",
                    n_jobs=-1,
                )
                model.fit(train_x, train_y_n)
                valid_pred_n = model.predict(valid_x)
                valid_pred = label_norm.inverse_transform(valid_pred_n)
                valid_report = mse_report(valid_pred, valid_y)
                result = {
                    "n_neighbors": n_neighbors,
                    "weights": weights,
                    "metric": metric,
                    "valid": valid_report,
                    "valid_normalized": mse_report(valid_pred_n, valid_y_n),
                }
                search_results.append(result)
                print(result, flush=True)

                if best is None or valid_report["mean_mse"] < best["valid"]["mean_mse"]:
                    test_pred_n = model.predict(test_x)
                    test_pred = label_norm.inverse_transform(test_pred_n)
                    best = {
                        "n_neighbors": n_neighbors,
                        "weights": weights,
                        "metric": metric,
                        "valid": valid_report,
                        "valid_normalized": mse_report(valid_pred_n, valid_y_n),
                        "test": mse_report(test_pred, test_y),
                        "test_normalized": mse_report(test_pred_n, test_y_n),
                        "test_pred": test_pred,
                        "test_pred_normalized": test_pred_n,
                    }

    if best is None:
        raise RuntimeError("kNN search produced no result")

    np.savez_compressed(
        out_dir / "best_test_predictions.npz",
        pred=np.asarray(best.pop("test_pred"), dtype=np.float32),
        target=test_y.astype(np.float32),
        pred_normalized=np.asarray(best.pop("test_pred_normalized"), dtype=np.float32),
        target_normalized=test_y_n.astype(np.float32),
    )
    save_json(
        out_dir / "metrics.json",
        {
            "best": best,
            "search_results": search_results,
            "feature_stats": feature_stats,
            "label_stats": label_norm.to_dict(),
        },
    )


if __name__ == "__main__":
    main()
