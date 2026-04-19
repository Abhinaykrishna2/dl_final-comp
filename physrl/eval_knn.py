from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor

from .utils import LabelNormalizer, choose_device, ensure_dir, mse_report, normalize_feature_splits, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run kNN regression on frozen embeddings.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/knn"))
    parser.add_argument("--neighbors", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100])
    parser.add_argument("--weights", nargs="+", choices=["uniform", "distance"], default=["uniform", "distance"])
    parser.add_argument("--metric", nargs="+", choices=["euclidean", "manhattan", "cosine"], default=["cosine", "euclidean"])
    parser.add_argument(
        "--feature-norm",
        nargs="+",
        choices=["none", "zscore", "l2", "zscore_l2"],
        default=["zscore_l2", "zscore", "l2"],
    )
    parser.add_argument("--backend", choices=["auto", "torch", "sklearn"], default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--query-batch-size", type=int, default=2048)
    return parser.parse_args()


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["embeddings"].astype(np.float32), payload["labels"].astype(np.float32)


def _resolve_backend(backend: str, device: torch.device) -> str:
    if backend != "auto":
        return backend
    return "torch" if device.type == "cuda" else "sklearn"


def _knn_predict_sklearn(
    *,
    train_x: np.ndarray,
    train_y_n: np.ndarray,
    query_x: np.ndarray,
    n_neighbors: int,
    weights: str,
    metric: str,
) -> np.ndarray:
    model = KNeighborsRegressor(
        n_neighbors=min(n_neighbors, train_x.shape[0]),
        weights=weights,
        metric=metric,
        algorithm="brute" if metric == "cosine" else "auto",
        n_jobs=-1,
    )
    model.fit(train_x, train_y_n)
    return model.predict(query_x).astype(np.float32)


def _gather_predictions(
    neighbor_targets: torch.Tensor,
    neighbor_distances: torch.Tensor,
    *,
    weights: str,
) -> torch.Tensor:
    if weights == "uniform":
        return neighbor_targets.mean(dim=1)

    inv_dist = neighbor_distances.clamp_min(1e-8).reciprocal()
    weight_sum = inv_dist.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (neighbor_targets * inv_dist.unsqueeze(-1)).sum(dim=1) / weight_sum


def _knn_predict_torch(
    *,
    train_x: torch.Tensor,
    train_y_n: torch.Tensor,
    query_x: torch.Tensor,
    n_neighbors: int,
    weights: str,
    metric: str,
    query_batch_size: int,
) -> torch.Tensor:
    n_neighbors = min(int(n_neighbors), int(train_x.shape[0]))
    predictions: list[torch.Tensor] = []

    if metric == "cosine":
        train_metric = F.normalize(train_x, dim=1)
    else:
        train_metric = train_x

    with torch.inference_mode():
        for start in range(0, int(query_x.shape[0]), max(1, query_batch_size)):
            stop = start + max(1, query_batch_size)
            query_batch = query_x[start:stop]
            if metric == "cosine":
                query_metric = F.normalize(query_batch, dim=1)
                similarity = query_metric @ train_metric.T
                top_similarity, top_indices = torch.topk(similarity, k=n_neighbors, dim=1, largest=True, sorted=False)
                neighbor_distances = (1.0 - top_similarity).clamp_min(0.0)
            else:
                p = 2.0 if metric == "euclidean" else 1.0
                distance = torch.cdist(query_batch, train_metric, p=p)
                neighbor_distances, top_indices = torch.topk(distance, k=n_neighbors, dim=1, largest=False, sorted=False)

            neighbor_targets = train_y_n[top_indices]
            predictions.append(
                _gather_predictions(
                    neighbor_targets,
                    neighbor_distances,
                    weights=weights,
                )
            )

    return torch.cat(predictions, dim=0)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    device = choose_device(args.device)
    backend = _resolve_backend(args.backend, device)

    train_x_raw, train_y = _load_split(args.train_file)
    valid_x_raw, valid_y = _load_split(args.valid_file)
    test_x_raw, test_y = _load_split(args.test_file)

    label_norm = LabelNormalizer.fit(train_y)
    train_y_n = label_norm.transform(train_y)
    valid_y_n = label_norm.transform(valid_y)
    test_y_n = label_norm.transform(test_y)

    best: dict[str, object] | None = None
    search_results: list[dict[str, object]] = []

    for feature_norm in args.feature_norm:
        train_x, valid_x, test_x, feature_stats = normalize_feature_splits(
            train_x_raw, valid_x_raw, test_x_raw, feature_norm
        )

        if backend == "torch":
            train_x_t = torch.from_numpy(train_x).to(device)
            valid_x_t = torch.from_numpy(valid_x).to(device)
            test_x_t = torch.from_numpy(test_x).to(device)
            train_y_n_t = torch.from_numpy(train_y_n).to(device)

        for n_neighbors in args.neighbors:
            for weights in args.weights:
                for metric in args.metric:
                    if backend == "torch":
                        valid_pred_n_t = _knn_predict_torch(
                            train_x=train_x_t,
                            train_y_n=train_y_n_t,
                            query_x=valid_x_t,
                            n_neighbors=n_neighbors,
                            weights=weights,
                            metric=metric,
                            query_batch_size=args.query_batch_size,
                        )
                        valid_pred_n = valid_pred_n_t.cpu().numpy().astype(np.float32)
                    else:
                        valid_pred_n = _knn_predict_sklearn(
                            train_x=train_x,
                            train_y_n=train_y_n,
                            query_x=valid_x,
                            n_neighbors=n_neighbors,
                            weights=weights,
                            metric=metric,
                        )

                    valid_pred = label_norm.inverse_transform(valid_pred_n)
                    valid_report = mse_report(valid_pred, valid_y)
                    result = {
                        "feature_norm": feature_norm,
                        "n_neighbors": int(min(n_neighbors, train_x.shape[0])),
                        "weights": weights,
                        "metric": metric,
                        "backend": backend,
                        "valid": valid_report,
                        "valid_normalized": mse_report(valid_pred_n, valid_y_n),
                    }
                    search_results.append(result)
                    print(result, flush=True)

                    if best is None or valid_report["mean_mse"] < best["valid"]["mean_mse"]:
                        if backend == "torch":
                            test_pred_n_t = _knn_predict_torch(
                                train_x=train_x_t,
                                train_y_n=train_y_n_t,
                                query_x=test_x_t,
                                n_neighbors=n_neighbors,
                                weights=weights,
                                metric=metric,
                                query_batch_size=args.query_batch_size,
                            )
                            test_pred_n = test_pred_n_t.cpu().numpy().astype(np.float32)
                        else:
                            test_pred_n = _knn_predict_sklearn(
                                train_x=train_x,
                                train_y_n=train_y_n,
                                query_x=test_x,
                                n_neighbors=n_neighbors,
                                weights=weights,
                                metric=metric,
                            )

                        test_pred = label_norm.inverse_transform(test_pred_n)
                        best = {
                            "feature_norm": feature_norm,
                            "n_neighbors": int(min(n_neighbors, train_x.shape[0])),
                            "weights": weights,
                            "metric": metric,
                            "backend": backend,
                            "valid": valid_report,
                            "valid_normalized": mse_report(valid_pred_n, valid_y_n),
                            "test": mse_report(test_pred, test_y),
                            "test_normalized": mse_report(test_pred_n, test_y_n),
                            "feature_stats": feature_stats,
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
    feature_stats = best.pop("feature_stats")
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
