from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from .utils import (
    LabelNormalizer,
    add_wandb_args,
    atomic_torch_save,
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    flatten_metrics,
    init_wandb_run,
    log_wandb_artifact,
    mse_report,
    normalize_feature_splits,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep linear-probe hyperparameters on frozen embeddings.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/linear_probe_sweep"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic PyTorch behavior where feasible.")
    parser.add_argument(
        "--feature-norms",
        nargs="+",
        choices=["none", "zscore", "l2", "zscore_l2"],
        default=["zscore", "zscore_l2", "l2"],
    )
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-5, 1e-4, 1e-3])
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[0, 1024, 4096],
        help="Use 0 for full-batch training.",
    )
    parser.add_argument(
        "--cache-on-device",
        choices=["auto", "always", "never"],
        default="auto",
        help="Move normalized embedding splits to the target device once per feature_norm sweep.",
    )
    add_wandb_args(parser)
    return parser.parse_args()


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["embeddings"].astype(np.float32), payload["labels"].astype(np.float32)


def _slugify_float(value: float) -> str:
    text = f"{value:g}"
    text = text.replace("-", "neg")
    return text.replace(".", "p")


def _slugify_batch_size(value: int) -> str:
    return "full" if value <= 0 else str(value)


def _should_cache_on_device(
    *,
    cache_on_device: str,
    device: torch.device,
    arrays: list[np.ndarray],
) -> bool:
    if cache_on_device == "always":
        return True
    if cache_on_device == "never":
        return False
    if device.type != "cuda":
        return False

    total_bytes = sum(int(array.nbytes) for array in arrays)
    free_bytes, _ = torch.cuda.mem_get_info(device=device)
    return total_bytes < int(0.7 * free_bytes)


def _predict_tensor(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(features)


def _build_linear_model(in_dim: int, out_dim: int, device: torch.device) -> torch.nn.Linear:
    return torch.nn.Linear(int(in_dim), int(out_dim)).to(device)


def _evaluate_tensor(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_raw: np.ndarray,
    y_norm: np.ndarray,
    label_norm: LabelNormalizer,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    x = x.to(next(model.parameters()).device)
    pred_n_t = _predict_tensor(model, x)
    pred_n = pred_n_t.detach().cpu().numpy().astype(np.float32)
    pred_raw = label_norm.inverse_transform(pred_n)
    return (
        mse_report(pred_raw, y_raw),
        mse_report(pred_n, y_norm),
        pred_raw,
        pred_n,
    )


def _train_one(
    *,
    train_x: torch.Tensor,
    train_y_n: torch.Tensor,
    valid_x: torch.Tensor,
    valid_y: np.ndarray,
    valid_y_n: np.ndarray,
    feature_stats: dict,
    label_norm: LabelNormalizer,
    device: torch.device,
    epochs: int,
    patience: int,
    min_lr: float,
    grad_clip: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = _build_linear_model(int(train_x.shape[1]), int(train_y_n.shape[1]), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    loss_fn = torch.nn.MSELoss()

    train_count = int(train_x.shape[0])
    effective_batch = train_count if batch_size <= 0 else min(batch_size, train_count)
    cached_on_device = train_x.device == device and train_y_n.device == device

    best_valid = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        permutation = torch.randperm(train_count, device=device if cached_on_device else torch.device("cpu"))

        for start in range(0, train_count, effective_batch):
            index = permutation[start : start + effective_batch]
            batch_x = train_x[index]
            batch_y = train_y_n[index]
            if not cached_on_device:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            current_batch = int(batch_x.shape[0])
            running += float(loss.item()) * current_batch
            seen += current_batch

        scheduler.step()
        train_loss = running / max(seen, 1)
        valid_report, valid_report_normalized, _, _ = _evaluate_tensor(
            model,
            valid_x,
            valid_y,
            valid_y_n,
            label_norm,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_mean_mse": valid_report["mean_mse"],
            "valid_alpha_mse": valid_report["alpha_mse"],
            "valid_zeta_mse": valid_report["zeta_mse"],
            "valid_mean_mse_normalized": valid_report_normalized["mean_mse"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        if valid_report_normalized["mean_mse"] < best_valid:
            best_valid = valid_report_normalized["mean_mse"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("linear-probe sweep produced no checkpoint")

    model.load_state_dict(best_state)
    valid_report, valid_report_normalized, valid_pred, valid_pred_n = _evaluate_tensor(
        model,
        valid_x,
        valid_y,
        valid_y_n,
        label_norm,
    )
    return {
        "state_dict": best_state,
        "history": history,
        "best_valid_mean_mse_normalized": best_valid,
        "valid": valid_report,
        "valid_normalized": valid_report_normalized,
        "valid_pred": valid_pred,
        "valid_pred_normalized": valid_pred_n,
        "feature_stats": feature_stats,
        "label_stats": label_norm.to_dict(),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_torch_runtime(deterministic=args.deterministic)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    train_x_raw, train_y = _load_split(args.train_file)
    valid_x_raw, valid_y = _load_split(args.valid_file)
    test_x_raw, test_y = _load_split(args.test_file)
    label_norm = LabelNormalizer.fit(train_y)
    train_y_n = label_norm.transform(train_y)
    valid_y_n = label_norm.transform(valid_y)
    test_y_n = label_norm.transform(test_y)
    wandb_run = init_wandb_run(
        mode=args.wandb_mode,
        entity=args.wandb_entity,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        out_dir=out_dir,
        config={
            **vars(args),
            "train_shape": list(train_x_raw.shape),
            "valid_shape": list(valid_x_raw.shape),
            "test_shape": list(test_x_raw.shape),
            "label_stats": label_norm.to_dict(),
            "selection_metric": "valid_normalized.mean_mse",
        },
        job_type="linear-probe-sweep",
    )

    best_result: dict[str, object] | None = None
    search_results: list[dict[str, object]] = []
    trial_index = 0

    for feature_norm in args.feature_norms:
        train_x, valid_x, test_x, feature_stats = normalize_feature_splits(
            train_x_raw, valid_x_raw, test_x_raw, feature_norm
        )

        cache_on_device = _should_cache_on_device(
            cache_on_device=args.cache_on_device,
            device=device,
            arrays=[train_x, valid_x, test_x, train_y_n],
        )
        if cache_on_device:
            train_x_t = torch.from_numpy(train_x).to(device)
            valid_x_t = torch.from_numpy(valid_x).to(device)
            test_x_t = torch.from_numpy(test_x).to(device)
            train_y_n_t = torch.from_numpy(train_y_n).to(device)
        else:
            train_x_t = torch.from_numpy(train_x)
            valid_x_t = torch.from_numpy(valid_x)
            test_x_t = torch.from_numpy(test_x)
            train_y_n_t = torch.from_numpy(train_y_n)

        for lr, weight_decay, batch_size in itertools.product(args.lrs, args.weight_decays, args.batch_sizes):
            trial_seed = args.seed + trial_index
            result = _train_one(
                train_x=train_x_t,
                train_y_n=train_y_n_t,
                valid_x=valid_x_t,
                valid_y=valid_y,
                valid_y_n=valid_y_n,
                feature_stats=feature_stats,
                label_norm=label_norm,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                min_lr=args.min_lr,
                grad_clip=args.grad_clip,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                seed=trial_seed,
            )
            trial_record = {
                "trial_index": trial_index,
                "feature_norm": feature_norm,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "cache_on_device": cache_on_device,
                "valid": result["valid"],
                "valid_normalized": result["valid_normalized"],
                "epochs_trained": len(result["history"]),
            }
            search_results.append(trial_record)
            print(trial_record, flush=True)
            if wandb_run is not None:
                wandb_run.log(flatten_metrics(trial_record, prefix="linear_probe_sweep/trial"), step=trial_index)

            if (
                best_result is None
                or result["valid_normalized"]["mean_mse"] < best_result["valid_normalized"]["mean_mse"]
            ):
                best_result = {
                    "trial_index": trial_index,
                    "feature_norm": feature_norm,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    **result,
                }

            trial_index += 1

    if best_result is None:
        raise RuntimeError("linear-probe sweep produced no result")

    best_train_x, best_valid_x, best_test_x, best_feature_stats = normalize_feature_splits(
        train_x_raw,
        valid_x_raw,
        test_x_raw,
        str(best_result["feature_norm"]),
    )
    best_cache_on_device = _should_cache_on_device(
        cache_on_device=args.cache_on_device,
        device=device,
        arrays=[best_train_x, best_valid_x, best_test_x, train_y_n],
    )
    if best_cache_on_device:
        best_test_x_t = torch.from_numpy(best_test_x).to(device)
    else:
        best_test_x_t = torch.from_numpy(best_test_x)

    best_model = _build_linear_model(best_train_x.shape[1], train_y_n.shape[1], device)
    best_model.load_state_dict(best_result["state_dict"])
    test_report, test_report_normalized, test_pred, test_pred_n = _evaluate_tensor(
        best_model,
        best_test_x_t,
        test_y,
        test_y_n,
        label_norm,
    )
    best_result["test"] = test_report
    best_result["test_normalized"] = test_report_normalized
    best_result["test_pred"] = test_pred
    best_result["test_pred_normalized"] = test_pred_n
    best_result["feature_stats"] = best_feature_stats

    best_slug = (
        f"norm-{best_result['feature_norm']}"
        f"_lr-{_slugify_float(float(best_result['lr']))}"
        f"_wd-{_slugify_float(float(best_result['weight_decay']))}"
        f"_bs-{_slugify_batch_size(int(best_result['batch_size']))}"
    )

    atomic_torch_save(
        out_dir / "best_linear_probe.pt",
        {
            "state_dict": best_result["state_dict"],
            "feature_stats": best_result["feature_stats"],
            "label_stats": best_result["label_stats"],
            "best_valid_mean_mse": best_result["valid"]["mean_mse"],
            "best_valid_mean_mse_normalized": best_result["best_valid_mean_mse_normalized"],
            "selection_metric": "valid_normalized.mean_mse",
            "feature_norm": best_result["feature_norm"],
            "lr": best_result["lr"],
            "weight_decay": best_result["weight_decay"],
            "batch_size": best_result["batch_size"],
            "train_file": str(args.train_file),
            "valid_file": str(args.valid_file),
            "test_file": str(args.test_file),
        },
    )
    np.savez_compressed(
        out_dir / "valid_predictions.npz",
        pred=np.asarray(best_result["valid_pred"], dtype=np.float32),
        target=valid_y.astype(np.float32),
        pred_normalized=np.asarray(best_result["valid_pred_normalized"], dtype=np.float32),
        target_normalized=valid_y_n.astype(np.float32),
    )
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        pred=np.asarray(best_result["test_pred"], dtype=np.float32),
        target=test_y.astype(np.float32),
        pred_normalized=np.asarray(best_result["test_pred_normalized"], dtype=np.float32),
        target_normalized=test_y_n.astype(np.float32),
    )
    metrics_payload = {
        "best_slug": best_slug,
        "best": {
            "trial_index": best_result["trial_index"],
            "feature_norm": best_result["feature_norm"],
            "lr": best_result["lr"],
            "weight_decay": best_result["weight_decay"],
            "batch_size": best_result["batch_size"],
            "best_valid_mean_mse": best_result["valid"]["mean_mse"],
            "best_valid_mean_mse_normalized": best_result["best_valid_mean_mse_normalized"],
            "selection_metric": "valid_normalized.mean_mse",
            "valid": best_result["valid"],
            "valid_normalized": best_result["valid_normalized"],
            "test": best_result["test"],
            "test_normalized": best_result["test_normalized"],
            "epochs_trained": len(best_result["history"]),
        },
        "search_results": search_results,
        "feature_stats": best_result["feature_stats"],
        "label_stats": best_result["label_stats"],
    }
    save_json(out_dir / "metrics.json", metrics_payload)
    if wandb_run is not None:
        wandb_run.log(flatten_metrics(metrics_payload["best"], prefix="linear_probe_sweep/best"))
        wandb_run.summary["best_slug"] = best_slug
        wandb_run.summary["best_feature_norm"] = str(best_result["feature_norm"])
        wandb_run.summary["best_lr"] = float(best_result["lr"])
        wandb_run.summary["best_weight_decay"] = float(best_result["weight_decay"])
        wandb_run.summary["best_batch_size"] = int(best_result["batch_size"])
        wandb_run.summary["best_valid_mean_mse_normalized"] = float(best_result["best_valid_mean_mse_normalized"])
        wandb_run.summary["test_mean_mse"] = float(best_result["test"]["mean_mse"])
        wandb_run.summary["test_mean_mse_normalized"] = float(best_result["test_normalized"]["mean_mse"])
        log_wandb_artifact(
            wandb_run,
            name=f"linear-probe-sweep-{out_dir.name}",
            artifact_type="linear-probe-sweep",
            paths=[
                out_dir / "best_linear_probe.pt",
                out_dir / "metrics.json",
                out_dir / "valid_predictions.npz",
                out_dir / "test_predictions.npz",
            ],
            metadata={
                "best_slug": best_slug,
                "selection_metric": "valid_normalized.mean_mse",
            },
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
