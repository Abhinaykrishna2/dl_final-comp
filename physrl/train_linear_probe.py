from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import atomic_torch_save, LabelNormalizer, choose_device, ensure_dir, mse_report, normalize_feature_splits, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single linear regression probe on frozen embeddings.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/linear_probe"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-norm", choices=["none", "zscore", "l2", "zscore_l2"], default="zscore")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a linear-probe checkpoint path, or use out-dir/last_linear_probe.pt with 'auto'.",
    )
    parser.add_argument("--save-every", type=int, default=1, help="Write numbered linear-probe checkpoints every N epochs.")
    return parser.parse_args()


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return payload["embeddings"].astype(np.float32), payload["labels"].astype(np.float32)


def _predict(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device)).cpu().numpy()


def _evaluate(
    model: torch.nn.Module,
    x: np.ndarray,
    y_raw: np.ndarray,
    label_norm: LabelNormalizer,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    pred_n = _predict(model, x, device)
    pred_raw = label_norm.inverse_transform(pred_n)
    return mse_report(pred_raw, y_raw), pred_raw, pred_n


def _resolve_resume_path(out_dir: Path, resume: str | None) -> Path | None:
    if resume is None:
        return None
    if resume == "auto":
        return out_dir / "last_linear_probe.pt"
    return Path(resume).expanduser().resolve()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
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

    model = torch.nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y_n)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_valid = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    epochs_without_improvement = 0
    start_epoch = 1

    resume_path = _resolve_resume_path(out_dir, args.resume)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        best_valid = float(payload.get("best_valid_mean_mse", best_valid))
        if "best_state" in payload and payload["best_state"] is not None:
            best_state = payload["best_state"]
        history = list(payload.get("history", []))
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(
            {
                "status": "resumed",
                "resume_path": str(resume_path),
                "next_epoch": start_epoch,
                "best_valid_mean_mse": best_valid,
            },
            flush=True,
        )

    if start_epoch > args.epochs:
        print(
            {
                "status": "already_complete",
                "requested_epochs": args.epochs,
                "checkpoint_epoch": start_epoch - 1,
            },
            flush=True,
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.item()) * batch_x.shape[0]
            seen += batch_x.shape[0]

        scheduler.step()
        train_loss = running / max(seen, 1)
        valid_report, _, valid_pred_n = _evaluate(model, valid_x, valid_y, label_norm, device)
        valid_report_normalized = mse_report(valid_pred_n, valid_y_n)
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
        print(epoch_record, flush=True)

        if valid_report["mean_mse"] < best_valid:
            best_valid = valid_report["mean_mse"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            atomic_torch_save(
                out_dir / "best_linear_probe.pt",
                {
                    "state_dict": best_state,
                    "feature_stats": feature_stats,
                    "label_stats": label_norm.to_dict(),
                    "best_valid_mean_mse": best_valid,
                    "history": history,
                    "train_file": str(args.train_file),
                    "valid_file": str(args.valid_file),
                    "test_file": str(args.test_file),
                },
            )
        else:
            epochs_without_improvement += 1

        last_checkpoint = {
            "epoch": epoch,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "best_state": best_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "feature_stats": feature_stats,
            "label_stats": label_norm.to_dict(),
            "best_valid_mean_mse": best_valid,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "train_file": str(args.train_file),
            "valid_file": str(args.valid_file),
            "test_file": str(args.test_file),
        }
        atomic_torch_save(out_dir / "last_linear_probe.pt", last_checkpoint)
        if args.save_every > 0 and epoch % args.save_every == 0:
            atomic_torch_save(out_dir / f"linear_probe_epoch_{epoch:04d}.pt", last_checkpoint)

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("linear probe training produced no checkpoint")

    model.load_state_dict(best_state)
    valid_report, valid_pred, valid_pred_n = _evaluate(model, valid_x, valid_y, label_norm, device)
    test_report, test_pred, test_pred_n = _evaluate(model, test_x, test_y, label_norm, device)

    atomic_torch_save(
        out_dir / "best_linear_probe.pt",
        {
            "state_dict": best_state,
            "feature_stats": feature_stats,
            "label_stats": label_norm.to_dict(),
            "best_valid_mean_mse": best_valid,
            "train_file": str(args.train_file),
            "valid_file": str(args.valid_file),
            "test_file": str(args.test_file),
        },
    )
    np.savez_compressed(
        out_dir / "valid_predictions.npz",
        pred=valid_pred.astype(np.float32),
        target=valid_y.astype(np.float32),
        pred_normalized=valid_pred_n.astype(np.float32),
        target_normalized=valid_y_n.astype(np.float32),
    )
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        pred=test_pred.astype(np.float32),
        target=test_y.astype(np.float32),
        pred_normalized=test_pred_n.astype(np.float32),
        target_normalized=label_norm.transform(test_y).astype(np.float32),
    )
    save_json(
        out_dir / "metrics.json",
        {
            "best_valid_mean_mse": best_valid,
            "valid": valid_report,
            "test": test_report,
            "valid_normalized": mse_report(valid_pred_n, valid_y_n),
            "test_normalized": mse_report(test_pred_n, label_norm.transform(test_y)),
            "feature_stats": feature_stats,
            "label_stats": label_norm.to_dict(),
            "history": history,
        },
    )


if __name__ == "__main__":
    main()
