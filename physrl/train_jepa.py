from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ActiveMatterWindowDataset
from .losses import vicreg_loss
from .models import JepaModel
from .utils import atomic_torch_save, choose_device, ensure_dir, parse_int_list, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a JEPA encoder on active_matter from scratch.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path containing train/valid/test or data/train/valid/test.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/jepa"), help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-frames", type=int, default=16)
    parser.add_argument("--target-frames", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--valid-stride", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--dims", type=str, default="16,32,64,128,128")
    parser.add_argument("--num-res-blocks", type=str, default="3,3,3,9,3")
    parser.add_argument("--sim-coeff", type=float, default=2.0)
    parser.add_argument("--std-coeff", type=float, default=40.0)
    parser.add_argument("--cov-coeff", type=float, default=2.0)
    parser.add_argument("--loss-chunks", type=int, default=5)
    parser.add_argument("--loss-groups", type=int, default=1)
    parser.add_argument("--loss-fp32-stats", action="store_true")
    parser.add_argument("--loss-zscore-for-cov", action="store_true")
    parser.add_argument("--loss-adaptive-cov-scale", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a full training checkpoint path, or use out-dir/last.pt with 'auto'.",
    )
    parser.add_argument("--save-every", type=int, default=1, help="Write numbered full checkpoints every N epochs.")
    return parser.parse_args()


def _build_loader(
    dataset: ActiveMatterWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"loss": float("nan")}
    keys = metrics[0].keys()
    return {key: sum(metric[key] for metric in metrics) / len(metrics) for key in keys}


def _resolve_resume_path(out_dir: Path, resume: str | None) -> Path | None:
    if resume is None:
        return None
    if resume == "auto":
        return out_dir / "last.pt"
    return Path(resume).expanduser().resolve()


def main() -> None:
    args = parse_args()
    dims = parse_int_list(args.dims)
    blocks = parse_int_list(args.num_res_blocks)
    if len(dims) != len(blocks):
        raise SystemExit("--dims and --num-res-blocks must have the same number of entries")

    seed_everything(args.seed)
    device = choose_device(args.device)

    out_dir = ensure_dir(args.out_dir)
    train_dataset = ActiveMatterWindowDataset(
        root=args.data_root,
        split="train",
        context_frames=args.context_frames,
        target_frames=args.target_frames,
        stride=args.train_stride,
        resolution=args.resolution,
        max_samples=args.max_train_samples,
    )
    valid_dataset = ActiveMatterWindowDataset(
        root=args.data_root,
        split="valid",
        context_frames=args.context_frames,
        target_frames=args.target_frames,
        stride=args.valid_stride,
        resolution=args.resolution,
        max_samples=args.max_valid_samples,
    )

    train_loader = _build_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_loader = _build_loader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = JepaModel(
        in_chans=sum(field.channels for field in train_dataset.field_specs),
        dims=dims,
        num_res_blocks=blocks,
        num_frames=args.context_frames,
    ).to(device)
    config_payload = vars(args).copy()
    config_payload["dims"] = dims
    config_payload["num_res_blocks"] = blocks
    config_payload["in_chans"] = sum(field.channels for field in train_dataset.field_specs)
    save_json(out_dir / "train_config.json", config_payload)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    start_epoch = 1

    resume_path = _resolve_resume_path(out_dir, args.resume)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location="cpu")
        if "encoder" not in payload or "predictor" not in payload:
            raise ValueError(f"resume checkpoint must be a full training checkpoint: {resume_path}")
        model.encoder.load_state_dict(payload["encoder"])
        model.predictor.load_state_dict(payload["predictor"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        best_val = float(payload.get("best_valid_loss", best_val))
        history = list(payload.get("history", []))
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(
            {
                "status": "resumed",
                "resume_path": str(resume_path),
                "next_epoch": start_epoch,
                "best_valid_loss": best_val,
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
        start_time = time.time()
        model.train()
        train_metrics: list[dict[str, float]] = []

        for batch in train_loader:
            context = batch["context"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                pred_latent, target_latent = model(context, target)
                loss_dict = vicreg_loss(
                    pred_latent,
                    target_latent,
                    sim_coeff=args.sim_coeff,
                    std_coeff=args.std_coeff,
                    cov_coeff=args.cov_coeff,
                    n_chunks=args.loss_chunks,
                    num_groups=args.loss_groups,
                    fp32_stats=args.loss_fp32_stats,
                    zscore_for_cov=args.loss_zscore_for_cov,
                    adaptive_cov_scale=args.loss_adaptive_cov_scale,
                )
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_metrics.append({key: float(value.item()) for key, value in loss_dict.items()})

        model.eval()
        valid_metrics: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in valid_loader:
                context = batch["context"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                pred_latent, target_latent = model(context, target)
                loss_dict = vicreg_loss(
                    pred_latent,
                    target_latent,
                    sim_coeff=args.sim_coeff,
                    std_coeff=args.std_coeff,
                    cov_coeff=args.cov_coeff,
                    n_chunks=args.loss_chunks,
                    num_groups=args.loss_groups,
                    fp32_stats=args.loss_fp32_stats,
                    zscore_for_cov=args.loss_zscore_for_cov,
                    adaptive_cov_scale=args.loss_adaptive_cov_scale,
                )
                valid_metrics.append({key: float(value.item()) for key, value in loss_dict.items()})

        train_summary = _mean_metrics(train_metrics)
        valid_summary = _mean_metrics(valid_metrics)
        elapsed = time.time() - start_time
        epoch_record = {
            "epoch": epoch,
            "seconds": round(elapsed, 2),
            "train_loss": train_summary["loss"],
            "train_repr_loss": train_summary["repr_loss"],
            "train_std_loss": train_summary["std_loss"],
            "train_cov_loss": train_summary["cov_loss"],
            "train_std_loss_pred": train_summary["std_loss_pred"],
            "train_std_loss_target": train_summary["std_loss_target"],
            "train_cov_loss_pred": train_summary["cov_loss_pred"],
            "train_cov_loss_target": train_summary["cov_loss_target"],
            "valid_loss": valid_summary["loss"],
            "valid_repr_loss": valid_summary["repr_loss"],
            "valid_std_loss": valid_summary["std_loss"],
            "valid_cov_loss": valid_summary["cov_loss"],
            "valid_std_loss_pred": valid_summary["std_loss_pred"],
            "valid_std_loss_target": valid_summary["std_loss_target"],
            "valid_cov_loss_pred": valid_summary["cov_loss_pred"],
            "valid_cov_loss_target": valid_summary["cov_loss_target"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        print(epoch_record, flush=True)

        improved = valid_summary["loss"] < best_val
        if improved:
            best_val = valid_summary["loss"]

        checkpoint = {
            "epoch": epoch,
            "config": config_payload,
            "encoder": model.encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_valid_loss": best_val,
            "history": history,
        }
        atomic_torch_save(out_dir / "last.pt", checkpoint)
        atomic_torch_save(
            out_dir / "encoder_last.pt",
            {
                "epoch": epoch,
                "config": config_payload,
                "state_dict": model.encoder.state_dict(),
                "best_valid_loss": best_val,
            },
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            atomic_torch_save(out_dir / f"epoch_{epoch:04d}.pt", checkpoint)

        if improved:
            atomic_torch_save(out_dir / "best.pt", checkpoint)
            atomic_torch_save(
                out_dir / "encoder_best.pt",
                {
                    "epoch": epoch,
                    "config": config_payload,
                    "state_dict": model.encoder.state_dict(),
                    "best_valid_loss": best_val,
                },
            )

        save_json(out_dir / "history.json", {"epochs": history, "best_valid_loss": best_val})


if __name__ == "__main__":
    main()
