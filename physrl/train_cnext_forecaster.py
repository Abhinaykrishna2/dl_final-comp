from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, Sampler

from .data import ActiveMatterWindowDataset
from .models import CNextUNetForecaster
from .train_jepa import (
    DistState,
    StridedShardSampler,
    WandbState,
    _build_loader,
    _count_params,
    _init_dist_state,
    _init_wandb,
    _resolve_resume_path,
    _unwrap_model,
)
from .utils import (
    atomic_torch_save,
    configure_torch_runtime,
    ensure_dir,
    load_torch_checkpoint,
    save_json,
    seed_everything,
)


METRIC_KEYS = ("loss", "mse", "relative_mse", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scratch CNext-U-Net forecaster on active_matter, then freeze its encoder for probing."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/cnext_unet96"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-start-factor", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--target-frames", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--valid-stride", type=int, default=1)
    parser.add_argument(
        "--train-index-mode",
        choices=["single_clip", "sliding_window"],
        default="sliding_window",
    )
    parser.add_argument(
        "--valid-index-mode",
        choices=["single_clip", "sliding_window"],
        default="single_clip",
    )
    parser.add_argument(
        "--clip-selection",
        choices=["start", "center", "end"],
        default="center",
    )
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--init-features", type=int, default=42)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--blocks-at-neck", type=int, default=1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--layer-scale-init-value", type=float, default=1e-6)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a full CNext-U-Net checkpoint path, or use out-dir/last.pt with 'auto'.",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dist-backend", choices=["nccl", "gloo"], default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-entity", type=str, default="abhinaykrishna60-new-york-university")
    parser.add_argument("--wandb-project", type=str, default="dl_final-comp")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def _empty_metric_sums() -> dict[str, float]:
    return {key: 0.0 for key in METRIC_KEYS}


def _forecast_loss(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_f = pred.float()
    target_f = target.float()
    mse = F.mse_loss(pred_f, target_f)
    relative_mse = mse / target_f.square().mean().clamp_min(1e-8)
    mae = F.l1_loss(pred_f, target_f)
    return {
        "loss": mse,
        "mse": mse.detach(),
        "relative_mse": relative_mse.detach(),
        "mae": mae.detach(),
    }


def _update_metric_sums(metric_sums: dict[str, float], loss_dict: dict[str, torch.Tensor], weight: int) -> None:
    for key in METRIC_KEYS:
        value = loss_dict.get(key)
        if value is not None:
            metric_sums[key] += float(value.item()) * weight


def _reduce_metric_summaries(
    metric_sums: dict[str, float],
    sample_count: int,
    *,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    payload = torch.tensor(
        [float(sample_count)] + [metric_sums[key] for key in METRIC_KEYS],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    total_count = payload[0].item()
    if total_count <= 0:
        return {key: float("nan") for key in METRIC_KEYS}
    return {
        key: float(payload[idx + 1].item() / total_count)
        for idx, key in enumerate(METRIC_KEYS)
    }


def _load_init_checkpoint(model: CNextUNetForecaster, checkpoint_path: Path) -> dict[str, Any]:
    payload = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = payload["model"] if "model" in payload else payload["state_dict"]
    model.load_state_dict(state_dict)
    return {
        "path": str(checkpoint_path),
        "epoch": payload.get("epoch"),
        "source_best_valid_loss": payload.get("best_valid_loss"),
    }


def _save_encoder_checkpoint(
    path: Path,
    *,
    model: CNextUNetForecaster,
    epoch: int,
    config: dict[str, Any],
    best_valid_loss: float,
) -> None:
    atomic_torch_save(
        path,
        {
            "epoch": epoch,
            "config": config,
            "model_type": "cnext_unet_forecaster",
            "state_dict": model.state_dict(),
            "best_valid_loss": best_valid_loss,
        },
    )


def main() -> None:
    dist_state: DistState | None = None
    wandb_state = WandbState(enabled=False)
    try:
        args = parse_args()
        if args.context_frames < 1:
            raise SystemExit("--context-frames must be >= 1")
        if args.target_frames < 1:
            raise SystemExit("--target-frames must be >= 1")
        if args.train_index_mode != "sliding_window" and args.train_stride != 1:
            raise SystemExit("--train-stride only applies when --train-index-mode sliding_window")
        if args.valid_index_mode != "sliding_window" and args.valid_stride != 1:
            raise SystemExit("--valid-stride only applies when --valid-index-mode sliding_window")
        if args.warmup_epochs < 0:
            raise SystemExit("--warmup-epochs must be non-negative")
        if not (0.0 < args.warmup_start_factor <= 1.0):
            raise SystemExit("--warmup-start-factor must be in (0, 1]")

        dist_state = _init_dist_state(args.device, args.dist_backend)
        seed_everything(args.seed)
        configure_torch_runtime(deterministic=args.deterministic)

        out_dir = ensure_dir(args.out_dir)
        train_dataset = ActiveMatterWindowDataset(
            root=args.data_root,
            split="train",
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            stride=args.train_stride,
            resolution=args.resolution,
            max_samples=args.max_train_samples,
            index_mode=args.train_index_mode,
            clip_selection=args.clip_selection,
        )
        valid_dataset = ActiveMatterWindowDataset(
            root=args.data_root,
            split="valid",
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            stride=args.valid_stride,
            resolution=args.resolution,
            max_samples=args.max_valid_samples,
            index_mode=args.valid_index_mode,
            clip_selection=args.clip_selection,
        )

        train_sampler: Sampler[int] | None = None
        valid_sampler: Sampler[int] | None = None
        if dist_state.enabled:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=dist_state.world_size,
                rank=dist_state.rank,
                shuffle=True,
                drop_last=False,
            )
            valid_sampler = StridedShardSampler(
                len(valid_dataset),
                rank=dist_state.rank,
                world_size=dist_state.world_size,
            )

        train_loader = _build_loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle=True,
            sampler=train_sampler,
            seed=args.seed + dist_state.rank,
        )
        valid_loader = _build_loader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle=False,
            sampler=valid_sampler,
            seed=args.seed + 10_000 + dist_state.rank,
        )

        model = CNextUNetForecaster(
            field_channels=sum(field.channels for field in train_dataset.field_specs),
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            init_features=args.init_features,
            stages=args.stages,
            blocks_per_stage=args.blocks_per_stage,
            blocks_at_neck=args.blocks_at_neck,
            drop_path_rate=args.drop_path_rate,
            layer_scale_init_value=args.layer_scale_init_value,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(dist_state.device)

        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = min(max(0, int(args.warmup_epochs * len(train_loader))), max(total_steps - 1, 0))

        config_payload = vars(args).copy()
        config_payload["model_type"] = "cnext_unet_forecaster"
        config_payload["architecture_reference"] = "PolymathicAI The Well UNetConvNext, trained from scratch"
        config_payload["in_chans"] = sum(field.channels for field in train_dataset.field_specs)
        config_payload["field_channels"] = sum(field.channels for field in train_dataset.field_specs)
        config_payload["distributed"] = dist_state.enabled
        config_payload["world_size"] = dist_state.world_size
        config_payload["backend"] = dist_state.backend
        config_payload["out_dir"] = str(out_dir)
        config_payload["train_samples"] = len(train_dataset)
        config_payload["valid_samples"] = len(valid_dataset)
        config_payload["train_batches"] = len(train_loader)
        config_payload["valid_batches"] = len(valid_loader)
        config_payload["global_batch_size"] = args.batch_size * dist_state.world_size
        config_payload["total_steps"] = total_steps
        config_payload["warmup_steps"] = warmup_steps
        config_payload["total_params"] = _count_params(model)
        config_payload["encoder_params"] = _count_params(model.in_proj) + _count_params(model.encoder) + _count_params(model.neck)
        config_payload["embed_dim"] = model.embed_dim

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=args.min_lr,
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=args.min_lr,
            )

        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(
            enabled=args.amp and dist_state.device.type == "cuda" and amp_dtype == torch.float16
        )

        best_val = float("inf")
        history: list[dict[str, float | int]] = []
        start_epoch = 1
        resume_payload: dict[str, Any] | None = None

        resume_path = _resolve_resume_path(out_dir, args.resume)
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
            resume_payload = load_torch_checkpoint(resume_path, map_location="cpu")
            model.load_state_dict(resume_payload["model"])
            if "optimizer" in resume_payload:
                optimizer.load_state_dict(resume_payload["optimizer"])
            if "scheduler" in resume_payload:
                scheduler.load_state_dict(resume_payload["scheduler"])
            if "scaler" in resume_payload:
                scaler.load_state_dict(resume_payload["scaler"])
            best_val = float(resume_payload.get("best_valid_loss", best_val))
            history = list(resume_payload.get("history", []))
            start_epoch = int(resume_payload.get("epoch", 0)) + 1
            if dist_state.is_main_process:
                print(
                    {
                        "status": "resumed",
                        "resume_path": str(resume_path),
                        "next_epoch": start_epoch,
                        "best_valid_loss": best_val,
                    },
                    flush=True,
                )
        elif args.init_checkpoint is not None:
            init_path = args.init_checkpoint.expanduser().resolve()
            if not init_path.exists():
                raise FileNotFoundError(f"init checkpoint not found: {init_path}")
            init_summary = _load_init_checkpoint(model, init_path)
            config_payload["init_checkpoint"] = init_summary["path"]
            config_payload["init_checkpoint_epoch"] = init_summary["epoch"]
            config_payload["init_source_best_valid_loss"] = init_summary["source_best_valid_loss"]
            if dist_state.is_main_process:
                print({"status": "initialized", **init_summary}, flush=True)

        if dist_state.is_main_process:
            save_json(out_dir / "train_config.json", config_payload)

        if start_epoch > args.epochs:
            if dist_state.is_main_process:
                print(
                    {
                        "status": "already_complete",
                        "requested_epochs": args.epochs,
                        "checkpoint_epoch": start_epoch - 1,
                    },
                    flush=True,
                )
            return

        wandb_state = _init_wandb(
            args=args,
            dist_state=dist_state,
            config_payload=config_payload,
            out_dir=out_dir,
            resume_payload=resume_payload,
        )

        if dist_state.enabled:
            if dist_state.device.type == "cuda":
                model = DDP(
                    model,
                    device_ids=[dist_state.device.index],
                    output_device=dist_state.device.index,
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                    static_graph=True,
                )
            else:
                model = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)

        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            model.train()
            train_metric_sums = _empty_metric_sums()
            train_sample_count = 0
            for batch in train_loader:
                context = batch["context"].to(dist_state.device, non_blocking=True)
                target = batch["target"].to(dist_state.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=dist_state.device.type,
                    dtype=amp_dtype,
                    enabled=args.amp and dist_state.device.type == "cuda",
                ):
                    pred = model(context)
                    loss_dict = _forecast_loss(pred, target)
                    loss = loss_dict["loss"]

                prev_scale = scaler.get_scale() if scaler.is_enabled() else None
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if not scaler.is_enabled() or scaler.get_scale() >= (prev_scale or 0.0):
                    scheduler.step()

                batch_size = int(context.shape[0])
                train_sample_count += batch_size
                _update_metric_sums(train_metric_sums, loss_dict, batch_size)

            train_summary = _reduce_metric_summaries(
                train_metric_sums,
                train_sample_count,
                device=dist_state.device,
                distributed=dist_state.enabled,
            )

            model.eval()
            valid_metric_sums = _empty_metric_sums()
            valid_sample_count = 0
            with torch.no_grad():
                for batch in valid_loader:
                    context = batch["context"].to(dist_state.device, non_blocking=True)
                    target = batch["target"].to(dist_state.device, non_blocking=True)
                    with torch.autocast(
                        device_type=dist_state.device.type,
                        dtype=amp_dtype,
                        enabled=args.amp and dist_state.device.type == "cuda",
                    ):
                        pred = model(context)
                        loss_dict = _forecast_loss(pred, target)
                    batch_size = int(context.shape[0])
                    valid_sample_count += batch_size
                    _update_metric_sums(valid_metric_sums, loss_dict, batch_size)

            valid_summary = _reduce_metric_summaries(
                valid_metric_sums,
                valid_sample_count,
                device=dist_state.device,
                distributed=dist_state.enabled,
            )
            elapsed = time.time() - start_time
            epoch_record = {
                "epoch": epoch,
                "seconds": round(elapsed, 2),
                "train_loss": train_summary["loss"],
                "train_mse": train_summary["mse"],
                "train_relative_mse": train_summary["relative_mse"],
                "train_mae": train_summary["mae"],
                "valid_loss": valid_summary["loss"],
                "valid_mse": valid_summary["mse"],
                "valid_relative_mse": valid_summary["relative_mse"],
                "valid_mae": valid_summary["mae"],
                "lr": optimizer.param_groups[0]["lr"],
                "world_size": dist_state.world_size,
            }
            history.append(epoch_record)
            if dist_state.is_main_process:
                print(epoch_record, flush=True)
                if wandb_state.enabled and wandb_state.run is not None:
                    wandb_state.run.log(epoch_record, step=epoch)

            improved = valid_summary["loss"] < best_val
            if improved:
                best_val = valid_summary["loss"]

            if dist_state.is_main_process:
                raw_model = _unwrap_model(model)
                checkpoint = {
                    "epoch": epoch,
                    "config": config_payload,
                    "model_type": "cnext_unet_forecaster",
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_valid_loss": best_val,
                    "history": history,
                    "wandb_run_id": wandb_state.run_id,
                }
                atomic_torch_save(out_dir / "last.pt", checkpoint)
                _save_encoder_checkpoint(
                    out_dir / "encoder_last.pt",
                    model=raw_model,
                    epoch=epoch,
                    config=config_payload,
                    best_valid_loss=best_val,
                )
                if args.save_every > 0 and epoch % args.save_every == 0:
                    atomic_torch_save(out_dir / f"epoch_{epoch:04d}.pt", checkpoint)

                if improved:
                    atomic_torch_save(out_dir / "best.pt", checkpoint)
                    _save_encoder_checkpoint(
                        out_dir / "encoder_best.pt",
                        model=raw_model,
                        epoch=epoch,
                        config=config_payload,
                        best_valid_loss=best_val,
                    )

                save_json(out_dir / "history.json", {"epochs": history, "best_valid_loss": best_val})
                if wandb_state.enabled and wandb_state.run is not None:
                    wandb_state.run.summary["best_valid_loss"] = best_val
                    wandb_state.run.summary["last_epoch"] = epoch

            if dist_state.enabled:
                dist.barrier()
    finally:
        if wandb_state.enabled and wandb_state.run is not None:
            wandb_state.run.finish()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
