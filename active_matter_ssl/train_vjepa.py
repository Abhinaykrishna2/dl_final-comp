from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, Sampler

from .data import ActiveMatterWindowDataset
from .losses import feature_map_std, masked_latent_prediction_loss
from .models import VJepaModel
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
    parse_int_list,
    save_json,
    seed_everything,
)


METRIC_KEYS = (
    "loss",
    "pred_loss",
    "repr_loss",
    "context_std",
    "pred_std",
    "target_std",
    "mask_ratio",
    "ema_momentum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a V-JEPA-style ConvNeXt encoder with spatial masking and an EMA target encoder."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/vjepa_96"))
    parser.add_argument("--epochs", type=int, default=40)
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
    parser.add_argument("--context-frames", type=int, default=16)
    parser.add_argument("--target-frames", type=int, default=16)
    parser.add_argument(
        "--target-mode",
        choices=["future", "same_clip"],
        default="future",
        help="future predicts the next 16 frames; same_clip predicts masked regions from the same clip.",
    )
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
    parser.add_argument("--dims", type=str, default="32,64,128,256,256")
    parser.add_argument("--num-res-blocks", type=str, default="2,2,4,8,2")
    parser.add_argument("--stem-patch-size", type=int, default=1)
    parser.add_argument("--stem-kernel-size", type=int, default=4)
    parser.add_argument("--drop-path-rate", type=float, default=0.05)
    parser.add_argument("--layer-scale-init-value", type=float, default=1e-6)
    parser.add_argument("--mask-ratio", type=float, default=0.55)
    parser.add_argument("--mask-min-block-size", type=int, default=1)
    parser.add_argument("--mask-max-block-size", type=int, default=3)
    parser.add_argument("--mask-max-blocks", type=int, default=8)
    parser.add_argument("--mask-min-keep", type=int, default=1)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--ema-final-momentum", type=float, default=1.0)
    parser.add_argument("--normalize-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-pred", action="store_true")
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
        help="Resume from a full V-JEPA checkpoint path, or use out-dir/last.pt with 'auto'.",
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


def _feature_grid_size(height: int, width: int, stem_patch_size: int, num_stages: int) -> tuple[int, int]:
    downsample = int(stem_patch_size) * (2 ** max(0, int(num_stages) - 1))
    if downsample < 1:
        raise ValueError("downsample factor must be positive")
    return max(1, height // downsample), max(1, width // downsample)


def _sample_block_masks(
    *,
    batch_size: int,
    height: int,
    width: int,
    mask_ratio: float,
    min_block_size: int,
    max_block_size: int,
    max_blocks: int,
    min_keep: int,
    device: torch.device,
) -> torch.Tensor:
    total = int(height * width)
    max_masked = max(1, total - max(0, int(min_keep)))
    target_masked = int(round(float(mask_ratio) * total))
    target_masked = max(1, min(max_masked, target_masked))
    min_block_size = max(1, int(min_block_size))
    max_block_size = max(min_block_size, int(max_block_size))
    max_blocks = max(1, int(max_blocks))

    masks = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.float32)
    for batch_idx in range(batch_size):
        attempts = 0
        while int(masks[batch_idx].sum().item()) < target_masked and attempts < max_blocks * 4:
            attempts += 1
            block_h = int(torch.randint(min_block_size, max_block_size + 1, (1,), device=device).item())
            block_w = int(torch.randint(min_block_size, max_block_size + 1, (1,), device=device).item())
            block_h = min(block_h, height)
            block_w = min(block_w, width)
            top = int(torch.randint(0, height - block_h + 1, (1,), device=device).item())
            left = int(torch.randint(0, width - block_w + 1, (1,), device=device).item())
            masks[batch_idx, :, top : top + block_h, left : left + block_w] = 1.0
            if attempts >= max_blocks and int(masks[batch_idx].sum().item()) >= target_masked:
                break

        current = int(masks[batch_idx].sum().item())
        if current < target_masked:
            flat = masks[batch_idx].view(-1)
            empty = torch.nonzero(flat == 0, as_tuple=False).flatten()
            if empty.numel() > 0:
                fill_count = min(target_masked - current, int(empty.numel()))
                order = torch.randperm(int(empty.numel()), device=device)[:fill_count]
                flat[empty[order]] = 1.0
        elif current > target_masked:
            flat = masks[batch_idx].view(-1)
            filled = torch.nonzero(flat > 0, as_tuple=False).flatten()
            drop_count = min(current - target_masked, int(filled.numel()))
            if drop_count > 0:
                order = torch.randperm(int(filled.numel()), device=device)[:drop_count]
                flat[filled[order]] = 0.0
    return masks


def _ema_momentum_for_step(base: float, final: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(final)
    progress = min(1.0, max(0.0, float(step) / float(total_steps - 1)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final) - (float(final) - float(base)) * cosine


def _load_init_checkpoint(model: VJepaModel, checkpoint_path: Path) -> dict[str, Any]:
    payload = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if "encoder" in payload:
        state_dict = payload["encoder"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        raise ValueError(f"init checkpoint must contain 'encoder' or 'state_dict': {checkpoint_path}")
    model.encoder.load_state_dict(state_dict)
    model.target_encoder.load_state_dict(model.encoder.state_dict())
    return {
        "path": str(checkpoint_path),
        "epoch": payload.get("epoch"),
        "source_best_valid_loss": payload.get("best_valid_loss"),
    }


def _target_from_batch(batch: dict[str, torch.Tensor], context: torch.Tensor, target_mode: str, device: torch.device) -> torch.Tensor:
    if target_mode == "same_clip":
        return context
    return batch["target"].to(device, non_blocking=True)


def main() -> None:
    dist_state: DistState | None = None
    wandb_state = WandbState(enabled=False)
    try:
        args = parse_args()
        dims = parse_int_list(args.dims)
        blocks = parse_int_list(args.num_res_blocks)
        if len(dims) != len(blocks):
            raise SystemExit("--dims and --num-res-blocks must have the same number of entries")
        if args.context_frames != 16:
            raise SystemExit("--context-frames must be 16 for the current ConvEncoder")
        if args.target_mode == "future" and args.target_frames != args.context_frames:
            raise SystemExit("--target-frames must equal --context-frames when --target-mode future")
        if not (0.0 < args.mask_ratio < 1.0):
            raise SystemExit("--mask-ratio must be in (0, 1)")
        if args.mask_min_keep < 0:
            raise SystemExit("--mask-min-keep must be non-negative")
        if args.warmup_epochs < 0:
            raise SystemExit("--warmup-epochs must be non-negative")
        if not (0.0 < args.warmup_start_factor <= 1.0):
            raise SystemExit("--warmup-start-factor must be in (0, 1]")
        if not (0.0 <= args.ema_momentum <= args.ema_final_momentum <= 1.0):
            raise SystemExit("EMA momentum must satisfy 0 <= --ema-momentum <= --ema-final-momentum <= 1")
        if args.train_index_mode != "sliding_window" and args.train_stride != 1:
            raise SystemExit("--train-stride only applies when --train-index-mode sliding_window")
        if args.valid_index_mode != "sliding_window" and args.valid_stride != 1:
            raise SystemExit("--valid-stride only applies when --valid-index-mode sliding_window")

        dist_state = _init_dist_state(args.device, args.dist_backend)
        seed_everything(args.seed)
        configure_torch_runtime(deterministic=args.deterministic)

        out_dir = ensure_dir(args.out_dir)
        dataset_target_frames = args.target_frames if args.target_mode == "future" else 0
        train_dataset = ActiveMatterWindowDataset(
            root=args.data_root,
            split="train",
            context_frames=args.context_frames,
            target_frames=dataset_target_frames,
            stride=args.train_stride,
            resolution=args.resolution,
            max_samples=args.max_train_samples,
            index_mode=args.train_index_mode,
            clip_selection=args.clip_selection,
            include_labels=False,
        )
        valid_dataset = ActiveMatterWindowDataset(
            root=args.data_root,
            split="valid",
            context_frames=args.context_frames,
            target_frames=dataset_target_frames,
            stride=args.valid_stride,
            resolution=args.resolution,
            max_samples=args.max_valid_samples,
            index_mode=args.valid_index_mode,
            clip_selection=args.clip_selection,
            include_labels=False,
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

        model = VJepaModel(
            in_chans=sum(field.channels for field in train_dataset.field_specs),
            dims=dims,
            num_res_blocks=blocks,
            num_frames=args.context_frames,
            stem_patch_size=args.stem_patch_size,
            stem_kernel_size=args.stem_kernel_size,
            drop_path_rate=args.drop_path_rate,
            layer_scale_init_value=args.layer_scale_init_value,
        ).to(dist_state.device)

        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = min(max(0, int(args.warmup_epochs * len(train_loader))), max(total_steps - 1, 0))
        config_payload = vars(args).copy()
        config_payload["uses_physical_labels"] = False
        config_payload["dims"] = dims
        config_payload["num_res_blocks"] = blocks
        config_payload["in_chans"] = sum(field.channels for field in train_dataset.field_specs)
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
        config_payload["trainable_params"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        config_payload["encoder_params"] = _count_params(model.encoder)
        config_payload["predictor_params"] = _count_params(model.predictor)
        config_payload["target_encoder_params"] = _count_params(model.target_encoder)

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
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
            model.encoder.load_state_dict(resume_payload["encoder"])
            model.target_encoder.load_state_dict(resume_payload.get("target_encoder", resume_payload["encoder"]))
            model.predictor.load_state_dict(resume_payload["predictor"])
            if "mask_token" in resume_payload:
                model.mask_token.data.copy_(resume_payload["mask_token"].to(model.mask_token.device))
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

        global_step = max(0, (start_epoch - 1) * len(train_loader))
        feature_h, feature_w = _feature_grid_size(
            args.resolution,
            args.resolution,
            args.stem_patch_size,
            len(dims),
        )

        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            model.train()
            train_metric_sums = _empty_metric_sums()
            train_sample_count = 0

            for batch in train_loader:
                context = batch["context"].to(dist_state.device, non_blocking=True)
                target = _target_from_batch(batch, context, args.target_mode, dist_state.device)
                mask = _sample_block_masks(
                    batch_size=int(context.shape[0]),
                    height=feature_h,
                    width=feature_w,
                    mask_ratio=args.mask_ratio,
                    min_block_size=args.mask_min_block_size,
                    max_block_size=args.mask_max_block_size,
                    max_blocks=args.mask_max_blocks,
                    min_keep=args.mask_min_keep,
                    device=dist_state.device,
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=dist_state.device.type,
                    dtype=amp_dtype,
                    enabled=args.amp and dist_state.device.type == "cuda",
                ):
                    output = model(context, target, mask)
                    loss_dict = masked_latent_prediction_loss(
                        output["predicted_features"],
                        output["target_features"],
                        output["mask"],
                        normalize_target=args.normalize_target,
                        normalize_pred=args.normalize_pred,
                    )
                    loss_dict["context_std"] = feature_map_std(output["context_features"]).detach()
                    momentum = _ema_momentum_for_step(
                        args.ema_momentum,
                        args.ema_final_momentum,
                        global_step,
                        total_steps,
                    )
                    loss_dict["ema_momentum"] = output["predicted_features"].new_tensor(momentum)
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
                    _unwrap_model(model).update_target_encoder(momentum)
                    global_step += 1

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
                    target = _target_from_batch(batch, context, args.target_mode, dist_state.device)
                    mask = _sample_block_masks(
                        batch_size=int(context.shape[0]),
                        height=feature_h,
                        width=feature_w,
                        mask_ratio=args.mask_ratio,
                        min_block_size=args.mask_min_block_size,
                        max_block_size=args.mask_max_block_size,
                        max_blocks=args.mask_max_blocks,
                        min_keep=args.mask_min_keep,
                        device=dist_state.device,
                    )
                    with torch.autocast(
                        device_type=dist_state.device.type,
                        dtype=amp_dtype,
                        enabled=args.amp and dist_state.device.type == "cuda",
                    ):
                        output = model(context, target, mask)
                        loss_dict = masked_latent_prediction_loss(
                            output["predicted_features"],
                            output["target_features"],
                            output["mask"],
                            normalize_target=args.normalize_target,
                            normalize_pred=args.normalize_pred,
                        )
                        loss_dict["context_std"] = feature_map_std(output["context_features"]).detach()
                        loss_dict["ema_momentum"] = output["predicted_features"].new_tensor(args.ema_final_momentum)

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
                "train_pred_loss": train_summary["pred_loss"],
                "train_context_std": train_summary["context_std"],
                "train_pred_std": train_summary["pred_std"],
                "train_target_std": train_summary["target_std"],
                "train_mask_ratio": train_summary["mask_ratio"],
                "train_ema_momentum": train_summary["ema_momentum"],
                "valid_loss": valid_summary["loss"],
                "valid_pred_loss": valid_summary["pred_loss"],
                "valid_context_std": valid_summary["context_std"],
                "valid_pred_std": valid_summary["pred_std"],
                "valid_target_std": valid_summary["target_std"],
                "valid_mask_ratio": valid_summary["mask_ratio"],
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
                    "encoder": raw_model.encoder.state_dict(),
                    "target_encoder": raw_model.target_encoder.state_dict(),
                    "predictor": raw_model.predictor.state_dict(),
                    "mask_token": raw_model.mask_token.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_valid_loss": best_val,
                    "history": history,
                    "wandb_run_id": wandb_state.run_id,
                }
                atomic_torch_save(out_dir / "last.pt", checkpoint)
                atomic_torch_save(
                    out_dir / "encoder_last.pt",
                    {
                        "epoch": epoch,
                        "config": config_payload,
                        "state_dict": raw_model.encoder.state_dict(),
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
                            "state_dict": raw_model.encoder.state_dict(),
                            "best_valid_loss": best_val,
                        },
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
