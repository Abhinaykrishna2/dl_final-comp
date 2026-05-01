"""End-to-end supervised baseline for active matter parameter estimation.

This is the upper-bound anchor for the report: same encoder as VideoMAE / SIGReg-JEPA
(byte-identical ConvEncoder) plus a single linear head, trained directly on the
(alpha, zeta) labels with z-scored MSE loss. Per the assignment FAQ, training a
purely supervised baseline "provides insights into the value of [the SSL] approach."

The encoder weights are also saved separately (in the same format that
videomae/export_embeddings.py consumes) so we can run a *frozen* linear probe and
kNN on the supervised encoder for an apples-to-apples comparison against the SSL
encoders. A low end-to-end MSE combined with a high frozen-probe MSE would be a
particularly interesting result.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from .data import ActiveMatterWindowDataset, collect_split_labels
from .models import ConvEncoder
from .utils import (
    LabelNormalizer,
    atomic_torch_save,
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    make_torch_generator,
    mse_report,
    parse_int_list,
    pool_features,
    save_json,
    seed_everything,
    seed_worker,
)


# Reuse the dist-state helper conventions from train_videomae.py
@dataclass(frozen=True)
class DistState:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass
class WandbState:
    enabled: bool = False
    run: Any | None = None
    run_id: str | None = None


class StridedShardSampler(Sampler[int]):
    def __init__(self, dataset_size: int, *, rank: int, world_size: int) -> None:
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        remaining = self.dataset_size - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.world_size - 1) // self.world_size


class SupervisedModel(nn.Module):
    """ConvEncoder + average-pool + nn.Linear(embed_dim, 2). End-to-end MSE on (alpha, zeta)."""

    def __init__(self, encoder: ConvEncoder, num_outputs: int = 2, pool: str = "avg") -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_outputs)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        pooled = pool_features(feat, self.pool)
        return self.head(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end supervised baseline on active_matter.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/supervised"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-start-factor", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--train-index-mode", choices=["single_clip", "sliding_window"], default="single_clip",
                        help="single_clip is appropriate for a labelled regression task: one clip per simulation per epoch.")
    parser.add_argument("--valid-index-mode", choices=["single_clip", "sliding_window"], default="single_clip")
    parser.add_argument("--clip-selection", choices=["start", "center", "end"], default="center")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--dims", type=str, default="32,64,128,256,256")
    parser.add_argument("--num-res-blocks", type=str, default="2,2,4,8,2")
    parser.add_argument("--stem-patch-size", type=int, default=2)
    parser.add_argument("--stem-kernel-size", type=int, default=4)
    parser.add_argument("--drop-path-rate", type=float, default=0.05)
    parser.add_argument("--layer-scale-init-value", type=float, default=1e-6)
    parser.add_argument("--pool", choices=["avg", "flatten", "avgmax"], default="avg")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--dist-backend", choices=["nccl", "gloo"], default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="dl-final-videomae")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def _build_loader(
    dataset: ActiveMatterWindowDataset,
    *,
    batch_size: int, num_workers: int, prefetch_factor: int, shuffle: bool,
    sampler: Sampler[int] | None, seed: int,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(seed),
    )


def _slurm_env_fallback() -> None:
    import os
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29500")


def _init_dist_state(device_arg: str, dist_backend: str | None) -> DistState:
    import os
    _slurm_env_fallback()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if not enabled:
        return DistState(False, 0, 1, 0, choose_device(device_arg), None)
    backend = dist_backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return DistState(True, rank, world_size, local_rank, device, backend)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _all_reduce_mean(sum_value: float, count: int, *, device: torch.device, distributed: bool) -> float:
    payload = torch.tensor([float(sum_value), float(count)], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    total = payload[1].item()
    if total <= 0:
        return float("nan")
    return float(payload[0].item() / total)


def main() -> None:
    dist_state: DistState | None = None
    wandb_state = WandbState(enabled=False)
    try:
        args = parse_args()
        dims = parse_int_list(args.dims)
        blocks = parse_int_list(args.num_res_blocks)
        if len(dims) != len(blocks):
            raise SystemExit("--dims and --num-res-blocks must have the same number of entries")

        dist_state = _init_dist_state(args.device, args.dist_backend)
        seed_everything(args.seed)
        configure_torch_runtime(deterministic=args.deterministic)

        out_dir = ensure_dir(args.out_dir)

        train_dataset = ActiveMatterWindowDataset(
            root=args.data_root, split="train",
            context_frames=args.num_frames, target_frames=0,
            stride=args.train_stride, resolution=args.resolution,
            max_samples=args.max_train_samples,
            index_mode=args.train_index_mode, clip_selection=args.clip_selection,
        )
        valid_dataset = ActiveMatterWindowDataset(
            root=args.data_root, split="valid",
            context_frames=args.num_frames, target_frames=0,
            stride=1, resolution=args.resolution,
            max_samples=args.max_valid_samples,
            index_mode=args.valid_index_mode, clip_selection=args.clip_selection,
        )

        # Z-score normalize labels using TRAIN labels only (no val/test leakage).
        train_labels_np = np.asarray(
            [train_dataset.file_infos[fi].labels[si] for fi, si, _ in train_dataset.index],
            dtype=np.float32,
        )
        label_norm = LabelNormalizer.fit(train_labels_np)
        label_mean_t = torch.from_numpy(label_norm.mean.astype(np.float32))
        label_std_t = torch.from_numpy(label_norm.std.astype(np.float32))

        train_sampler: Sampler[int] | None = None
        valid_sampler: Sampler[int] | None = None
        if dist_state.enabled:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=dist_state.world_size, rank=dist_state.rank,
                shuffle=True, drop_last=False,
            )
            valid_sampler = StridedShardSampler(
                len(valid_dataset), rank=dist_state.rank, world_size=dist_state.world_size,
            )
        train_loader = _build_loader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor, shuffle=True, sampler=train_sampler,
            seed=args.seed + dist_state.rank,
        )
        valid_loader = _build_loader(
            valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor, shuffle=False, sampler=valid_sampler,
            seed=args.seed + 10_000 + dist_state.rank,
        )

        encoder = ConvEncoder(
            in_chans=sum(field.channels for field in train_dataset.field_specs),
            dims=list(dims), num_res_blocks=list(blocks),
            num_frames=args.num_frames,
            stem_patch_size=args.stem_patch_size, stem_kernel_size=args.stem_kernel_size,
            drop_path_rate=args.drop_path_rate, layer_scale_init_value=args.layer_scale_init_value,
        )
        model = SupervisedModel(encoder, num_outputs=label_norm.mean.shape[0], pool=args.pool).to(dist_state.device)
        label_mean_t = label_mean_t.to(dist_state.device)
        label_std_t = label_std_t.to(dist_state.device)

        total_params = sum(p.numel() for p in model.parameters())
        if total_params >= 100_000_000:
            raise SystemExit(f"Total params {total_params:,} >= 100M cap")

        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = min(max(0, int(args.warmup_epochs * len(train_loader))), max(total_steps - 1, 0))

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if warmup_steps > 0:
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=args.warmup_start_factor, end_factor=1.0,
                        total_iters=warmup_steps,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.min_lr,
                    ),
                ],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=args.min_lr,
            )
        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        scaler = torch.amp.GradScaler(
            "cuda", enabled=args.amp and dist_state.device.type == "cuda" and amp_dtype == torch.float16,
        )

        config_payload = vars(args).copy()
        config_payload["dims"] = dims
        config_payload["num_res_blocks"] = blocks
        config_payload["in_chans"] = sum(field.channels for field in train_dataset.field_specs)
        config_payload["distributed"] = dist_state.enabled
        config_payload["world_size"] = dist_state.world_size
        config_payload["backend"] = dist_state.backend
        config_payload["out_dir"] = str(out_dir)
        config_payload["train_samples"] = len(train_dataset)
        config_payload["valid_samples"] = len(valid_dataset)
        config_payload["total_params"] = total_params
        config_payload["encoder_params"] = sum(p.numel() for p in model.encoder.parameters())
        config_payload["head_params"] = sum(p.numel() for p in model.head.parameters())
        config_payload["context_frames"] = args.num_frames
        config_payload["label_norm"] = label_norm.to_dict()
        if dist_state.is_main_process:
            save_json(out_dir / "train_config.json", config_payload)

        if args.wandb_mode != "disabled" and dist_state.is_main_process and wandb is not None:
            wandb_state = WandbState(
                enabled=True,
                run=wandb.init(
                    entity=args.wandb_entity, project=args.wandb_project,
                    name=args.wandb_run_name, dir=str(out_dir),
                    config=config_payload, mode=args.wandb_mode,
                ),
            )
            wandb_state.run_id = wandb_state.run.id

        if dist_state.enabled:
            ddp_kwargs = dict(broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)
            if dist_state.device.type == "cuda":
                model = DDP(model, device_ids=[dist_state.device.index], output_device=dist_state.device.index, **ddp_kwargs)
            else:
                model = DDP(model, **ddp_kwargs)

        best_val = float("inf")
        history: list[dict[str, float | int]] = []
        loss_fn = nn.MSELoss()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            model.train()
            train_loss_sum = 0.0
            train_n = 0
            for batch in train_loader:
                clip = batch["context"].to(dist_state.device, non_blocking=True)
                label_raw = batch["label"].to(dist_state.device, non_blocking=True)
                label_n = (label_raw - label_mean_t) / label_std_t
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=dist_state.device.type, dtype=amp_dtype,
                    enabled=args.amp and dist_state.device.type == "cuda",
                ):
                    pred = model(clip)
                    loss = loss_fn(pred, label_n)
                prev_scale = scaler.get_scale() if scaler.is_enabled() else None
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if not scaler.is_enabled() or scaler.get_scale() >= (prev_scale or 0.0):
                    scheduler.step()
                bs = int(clip.shape[0])
                train_n += bs
                train_loss_sum += float(loss.item()) * bs

            train_loss_avg = _all_reduce_mean(
                train_loss_sum, train_n, device=dist_state.device, distributed=dist_state.enabled,
            )

            model.eval()
            valid_loss_sum = 0.0
            valid_n = 0
            preds_list: list[np.ndarray] = []
            targs_list: list[np.ndarray] = []
            with torch.no_grad():
                for batch in valid_loader:
                    clip = batch["context"].to(dist_state.device, non_blocking=True)
                    label_raw = batch["label"].to(dist_state.device, non_blocking=True)
                    label_n = (label_raw - label_mean_t) / label_std_t
                    with torch.autocast(
                        device_type=dist_state.device.type, dtype=amp_dtype,
                        enabled=args.amp and dist_state.device.type == "cuda",
                    ):
                        pred = model(clip)
                        loss = loss_fn(pred, label_n)
                    valid_n += int(clip.shape[0])
                    valid_loss_sum += float(loss.item()) * int(clip.shape[0])
                    preds_list.append(pred.detach().float().cpu().numpy())
                    targs_list.append(label_raw.detach().float().cpu().numpy())
            valid_loss_avg = _all_reduce_mean(
                valid_loss_sum, valid_n, device=dist_state.device, distributed=dist_state.enabled,
            )

            preds_n = np.concatenate(preds_list, axis=0)
            targs = np.concatenate(targs_list, axis=0)
            preds_raw = label_norm.inverse_transform(preds_n)
            valid_raw_mse = mse_report(preds_raw, targs)
            valid_norm_mse = mse_report(preds_n, label_norm.transform(targs))

            elapsed = time.time() - t0
            epoch_record = {
                "epoch": epoch, "seconds": round(elapsed, 2),
                "train_loss": train_loss_avg, "valid_loss": valid_loss_avg,
                "lr": optimizer.param_groups[0]["lr"], "world_size": dist_state.world_size,
                "valid_alpha_mse": valid_raw_mse["alpha_mse"],
                "valid_zeta_mse": valid_raw_mse["zeta_mse"],
                "valid_mean_mse": valid_raw_mse["mean_mse"],
                "valid_mean_mse_normalized": valid_norm_mse["mean_mse"],
            }
            history.append(epoch_record)
            if dist_state.is_main_process:
                print(epoch_record, flush=True)
                if wandb_state.enabled and wandb_state.run is not None:
                    wandb_state.run.log(epoch_record, step=epoch)

            improved = valid_norm_mse["mean_mse"] < best_val
            if improved:
                best_val = valid_norm_mse["mean_mse"]

            if dist_state.is_main_process:
                raw_model = _unwrap_model(model)
                full_state = {
                    "epoch": epoch,
                    "config": config_payload,
                    "encoder": raw_model.encoder.state_dict(),
                    "head": raw_model.head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_valid_mean_mse_normalized": best_val,
                    "history": history,
                    "label_stats": label_norm.to_dict(),
                    "wandb_run_id": wandb_state.run_id,
                }
                atomic_torch_save(out_dir / "last.pt", full_state)
                atomic_torch_save(
                    out_dir / "encoder_last.pt",
                    {
                        "epoch": epoch, "config": config_payload,
                        "state_dict": raw_model.encoder.state_dict(),
                        "best_valid_mean_mse_normalized": best_val,
                    },
                )
                if args.save_every > 0 and epoch % args.save_every == 0:
                    atomic_torch_save(out_dir / f"epoch_{epoch:04d}.pt", full_state)
                if improved:
                    atomic_torch_save(out_dir / "best.pt", full_state)
                    atomic_torch_save(
                        out_dir / "encoder_best.pt",
                        {
                            "epoch": epoch, "config": config_payload,
                            "state_dict": raw_model.encoder.state_dict(),
                            "best_valid_mean_mse_normalized": best_val,
                        },
                    )
                save_json(out_dir / "history.json", {"epochs": history, "best_valid_mean_mse_normalized": best_val})
                if wandb_state.enabled and wandb_state.run is not None:
                    wandb_state.run.summary["best_valid_mean_mse_normalized"] = best_val
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
