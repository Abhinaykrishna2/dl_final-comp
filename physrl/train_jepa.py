from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .data import ActiveMatterWindowDataset
from .losses import vicreg_loss
from .models import JepaModel
from .utils import atomic_torch_save, choose_device, ensure_dir, parse_int_list, save_json, seed_everything


METRIC_KEYS = (
    "loss",
    "repr_loss",
    "std_loss",
    "cov_loss",
    "std_loss_pred",
    "std_loss_target",
    "cov_loss_pred",
    "cov_loss_target",
)


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
    parser.add_argument(
        "--dist-backend",
        choices=["nccl", "gloo"],
        default=None,
        help="Distributed backend. Default is nccl on CUDA and gloo otherwise.",
    )
    return parser.parse_args()


def _build_loader(
    dataset: ActiveMatterWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _resolve_resume_path(out_dir: Path, resume: str | None) -> Path | None:
    if resume is None:
        return None
    if resume == "auto":
        return out_dir / "last.pt"
    return Path(resume).expanduser().resolve()


def _slurm_env_fallback() -> None:
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
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
        if "SLURM_LAUNCH_NODE_IPADDR" in os.environ:
            os.environ["MASTER_ADDR"] = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
        elif num_nodes == 1:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        else:
            raise RuntimeError(
                "Distributed run is missing MASTER_ADDR. Use torchrun, or export MASTER_ADDR and MASTER_PORT in Slurm."
            )
    os.environ.setdefault("MASTER_PORT", "29500")


def _init_dist_state(device_arg: str, dist_backend: str | None) -> DistState:
    _slurm_env_fallback()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if not enabled:
        return DistState(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=choose_device(device_arg),
            backend=None,
        )

    backend = dist_backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requires CUDA. Use --dist-backend gloo for CPU distributed runs.")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return DistState(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def _unwrap_model(model: torch.nn.Module) -> JepaModel:
    if isinstance(model, DDP):
        return model.module
    return model  # type: ignore[return-value]


def _empty_metric_sums() -> dict[str, float]:
    return {key: 0.0 for key in METRIC_KEYS}


def _update_metric_sums(metric_sums: dict[str, float], loss_dict: dict[str, torch.Tensor], weight: int) -> None:
    for key in METRIC_KEYS:
        metric_sums[key] += float(loss_dict[key].item()) * weight


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


def main() -> None:
    dist_state: DistState | None = None
    try:
        args = parse_args()
        dims = parse_int_list(args.dims)
        blocks = parse_int_list(args.num_res_blocks)
        if len(dims) != len(blocks):
            raise SystemExit("--dims and --num-res-blocks must have the same number of entries")

        dist_state = _init_dist_state(args.device, args.dist_backend)
        seed_everything(args.seed)

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
            shuffle=True,
            sampler=train_sampler,
        )
        valid_loader = _build_loader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=valid_sampler,
        )

        model = JepaModel(
            in_chans=sum(field.channels for field in train_dataset.field_specs),
            dims=dims,
            num_res_blocks=blocks,
            num_frames=args.context_frames,
        ).to(dist_state.device)
        config_payload = vars(args).copy()
        config_payload["dims"] = dims
        config_payload["num_res_blocks"] = blocks
        config_payload["in_chans"] = sum(field.channels for field in train_dataset.field_specs)
        config_payload["distributed"] = dist_state.enabled
        config_payload["world_size"] = dist_state.world_size
        config_payload["backend"] = dist_state.backend
        if dist_state.is_main_process:
            save_json(out_dir / "train_config.json", config_payload)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = max(1, args.epochs * len(train_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and dist_state.device.type == "cuda")

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
            if dist_state.is_main_process:
                print(
                    {
                        "status": "resumed",
                        "resume_path": str(resume_path),
                        "next_epoch": start_epoch,
                        "best_valid_loss": best_val,
                        "world_size": dist_state.world_size,
                    },
                    flush=True,
                )

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

        if dist_state.enabled:
            if dist_state.device.type == "cuda":
                model = DDP(
                    model,
                    device_ids=[dist_state.device.index],
                    output_device=dist_state.device.index,
                    broadcast_buffers=False,
                )
            else:
                model = DDP(model, broadcast_buffers=False)

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
                    dtype=torch.float16,
                    enabled=scaler.is_enabled(),
                ):
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
                "world_size": dist_state.world_size,
            }
            history.append(epoch_record)
            if dist_state.is_main_process:
                print(epoch_record, flush=True)

            improved = valid_summary["loss"] < best_val
            if improved:
                best_val = valid_summary["loss"]

            if dist_state.is_main_process:
                raw_model = _unwrap_model(model)
                checkpoint = {
                    "epoch": epoch,
                    "config": config_payload,
                    "encoder": raw_model.encoder.state_dict(),
                    "predictor": raw_model.predictor.state_dict(),
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

            if dist_state.enabled:
                dist.barrier()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
