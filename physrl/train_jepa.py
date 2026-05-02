from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from .data import ActiveMatterWindowDataset
from .losses import sigreg_jepa_loss, sigreg_loss, vicreg_loss
from .models import JepaModel, SigRegJepaModel
from .utils import (
    atomic_torch_save,
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    load_torch_checkpoint,
    make_torch_generator,
    parse_int_list,
    save_json,
    seed_everything,
    seed_worker,
)


METRIC_KEYS = (
    "loss",
    "pred_loss",
    "sigreg_loss",
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


@dataclass
class WandbState:
    enabled: bool
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a JEPA encoder on active_matter from scratch.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path containing train/valid/test or data/train/valid/test.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/jepa"), help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-start-factor", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-frames", type=int, default=16)
    parser.add_argument("--target-frames", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--valid-stride", type=int, default=1)
    parser.add_argument(
        "--train-index-mode",
        choices=["single_clip", "sliding_window"],
        default="sliding_window",
        help="How to sample training clips from raw trajectories.",
    )
    parser.add_argument(
        "--valid-index-mode",
        choices=["single_clip", "sliding_window"],
        default="single_clip",
        help="How to sample validation clips from raw trajectories.",
    )
    parser.add_argument(
        "--clip-selection",
        choices=["start", "center", "end"],
        default="center",
        help="Clip position used when an index mode is single_clip.",
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--dims", type=str, default="32,64,128,256,256")
    parser.add_argument("--num-res-blocks", type=str, default="2,2,4,8,2")
    parser.add_argument("--stem-patch-size", type=int, default=2)
    parser.add_argument("--stem-kernel-size", type=int, default=4)
    parser.add_argument("--drop-path-rate", type=float, default=0.05)
    parser.add_argument("--layer-scale-init-value", type=float, default=1e-6)
    parser.add_argument("--loss-type", choices=["sigreg", "sicreg", "vicreg"], default="sigreg")
    parser.add_argument("--sim-coeff", type=float, default=2.0)
    parser.add_argument("--std-coeff", type=float, default=40.0)
    parser.add_argument("--cov-coeff", type=float, default=2.0)
    parser.add_argument("--loss-chunks", type=int, default=5)
    parser.add_argument("--loss-groups", type=int, default=1)
    parser.add_argument("--loss-fp32-stats", action="store_true")
    parser.add_argument("--loss-zscore-for-cov", action="store_true")
    parser.add_argument("--loss-adaptive-cov-scale", action="store_true")
    parser.add_argument(
        "--lejepa-lambda",
        type=float,
        default=0.05,
        help="SIGReg trade-off lambda. Used as pred=(1-lambda), sigreg=lambda unless coeff overrides are set.",
    )
    parser.add_argument("--pred-coeff", type=float, default=None, help="Override the resolved LeJEPA prediction coefficient.")
    parser.add_argument("--sigreg-coeff", type=float, default=None, help="Override the resolved SIGReg coefficient.")
    parser.add_argument("--sigreg-slices", type=int, default=1024)
    parser.add_argument("--sigreg-points", type=int, default=17)
    parser.add_argument("--sigreg-t-max", type=float, default=3.0)
    parser.add_argument(
        "--sigreg-on",
        choices=["projection", "embedding", "both"],
        default="projection",
        help="Tensor regularized by SIGReg. 'embedding' targets the pooled encoder output used for downstream export.",
    )
    parser.add_argument("--projector-dim", type=int, default=256)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument("--projector-layers", type=int, default=3)
    parser.add_argument("--predictor-hidden-dim", type=int, default=1024)
    parser.add_argument("--predictor-layers", type=int, default=2)
    parser.add_argument(
        "--target-stop-grad",
        action="store_true",
        help="Use a stop-gradient target branch for SIGReg JEPA ablations. Default follows LeJEPA and keeps gradients on both views.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Autocast dtype used when --amp is enabled.",
    )
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic PyTorch behavior where feasible.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start from a previous checkpoint without resuming optimizer/scheduler/history.",
    )
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
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Weights & Biases logging mode.",
    )
    parser.add_argument("--wandb-entity", type=str, default="abhinaykrishna60-new-york-university")
    parser.add_argument("--wandb-project", type=str, default="dl_final-comp")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def _build_loader(
    dataset: ActiveMatterWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    sampler: Sampler[int] | None = None,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(seed),
    )


def _resolve_resume_path(out_dir: Path, resume: str | None) -> Path | None:
    if resume is None:
        return None
    if resume == "auto":
        return out_dir / "last.pt"
    return Path(resume).expanduser().resolve()


def _resolve_init_path(init_checkpoint: Path | None) -> Path | None:
    if init_checkpoint is None:
        return None
    return init_checkpoint.expanduser().resolve()


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


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


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


def _load_init_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    payload = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    loaded_encoder = False
    loaded_predictor = False
    loaded_projector = False

    if "encoder" in payload:
        model.encoder.load_state_dict(payload["encoder"])
        loaded_encoder = True
    elif "state_dict" in payload:
        model.encoder.load_state_dict(payload["state_dict"])
        loaded_encoder = True

    if "predictor" in payload and hasattr(model, "predictor"):
        try:
            model.predictor.load_state_dict(payload["predictor"])
            loaded_predictor = True
        except RuntimeError:
            loaded_predictor = False
    if "projector" in payload and hasattr(model, "projector"):
        model.projector.load_state_dict(payload["projector"])
        loaded_projector = True

    if not loaded_encoder:
        raise ValueError(
            f"init checkpoint must contain 'encoder' or 'state_dict': {checkpoint_path}"
        )

    return {
        "path": str(checkpoint_path),
        "epoch": payload.get("epoch"),
        "loaded_encoder": loaded_encoder,
        "loaded_predictor": loaded_predictor,
        "loaded_projector": loaded_projector,
        "source_best_valid_loss": payload.get("best_valid_loss"),
    }


def _init_wandb(
    *,
    args: argparse.Namespace,
    dist_state: DistState,
    config_payload: dict[str, Any],
    out_dir: Path,
    resume_payload: dict[str, Any] | None,
) -> WandbState:
    if args.wandb_mode == "disabled" or not dist_state.is_main_process:
        return WandbState(enabled=False)
    if wandb is None:
        raise RuntimeError("wandb is not installed, but wandb logging is enabled")

    run_id = None
    if resume_payload is not None:
        run_id = resume_payload.get("wandb_run_id")

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        dir=str(out_dir),
        config=config_payload,
        id=run_id,
        resume="allow" if run_id else None,
        mode=args.wandb_mode,
    )
    if run_id is None:
        run_id = run.id
    return WandbState(enabled=True, run=run, run_id=run_id)


def _count_params(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _sigreg_jepa_loss_for_output(
    output: dict[str, torch.Tensor],
    *,
    sigreg_on: str,
    pred_coeff: float,
    sigreg_coeff: float,
    num_slices: int,
    num_points: int,
    t_max: float,
    seed: int,
    distributed: bool,
) -> dict[str, torch.Tensor]:
    if sigreg_on == "projection":
        sigreg_embeddings = output["sigreg_projection"]
        return sigreg_jepa_loss(
            output["predicted_projection"],
            output["target_projection"],
            sigreg_embeddings,
            pred_coeff=pred_coeff,
            sigreg_coeff=sigreg_coeff,
            num_slices=num_slices,
            num_points=num_points,
            t_max=t_max,
            seed=seed,
            distributed=distributed,
        )

    if sigreg_on == "embedding":
        sigreg_embeddings = output["sigreg_embedding"]
        return sigreg_jepa_loss(
            output["predicted_projection"],
            output["target_projection"],
            sigreg_embeddings,
            pred_coeff=pred_coeff,
            sigreg_coeff=sigreg_coeff,
            num_slices=num_slices,
            num_points=num_points,
            t_max=t_max,
            seed=seed,
            distributed=distributed,
        )

    if sigreg_on != "both":
        raise ValueError(f"unsupported sigreg_on: {sigreg_on}")

    loss_dict = sigreg_jepa_loss(
        output["predicted_projection"],
        output["target_projection"],
        output["sigreg_projection"],
        pred_coeff=pred_coeff,
        sigreg_coeff=0.5 * sigreg_coeff,
        num_slices=num_slices,
        num_points=num_points,
        t_max=t_max,
        seed=seed,
        distributed=distributed,
    )
    embedding_sigreg = sigreg_loss(
        output["sigreg_embedding"],
        num_slices=num_slices,
        num_points=num_points,
        t_max=t_max,
        seed=seed + 1,
        distributed=distributed,
    )
    loss_dict["loss"] = loss_dict["loss"] + 0.5 * float(sigreg_coeff) * embedding_sigreg
    loss_dict["sigreg_loss"] = 0.5 * (loss_dict["sigreg_loss"] + embedding_sigreg.detach())
    return loss_dict


def main() -> None:
    dist_state: DistState | None = None
    wandb_state = WandbState(enabled=False)
    try:
        args = parse_args()
        loss_type = "sigreg" if args.loss_type == "sicreg" else args.loss_type
        dims = parse_int_list(args.dims)
        blocks = parse_int_list(args.num_res_blocks)
        if len(dims) != len(blocks):
            raise SystemExit("--dims and --num-res-blocks must have the same number of entries")
        if args.warmup_epochs < 0:
            raise SystemExit("--warmup-epochs must be non-negative")
        if not (0.0 < args.warmup_start_factor <= 1.0):
            raise SystemExit("--warmup-start-factor must be in (0, 1]")
        if not (0.0 <= args.lejepa_lambda <= 1.0):
            raise SystemExit("--lejepa-lambda must be in [0, 1]")
        pred_coeff = float(1.0 - args.lejepa_lambda) if args.pred_coeff is None else float(args.pred_coeff)
        sigreg_coeff = float(args.lejepa_lambda) if args.sigreg_coeff is None else float(args.sigreg_coeff)
        if pred_coeff < 0.0 or sigreg_coeff < 0.0:
            raise SystemExit("--pred-coeff and --sigreg-coeff must be non-negative")
        if args.train_index_mode != "sliding_window" and args.train_stride != 1:
            raise SystemExit("--train-stride only applies when --train-index-mode sliding_window")
        if args.valid_index_mode != "sliding_window" and args.valid_stride != 1:
            raise SystemExit("--valid-stride only applies when --valid-index-mode sliding_window")

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

        model_kwargs = {
            "in_chans": sum(field.channels for field in train_dataset.field_specs),
            "dims": dims,
            "num_res_blocks": blocks,
            "num_frames": args.context_frames,
            "stem_patch_size": args.stem_patch_size,
            "stem_kernel_size": args.stem_kernel_size,
            "drop_path_rate": args.drop_path_rate,
            "layer_scale_init_value": args.layer_scale_init_value,
        }
        if loss_type == "sigreg":
            model = SigRegJepaModel(
                **model_kwargs,
                projector_dim=args.projector_dim,
                projector_hidden_dim=args.projector_hidden_dim,
                projector_layers=args.projector_layers,
                predictor_hidden_dim=args.predictor_hidden_dim,
                predictor_layers=args.predictor_layers,
            ).to(dist_state.device)
        else:
            model = JepaModel(**model_kwargs).to(dist_state.device)

        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = min(max(0, int(args.warmup_epochs * len(train_loader))), max(total_steps - 1, 0))

        config_payload = vars(args).copy()
        config_payload["loss_type"] = loss_type
        config_payload["resolved_pred_coeff"] = pred_coeff
        config_payload["resolved_sigreg_coeff"] = sigreg_coeff
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
        config_payload["encoder_params"] = _count_params(model.encoder)
        if hasattr(model, "projector"):
            config_payload["projector_params"] = _count_params(model.projector)
        if hasattr(model, "predictor"):
            config_payload["predictor_params"] = _count_params(model.predictor)

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
        init_summary: dict[str, Any] | None = None

        resume_path = _resolve_resume_path(out_dir, args.resume)
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
            resume_payload = load_torch_checkpoint(resume_path, map_location="cpu")
            if "encoder" not in resume_payload or "predictor" not in resume_payload:
                raise ValueError(f"resume checkpoint must be a full training checkpoint: {resume_path}")
            model.encoder.load_state_dict(resume_payload["encoder"])
            model.predictor.load_state_dict(resume_payload["predictor"])
            if hasattr(model, "projector"):
                if "projector" not in resume_payload:
                    raise ValueError(f"SIGReg resume checkpoint is missing projector state: {resume_path}")
                model.projector.load_state_dict(resume_payload["projector"])
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
                        "world_size": dist_state.world_size,
                    },
                    flush=True,
                )
        else:
            init_path = _resolve_init_path(args.init_checkpoint)
            if init_path is not None:
                if not init_path.exists():
                    raise FileNotFoundError(f"init checkpoint not found: {init_path}")
                init_summary = _load_init_checkpoint(model, init_path)
                config_payload["init_checkpoint"] = init_summary["path"]
                config_payload["init_checkpoint_epoch"] = init_summary["epoch"]
                config_payload["init_loaded_predictor"] = init_summary["loaded_predictor"]
                config_payload["init_loaded_projector"] = init_summary["loaded_projector"]
                config_payload["init_source_best_valid_loss"] = init_summary["source_best_valid_loss"]
                if dist_state.is_main_process:
                    print(
                        {
                            "status": "initialized",
                            "init_checkpoint": init_summary["path"],
                            "checkpoint_epoch": init_summary["epoch"],
                            "loaded_predictor": init_summary["loaded_predictor"],
                            "loaded_projector": init_summary["loaded_projector"],
                        },
                        flush=True,
                    )

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

            for batch_idx, batch in enumerate(train_loader):
                context = batch["context"].to(dist_state.device, non_blocking=True)
                target = batch["target"].to(dist_state.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=dist_state.device.type,
                    dtype=amp_dtype,
                    enabled=args.amp and dist_state.device.type == "cuda",
                ):
                    if loss_type == "sigreg":
                        output = model(context, target, target_stop_grad=args.target_stop_grad)
                        loss_dict = _sigreg_jepa_loss_for_output(
                            output,
                            sigreg_on=args.sigreg_on,
                            pred_coeff=pred_coeff,
                            sigreg_coeff=sigreg_coeff,
                            num_slices=args.sigreg_slices,
                            num_points=args.sigreg_points,
                            t_max=args.sigreg_t_max,
                            seed=args.seed + epoch * max(len(train_loader), 1) + batch_idx,
                            distributed=dist_state.enabled,
                        )
                    else:
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
                for batch_idx, batch in enumerate(valid_loader):
                    context = batch["context"].to(dist_state.device, non_blocking=True)
                    target = batch["target"].to(dist_state.device, non_blocking=True)
                    with torch.autocast(
                        device_type=dist_state.device.type,
                        dtype=amp_dtype,
                        enabled=args.amp and dist_state.device.type == "cuda",
                    ):
                        if loss_type == "sigreg":
                            output = model(context, target, target_stop_grad=args.target_stop_grad)
                            loss_dict = _sigreg_jepa_loss_for_output(
                                output,
                                sigreg_on=args.sigreg_on,
                                pred_coeff=pred_coeff,
                                sigreg_coeff=sigreg_coeff,
                                num_slices=args.sigreg_slices,
                                num_points=args.sigreg_points,
                                t_max=args.sigreg_t_max,
                                seed=args.seed + 10_000_000 + epoch * max(len(valid_loader), 1) + batch_idx,
                                distributed=False,
                            )
                        else:
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
                "valid_loss": valid_summary["loss"],
                "lr": optimizer.param_groups[0]["lr"],
                "world_size": dist_state.world_size,
            }
            if loss_type == "sigreg":
                epoch_record.update(
                    {
                        "train_pred_loss": train_summary["pred_loss"],
                        "train_sigreg_loss": train_summary["sigreg_loss"],
                        "valid_pred_loss": valid_summary["pred_loss"],
                        "valid_sigreg_loss": valid_summary["sigreg_loss"],
                    }
                )
            else:
                epoch_record.update(
                    {
                        "train_repr_loss": train_summary["repr_loss"],
                        "train_std_loss": train_summary["std_loss"],
                        "train_cov_loss": train_summary["cov_loss"],
                        "train_std_loss_pred": train_summary["std_loss_pred"],
                        "train_std_loss_target": train_summary["std_loss_target"],
                        "train_cov_loss_pred": train_summary["cov_loss_pred"],
                        "train_cov_loss_target": train_summary["cov_loss_target"],
                        "valid_repr_loss": valid_summary["repr_loss"],
                        "valid_std_loss": valid_summary["std_loss"],
                        "valid_cov_loss": valid_summary["cov_loss"],
                        "valid_std_loss_pred": valid_summary["std_loss_pred"],
                        "valid_std_loss_target": valid_summary["std_loss_target"],
                        "valid_cov_loss_pred": valid_summary["cov_loss_pred"],
                        "valid_cov_loss_target": valid_summary["cov_loss_target"],
                    }
                )
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
                    "predictor": raw_model.predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_valid_loss": best_val,
                    "history": history,
                    "wandb_run_id": wandb_state.run_id,
                }
                if hasattr(raw_model, "projector"):
                    checkpoint["projector"] = raw_model.projector.state_dict()
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
