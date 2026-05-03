"""VideoMAE / SimMIM-hybrid trainer for active matter (Person B's main SSL run).

Key design choices (with the paper that motivates each):

* Encoder: byte-identical copy of Person A's ``ConvEncoder`` from physrl/models.py
  so the SSL-objective comparison is causally clean.
* Mask ratio 0.60 default (SimMIM Tab. 1 optimal at patch-size = encoder downsampling).
* Tube size (16, 32, 32) = encoder downsampling factor (SimMIM Sec 4.1.2).
* Lightweight 5-stage transposed-conv decoder (~350K params; SimMIM Tab. 2 -- light heads transfer better).
* Per-tube z-score normalization of the reconstruction target (VideoMAE Sec 3.3, Tab. 1c).
* MSE loss on masked positions only (SimMIM Tab. 4: prediction-only beats full reconstruction).
* AdamW + cosine LR schedule with linear warmup; BF16 autocast on B200.
* Saves checkpoints in the same format ``physrl/export_embeddings.py`` consumes
  ({"encoder": ..., "config": ...}), so the colleague's eval pipeline works unchanged.
* Dual experiment tracking: writes JSON logs (history.json / train_config.json /
  metrics.json) AND W&B offline-mode logs by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
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
from .models import VideoMAEModel
from .utils import (
    atomic_torch_save,
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    make_torch_generator,
    parse_int_list,
    save_json,
    seed_everything,
    seed_worker,
)


@dataclass(frozen=True)
class DistState:
    """
    Parameters
    ----------
    **Input parameter 1:** enabled - True if running under distributed training (world_size > 1).
    **Input parameter 2:** rank - Global rank of this process across all nodes (0 to world_size - 1).
    **Input parameter 3:** world_size - Total number of processes participating in training.
    **Input parameter 4:** local_rank - Rank of this process within its node (0 to local_world_size - 1).
    **Input parameter 5:** device - Torch device this process should pin tensors to (CUDA index = local_rank, or CPU).
    **Input parameter 6:** backend - The torch.distributed backend in use ("nccl" or "gloo"); ``None`` if distributed is disabled.

    Output
    ------
    Output returned: A frozen dataclass instance representing the state of distributed training.

    Purpose
    -------
    Immutable bundle of distributed-training metadata, populated once by ``_init_dist_state`` and consumed throughout ``main``. Frozen so it cannot accidentally be mutated mid-run.

    Assumptions
    -----------
    Designed for both single-process (``enabled=False, world_size=1``) and multi-process (``torchrun``) execution. The single-process path does not call ``dist.init_process_group``, so ``backend`` stays ``None``.

    Notes
    -----
    ``is_main_process`` is provided as a property for code clarity ("if the main process" reads better than "if rank == 0").
    """

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None = None

    @property
    def is_main_process(self) -> bool:
        """
        Parameters
        ----------
        (No input parameters; reads ``self.rank``.)

        Output
        ------
        Output returned: A bool. ``True`` only on the rank-0 process; ``False`` on all other ranks.

        Purpose
        -------
        Convenience predicate guarding all "main-process-only" side effects (logging, checkpoint writing, W&B init).

        Assumptions
        -----------
        Designed to be called after ``DistState`` has been populated by ``_init_dist_state``.

        Notes
        -----
        Equivalent to ``self.rank == 0``; provided as a property only to make call sites read like English.
        """
        return self.rank == 0


@dataclass
class WandbState:
    """
    Parameters
    ----------
    **Input parameter 1:** enabled - True if W&B logging is active for this process (only on rank 0 with ``--wandb-mode != disabled``).
    **Input parameter 2:** run - The wandb ``Run`` instance returned by ``wandb.init`` (kept as ``Any`` so we don't import wandb at module import time).
    **Input parameter 3:** run_id - The W&B run id, captured for resume support.

    Output
    ------
    Output returned: A dataclass instance bundling W&B state for clean shutdown in the trainer's ``finally`` block.

    Purpose
    -------
    Mutable bundle so the trainer can either log to W&B (offline by default) or skip W&B entirely without scattering ``if wandb is not None`` checks throughout ``main``.

    Assumptions
    -----------
    Designed for ``--wandb-mode offline`` by default so no API key or network access is required. ``--wandb-mode online`` and ``--wandb-mode disabled`` are also supported.

    Notes
    -----
    ``enabled=False`` is the default so any process that hasn't initialized W&B (including all non-rank-0 processes) is a no-op for the W&B-related code paths.
    """

    enabled: bool = False
    run: Any | None = None
    run_id: str | None = None


class StridedShardSampler(Sampler[int]):
    """
    Parameters
    ----------
    **Input parameter 1:** dataset_size - Number of items in the underlying dataset.
    **Input parameter 2:** rank - Rank of this process; receives indices ``rank, rank + world_size, rank + 2 * world_size, ...``.
    **Input parameter 3:** world_size - Total process count across which the dataset is sharded.

    Output
    ------
    Output returned: An iterable ``Sampler`` that yields a strided shard of dataset indices, one shard per rank.

    Purpose
    -------
    Lightweight DDP sampler for the validation loader. Unlike ``DistributedSampler``, this does not pad the last batch and does not shuffle, so the same validation example is always assigned to the same rank across epochs. That makes the rank-summed validation loss reproducible across runs.

    Assumptions
    -----------
    Designed for read-only validation loaders where deterministic, non-padded sharding is preferred to perfect load-balancing. Train loaders should still use ``DistributedSampler`` because they need shuffling and even-batch behavior for AdamW.

    Notes
    -----
    ``__len__`` returns the exact (potentially uneven) shard size so the DataLoader's progress bar is accurate per rank.
    """

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
    """
    Parameters
    ----------
    (No input parameters; reads from ``sys.argv`` via ``argparse``.)

    Output
    ------
    Output returned: An ``argparse.Namespace`` with all CLI-configurable VideoMAE training options (dataset paths, optimizer / scheduler hyperparameters, encoder config, mask ratio, tube size, AMP / determinism flags, resume support, W&B knobs).

    Purpose
    -------
    Define the full CLI for ``python -m videomae.train_videomae``. Defaults are tuned for our active_matter setup on B200: BF16 autocast, mask_ratio=0.6, tube_size=(16, 32, 32), encoder dims/blocks matching Person A's SIGReg-JEPA so the SSL-objective comparison is causally clean.

    Assumptions
    -----------
    Designed to be called once at the start of ``main`` (before ``_init_dist_state``). Strict invariants (``mask_ratio`` in (0, 1), ``warmup_epochs >= 0``, ``warmup_start_factor`` in (0, 1]) are enforced in ``main`` after calling this function.

    Notes
    -----
    ``--wandb-mode`` defaults to ``offline`` so the trainer runs without an API key by default; the ``wandb sync`` post-hoc workflow can upload runs later if desired.
    """
    parser = argparse.ArgumentParser(description="Train a VideoMAE/SimMIM-hybrid encoder on active_matter from scratch.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path containing train/valid/test or data/train/valid/test.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/videomae"), help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1.5e-6)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--warmup-start-factor", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Dataset / clip indexing
    parser.add_argument("--num-frames", type=int, default=16)
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
    parser.add_argument("--resolution", type=int, default=224)

    # Encoder config (matches colleague's SigRegJepaModel defaults for fair comparison)
    parser.add_argument("--dims", type=str, default="32,64,128,256,256")
    parser.add_argument("--num-res-blocks", type=str, default="2,2,4,8,2")
    parser.add_argument("--stem-patch-size", type=int, default=2)
    parser.add_argument("--stem-kernel-size", type=int, default=4)
    parser.add_argument("--drop-path-rate", type=float, default=0.05)
    parser.add_argument("--layer-scale-init-value", type=float, default=1e-6)

    # VideoMAE / SimMIM-specific
    parser.add_argument("--mask-ratio", type=float, default=0.60)
    parser.add_argument("--tube-size", type=str, default="16,32,32", help="(time, height, width) tube extent.")
    parser.add_argument("--norm-pix-loss", action="store_true", default=True, help="Per-tube z-score the reconstruction target.")
    parser.add_argument("--no-norm-pix-loss", dest="norm_pix_loss", action="store_false")

    # Misc
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True, help="Enable autocast on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Warm-start the encoder + decoder from a previous checkpoint without resuming optimizer state.")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from a full training checkpoint or 'auto' for out-dir/last.pt.")
    parser.add_argument("--save-every", type=int, default=2,
                        help="Write numbered full checkpoints every N epochs.")
    parser.add_argument("--dist-backend", choices=["nccl", "gloo"], default=None)

    # Experiment tracking
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline",
                        help="Default: offline, no account needed; runs sync later via 'wandb sync'.")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="dl-final-videomae")
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
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29500")


def _init_dist_state(device_arg: str, dist_backend: str | None) -> DistState:
    _slurm_env_fallback()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if not enabled:
        return DistState(
            enabled=False, rank=0, world_size=1, local_rank=0,
            device=choose_device(device_arg), backend=None,
        )
    backend = dist_backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return DistState(
        enabled=True, rank=rank, world_size=world_size, local_rank=local_rank,
        device=device, backend=backend,
    )


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


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
        return WandbState(enabled=False)
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


def _load_init_checkpoint(model: VideoMAEModel, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    summary: dict[str, Any] = {"path": str(path), "loaded_encoder": False, "loaded_decoder": False, "loaded_mask_token": False}
    if "encoder" in payload:
        model.encoder.load_state_dict(payload["encoder"])
        summary["loaded_encoder"] = True
    elif "state_dict" in payload:
        model.encoder.load_state_dict(payload["state_dict"])
        summary["loaded_encoder"] = True
    if "decoder" in payload:
        try:
            model.decoder.load_state_dict(payload["decoder"])
            summary["loaded_decoder"] = True
        except RuntimeError:
            pass
    if "mask_token" in payload:
        try:
            with torch.no_grad():
                model.mask_token.copy_(payload["mask_token"])
            summary["loaded_mask_token"] = True
        except Exception:
            pass
    if not summary["loaded_encoder"]:
        raise ValueError(f"init checkpoint does not contain encoder weights: {path}")
    summary["epoch"] = payload.get("epoch")
    return summary


def main() -> None:
    """
    Parameters
    ----------
    (No input parameters; CLI args come from ``parse_args``.)

    Output
    ------
    Output returned: ``None``. Side effect: trains a VideoMAE encoder, writes per-epoch checkpoints (``last.pt``, ``encoder_last.pt``, optionally ``epoch_NNNN.pt`` and ``best.pt`` / ``encoder_best.pt``) and JSON logs (``train_config.json``, ``history.json``) under ``--out-dir``, plus optional W&B-offline run files.

    Purpose
    -------
    End-to-end VideoMAE / SimMIM-hybrid training loop. Builds the model from CLI args, sets up DDP if launched via torchrun, fits ``AdamW`` with cosine LR and linear warmup, runs train + valid passes per epoch under BF16 autocast, and saves checkpoints idempotently. Supports both clean starts and resume-from-``last.pt`` so SSH drops never lose progress.

    Assumptions
    -----------
    Designed for B200 / A100 GPUs with BF16 autocast; FP16 with grad-scaling is supported via ``--amp-dtype float16`` (rarely needed). Single-process training is the default; multi-GPU DDP activates automatically when launched under ``torchrun`` (detects ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` env vars or SLURM equivalents). Hard-asserts encoder/decoder total parameters stay under the assignment's 100M cap.

    Notes
    -----
    Encoder weights are saved separately as ``encoder_last.pt`` / ``encoder_best.pt`` in the format ``physrl/export_embeddings.py`` consumes (``state_dict`` + ``config``), so the colleague's eval pipeline works on our checkpoints unchanged. Validation is performed on every epoch; "best" is determined by validation loss. If ``--resume auto`` finds a complete previous run (start_epoch > epochs), the trainer prints a status line and exits cleanly without re-training.
    """
    dist_state: DistState | None = None
    wandb_state = WandbState(enabled=False)
    try:
        args = parse_args()
        dims = parse_int_list(args.dims)
        blocks = parse_int_list(args.num_res_blocks)
        if len(dims) != len(blocks):
            raise SystemExit("--dims and --num-res-blocks must have the same number of entries")
        tube_size = parse_int_list(args.tube_size)
        if len(tube_size) != 3:
            raise SystemExit("--tube-size must have 3 comma-separated integers (T,H,W)")
        if not (0.0 < args.mask_ratio < 1.0):
            raise SystemExit("--mask-ratio must be in (0, 1)")
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
            context_frames=args.num_frames,
            target_frames=0,  # VideoMAE only needs one window per sample
            stride=args.train_stride,
            resolution=args.resolution,
            max_samples=args.max_train_samples,
            index_mode=args.train_index_mode,
            clip_selection=args.clip_selection,
        )
        valid_dataset = ActiveMatterWindowDataset(
            root=args.data_root,
            split="valid",
            context_frames=args.num_frames,
            target_frames=0,
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

        model = VideoMAEModel(
            in_chans=sum(field.channels for field in train_dataset.field_specs),
            dims=dims,
            num_res_blocks=blocks,
            num_frames=args.num_frames,
            stem_patch_size=args.stem_patch_size,
            stem_kernel_size=args.stem_kernel_size,
            drop_path_rate=args.drop_path_rate,
            layer_scale_init_value=args.layer_scale_init_value,
            mask_ratio=args.mask_ratio,
            tube_size=tuple(tube_size),
            norm_pix_loss=args.norm_pix_loss,
        ).to(dist_state.device)

        # Hard cap enforcement: assignment requires <100M parameters total.
        total_params = _count_params(model)
        if total_params >= 100_000_000:
            raise SystemExit(f"Total params {total_params:,} >= 100M cap")

        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = min(max(0, int(args.warmup_epochs * len(train_loader))), max(total_steps - 1, 0))

        config_payload = vars(args).copy()
        config_payload["dims"] = dims
        config_payload["num_res_blocks"] = blocks
        config_payload["tube_size"] = list(tube_size)
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
        config_payload["total_params"] = total_params
        config_payload["encoder_params"] = _count_params(model.encoder)
        config_payload["decoder_params"] = _count_params(model.decoder)
        # context_frames is what export_embeddings.py looks for
        config_payload["context_frames"] = args.num_frames

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.warmup_start_factor, end_factor=1.0, total_iters=warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.min_lr,
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=args.min_lr,
            )
        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=args.amp and dist_state.device.type == "cuda" and amp_dtype == torch.float16,
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
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.encoder.load_state_dict(resume_payload["encoder"])
            model.decoder.load_state_dict(resume_payload["decoder"])
            with torch.no_grad():
                model.mask_token.copy_(resume_payload["mask_token"])
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            scaler.load_state_dict(resume_payload["scaler"])
            best_val = float(resume_payload.get("best_valid_loss", best_val))
            history = list(resume_payload.get("history", []))
            start_epoch = int(resume_payload.get("epoch", 0)) + 1
            if dist_state.is_main_process:
                print({"status": "resumed", "resume_path": str(resume_path), "next_epoch": start_epoch}, flush=True)
        elif args.init_checkpoint is not None:
            init_path = args.init_checkpoint.expanduser().resolve()
            if not init_path.exists():
                raise FileNotFoundError(f"init checkpoint not found: {init_path}")
            init_summary = _load_init_checkpoint(model, init_path)
            config_payload["init_checkpoint"] = init_summary["path"]
            if dist_state.is_main_process:
                print({"status": "initialized", **init_summary}, flush=True)

        if dist_state.is_main_process:
            save_json(out_dir / "train_config.json", config_payload)

        if start_epoch > args.epochs:
            if dist_state.is_main_process:
                print({"status": "already_complete", "requested_epochs": args.epochs, "checkpoint_epoch": start_epoch - 1}, flush=True)
            return

        wandb_state = _init_wandb(
            args=args, dist_state=dist_state, config_payload=config_payload,
            out_dir=out_dir, resume_payload=resume_payload,
        )

        if dist_state.enabled:
            ddp_kwargs = dict(broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)
            if dist_state.device.type == "cuda":
                model = DDP(model, device_ids=[dist_state.device.index], output_device=dist_state.device.index, **ddp_kwargs)
            else:
                model = DDP(model, **ddp_kwargs)

        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            model.train()
            train_loss_sum = 0.0
            train_sample_count = 0

            for batch_idx, batch in enumerate(train_loader):
                clip = batch["context"].to(dist_state.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=dist_state.device.type,
                    dtype=amp_dtype,
                    enabled=args.amp and dist_state.device.type == "cuda",
                ):
                    out = model(clip)
                    loss = out["loss"]

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
                train_sample_count += bs
                train_loss_sum += float(loss.item()) * bs

            train_loss_avg = _all_reduce_mean(
                train_loss_sum, train_sample_count, device=dist_state.device, distributed=dist_state.enabled,
            )

            model.eval()
            valid_loss_sum = 0.0
            valid_sample_count = 0
            with torch.no_grad():
                for batch in valid_loader:
                    clip = batch["context"].to(dist_state.device, non_blocking=True)
                    with torch.autocast(
                        device_type=dist_state.device.type,
                        dtype=amp_dtype,
                        enabled=args.amp and dist_state.device.type == "cuda",
                    ):
                        out = model(clip)
                        loss = out["loss"]
                    bs = int(clip.shape[0])
                    valid_sample_count += bs
                    valid_loss_sum += float(loss.item()) * bs
            valid_loss_avg = _all_reduce_mean(
                valid_loss_sum, valid_sample_count, device=dist_state.device, distributed=dist_state.enabled,
            )

            elapsed = time.time() - start_time
            epoch_record = {
                "epoch": epoch,
                "seconds": round(elapsed, 2),
                "train_loss": train_loss_avg,
                "valid_loss": valid_loss_avg,
                "lr": optimizer.param_groups[0]["lr"],
                "world_size": dist_state.world_size,
            }
            history.append(epoch_record)
            if dist_state.is_main_process:
                print(epoch_record, flush=True)
                if wandb_state.enabled and wandb_state.run is not None:
                    wandb_state.run.log(epoch_record, step=epoch)

            improved = valid_loss_avg < best_val
            if improved:
                best_val = valid_loss_avg

            if dist_state.is_main_process:
                raw_model = _unwrap_model(model)
                checkpoint = {
                    "epoch": epoch,
                    "config": config_payload,
                    "encoder": raw_model.encoder.state_dict(),
                    "decoder": raw_model.decoder.state_dict(),
                    "mask_token": raw_model.mask_token.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_valid_loss": best_val,
                    "history": history,
                    "wandb_run_id": wandb_state.run_id,
                }
                atomic_torch_save(out_dir / "last.pt", checkpoint)
                # encoder_last.pt: format compatible with physrl/export_embeddings.py
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


def _all_reduce_mean(
    sum_value: float, count: int, *, device: torch.device, distributed: bool,
) -> float:
    payload = torch.tensor([float(sum_value), float(count)], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    total = payload[1].item()
    if total <= 0:
        return float("nan")
    return float(payload[0].item() / total)


if __name__ == "__main__":
    main()
