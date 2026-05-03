from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ActiveMatterWindowDataset
from .models import CNextUNetForecaster
from .utils import (
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    load_torch_checkpoint,
    make_torch_generator,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen CNext-U-Net encoder embeddings for train/valid/test.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to encoder_best.pt or encoder_last.pt.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/cnext_embeddings"))
    parser.add_argument("--split", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--dataset-stride", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--pool", choices=["avg", "avgmax", "flatten"], default="avgmax")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--index-mode",
        choices=["single_clip", "sliding_window"],
        default="single_clip",
        help="Default keeps one deterministic 16-frame clip per simulation for final probing.",
    )
    parser.add_argument(
        "--clip-selection",
        choices=["start", "center", "end"],
        default="center",
    )
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[CNextUNetForecaster, dict]:
    payload = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    config = payload["config"]
    if payload.get("model_type", config.get("model_type")) != "cnext_unet_forecaster":
        raise ValueError(f"checkpoint is not a CNext-U-Net forecaster checkpoint: {checkpoint_path}")

    model = CNextUNetForecaster(
        field_channels=int(config.get("field_channels", config.get("in_chans", 11))),
        context_frames=int(config.get("context_frames", 4)),
        target_frames=int(config.get("target_frames", 1)),
        init_features=int(config.get("init_features", 42)),
        stages=int(config.get("stages", 4)),
        blocks_per_stage=int(config.get("blocks_per_stage", 2)),
        blocks_at_neck=int(config.get("blocks_at_neck", 1)),
        drop_path_rate=float(config.get("drop_path_rate", 0.0)),
        layer_scale_init_value=float(config.get("layer_scale_init_value", 1e-6)),
        gradient_checkpointing=False,
    )
    state_dict = payload["model"] if "model" in payload else payload["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def _pool_features(features: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "avg":
        return features.mean(dim=(-1, -2))
    if pool == "avgmax":
        avg = features.mean(dim=(-1, -2))
        maxv = features.amax(dim=(-1, -2))
        return torch.cat([avg, maxv], dim=1)
    if pool == "flatten":
        return features.flatten(1)
    raise ValueError(f"unsupported pool: {pool}")


def main() -> None:
    args = parse_args()
    if args.window_stride < 1:
        raise SystemExit("--window-stride must be >= 1")
    if args.dataset_stride < 1:
        raise SystemExit("--dataset-stride must be >= 1")
    if args.index_mode != "sliding_window" and args.dataset_stride != 1:
        raise SystemExit("--dataset-stride only applies when --index-mode sliding_window")
    seed_everything(args.seed)
    configure_torch_runtime(deterministic=args.deterministic)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    model, train_config = _load_model(args.checkpoint, device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    resolution = int(train_config.get("resolution", 96)) if args.resolution is None else args.resolution

    if args.clip_frames < model.context_frames:
        raise SystemExit(f"--clip-frames must be at least model context_frames={model.context_frames}")

    for split in args.split:
        dataset = ActiveMatterWindowDataset(
            root=args.data_root,
            split=split,
            context_frames=args.clip_frames,
            target_frames=0,
            stride=args.dataset_stride,
            resolution=resolution,
            max_samples=args.max_samples,
            index_mode=args.index_mode,
            clip_selection=args.clip_selection,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            generator=make_torch_generator(args.seed),
        )

        all_embeddings: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                clip = batch["context"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=args.amp and device.type == "cuda",
                ):
                    features = model.encode_clip_windows(
                        clip,
                        window_frames=model.context_frames,
                        stride=args.window_stride,
                    )
                    pooled = _pool_features(features, args.pool)
                all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
                all_labels.append(batch["label"].numpy().astype(np.float32))

        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        out_path = out_dir / f"{split}.npz"
        np.savez_compressed(
            out_path,
            embeddings=embeddings,
            labels=labels,
            split=np.asarray(split),
            pool=np.asarray(args.pool),
            index_mode=np.asarray(args.index_mode),
            clip_selection=np.asarray(args.clip_selection),
            clip_frames=np.asarray(args.clip_frames),
            window_stride=np.asarray(args.window_stride),
            dataset_stride=np.asarray(args.dataset_stride),
            train_checkpoint=np.asarray(str(args.checkpoint)),
            train_config=np.asarray(str(train_config)),
        )
        print(
            {
                "split": split,
                "samples": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "pool": args.pool,
                "clip_frames": args.clip_frames,
                "window_stride": args.window_stride,
                "out_path": str(out_path),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
