from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ActiveMatterWindowDataset
from .models import ConvEncoder
from .utils import (
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    make_torch_generator,
    parse_int_list,
    pool_features,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen JEPA embeddings for train/valid/test.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best.pt, last.pt, encoder_best.pt, or encoder_last.pt from JEPA training.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/embeddings"))
    parser.add_argument("--split", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic PyTorch behavior where feasible.")
    parser.add_argument("--amp", action="store_true", help="Use mixed-precision inference on CUDA during export.")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Autocast dtype used when --amp is enabled.",
    )
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--pool", type=str, choices=["avg", "flatten", "avgmax"], default="avg")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--index-mode",
        choices=["single_clip", "sliding_window"],
        default="single_clip",
        help="How to convert raw trajectories into encoder inputs. Default keeps one deterministic clip per simulation.",
    )
    parser.add_argument(
        "--clip-selection",
        choices=["start", "center", "end"],
        default="center",
        help="Clip position used when --index-mode single_clip.",
    )
    return parser.parse_args()


def _load_encoder(checkpoint_path: Path, device: torch.device) -> tuple[ConvEncoder, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = payload["config"]
    dims = parse_int_list(config["dims"]) if isinstance(config["dims"], str) else list(config["dims"])
    blocks = (
        parse_int_list(config["num_res_blocks"])
        if isinstance(config["num_res_blocks"], str)
        else list(config["num_res_blocks"])
    )
    in_chans = int(config.get("in_chans", 11))
    if "context_frames" in config:
        num_frames = int(config["context_frames"])
    else:
        num_frames = 16

    encoder = ConvEncoder(
        in_chans=in_chans,
        dims=dims,
        num_res_blocks=blocks,
        num_frames=num_frames,
        stem_patch_size=int(config.get("stem_patch_size", 1)),
        stem_kernel_size=int(config.get("stem_kernel_size", 4)),
        drop_path_rate=float(config.get("drop_path_rate", 0.0)),
        layer_scale_init_value=float(config.get("layer_scale_init_value", 1e-6)),
    )
    state_dict = payload["encoder"] if "encoder" in payload else payload["state_dict"]
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    return encoder, config


def main() -> None:
    args = parse_args()
    if args.index_mode != "sliding_window" and args.stride != 1:
        raise SystemExit("--stride only applies when --index-mode sliding_window")
    seed_everything(args.seed)
    configure_torch_runtime(deterministic=args.deterministic)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    encoder, train_config = _load_encoder(args.checkpoint, device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    context_frames = int(train_config.get("context_frames", 16)) if args.context_frames is None else args.context_frames
    resolution = int(train_config.get("resolution", 224)) if args.resolution is None else args.resolution

    for split in args.split:
        dataset = ActiveMatterWindowDataset(
            root=args.data_root,
            split=split,
            context_frames=context_frames,
            target_frames=0,
            stride=args.stride,
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
                context = batch["context"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=args.amp and device.type == "cuda",
                ):
                    features = encoder(context)
                pooled = pool_features(features, args.pool)
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
            train_checkpoint=np.asarray(str(args.checkpoint)),
            train_config=np.asarray(str(train_config)),
        )
        print(
            {
                "split": split,
                "samples": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "index_mode": args.index_mode,
                "clip_selection": args.clip_selection,
                "out_path": str(out_path),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
