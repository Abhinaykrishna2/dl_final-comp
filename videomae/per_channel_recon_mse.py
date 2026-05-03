"""Per-channel reconstruction MSE for VideoMAE encoders.

For each VideoMAE checkpoint and each of the 11 physical channels, computes
the MSE between the per-tube z-scored target and the decoder reconstruction
on masked positions, on a held-out split (default: valid). This complements
the channel-importance ablation: that one says "which input channels are
informative for downstream MSE", this one says "which output channels does
VideoMAE actually try to reconstruct".

Saves a JSON with per-channel masked-position MSE per encoder, plus an
overlay bar chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ActiveMatterWindowDataset
from .models import VideoMAEModel
from .utils import (
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    make_torch_generator,
    parse_int_list,
    save_json,
    seed_everything,
    seed_worker,
)


CHANNEL_NAMES = [
    "concentration", "vel_x", "vel_y",
    "orient_xx", "orient_xy", "orient_yx", "orient_yy",
    "strain_xx", "strain_xy", "strain_yx", "strain_yy",
]


def parse_args() -> argparse.Namespace:
    """
    Parameters
    ----------
    (No input parameters; reads from ``sys.argv`` via ``argparse``.)

    Output
    ------
    Output returned: An ``argparse.Namespace`` with the parsed CLI options listed below.

    Purpose
    -------
    Define the CLI for ``python -m videomae.per_channel_recon_mse``: data root, one or more VideoMAE checkpoints (with optional matching ``--labels``), output directory, split, mask realization count, batch size, num workers, seed, device, AMP dtype.

    Assumptions
    -----------
    Designed to be called once at the start of ``main``. Each checkpoint must be a VideoMAE ``last.pt`` or ``best.pt`` that contains ``encoder``, ``decoder``, and ``mask_token`` keys; pure ``encoder_best.pt`` files cannot be reused here.

    Notes
    -----
    ``--n-mask-realizations`` defaults to 4: this averages the per-channel MSE across 4 fresh tube masks per sample, reducing the noise floor enough to compare channels reliably.
    """
    parser = argparse.ArgumentParser(description="Per-channel reconstruction MSE for VideoMAE.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                        help="VideoMAE last.pt or best.pt files (need encoder + decoder + mask_token).")
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-mask-realizations", type=int, default=4,
                        help="Number of random mask draws per sample (averaged).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def _load_videomae(checkpoint_path: Path, device: torch.device) -> VideoMAEModel:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    dims = parse_int_list(cfg["dims"]) if isinstance(cfg["dims"], str) else list(cfg["dims"])
    blocks = parse_int_list(cfg["num_res_blocks"]) if isinstance(cfg["num_res_blocks"], str) else list(cfg["num_res_blocks"])
    tube_size = cfg.get("tube_size", [16, 32, 32])
    if isinstance(tube_size, str):
        tube_size = parse_int_list(tube_size)
    model = VideoMAEModel(
        in_chans=int(cfg.get("in_chans", 11)),
        dims=dims, num_res_blocks=blocks,
        num_frames=int(cfg.get("context_frames", cfg.get("num_frames", 16))),
        stem_patch_size=int(cfg.get("stem_patch_size", 2)),
        stem_kernel_size=int(cfg.get("stem_kernel_size", 4)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.05)),
        layer_scale_init_value=float(cfg.get("layer_scale_init_value", 1e-6)),
        mask_ratio=float(cfg.get("mask_ratio", 0.6)),
        tube_size=tuple(tube_size),
        norm_pix_loss=bool(cfg.get("norm_pix_loss", True)),
    )
    if "encoder" not in payload or "decoder" not in payload:
        raise SystemExit(f"checkpoint {checkpoint_path} not a VideoMAE last.pt")
    model.encoder.load_state_dict(payload["encoder"])
    model.decoder.load_state_dict(payload["decoder"])
    with torch.no_grad():
        model.mask_token.copy_(payload["mask_token"])
    model.to(device).eval()
    return model


def main() -> None:
    """
    Parameters
    ----------
    (No input parameters; CLI args come from ``parse_args``.)

    Output
    ------
    Output returned: ``None``. Side effect: writes ``per_channel_recon_mse.json`` (per-encoder per-channel masked MSE) and ``per_channel_recon_mse.pdf`` (overlay bar chart) to the output directory and prints per-channel MSE values to stdout.

    Purpose
    -------
    For each VideoMAE checkpoint and each of the 11 physical channels, compute the masked-position reconstruction MSE on a held-out split, averaged over ``--n-mask-realizations`` random tube masks per sample. Complements the channel-importance ablation (``run_ablations.py``): that one says "which input channels are informative for downstream MSE", this one says "which output channels does VideoMAE actually try to reconstruct".

    Assumptions
    -----------
    Designed for VideoMAE checkpoints saved by ``train_videomae.py``. Reads the val split by default; enforces 16-frame center clips at 224x224 (the training-time clip geometry). Multiple checkpoints can be evaluated in one call; their plots overlay on the same figure.

    Notes
    -----
    The squared error is reduced over the batch and spatiotemporal axes but kept per-channel; the mask weight (number of masked pixels) is also tracked per-channel so the final MSE is correctly normalized. ``torch.cuda.empty_cache`` is called between encoders so peak memory stays bounded across many checkpoints.
    """
    args = parse_args()
    seed_everything(args.seed)
    configure_torch_runtime(deterministic=False)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        raise SystemExit("--labels must match --checkpoints in length")
    labels = args.labels or [p.parent.name for p in args.checkpoints]

    dataset = ActiveMatterWindowDataset(
        root=args.data_root, split=args.split, context_frames=16, target_frames=0,
        stride=1, resolution=224, max_samples=args.max_samples,
        index_mode="single_clip", clip_selection="center",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=make_torch_generator(args.seed),
    )

    summary: dict[str, dict] = {"split": args.split, "n_samples": len(dataset), "encoders": {}}
    per_encoder_per_channel: dict[str, np.ndarray] = {}

    for ckpt_path, label in zip(args.checkpoints, labels):
        print(f"[recon-mse] {label} from {ckpt_path}", flush=True)
        model = _load_videomae(ckpt_path, device)
        # Sum of squared error per channel + sum of mask weight per channel
        sse = torch.zeros(11, dtype=torch.float64, device=device)
        msum = torch.zeros(11, dtype=torch.float64, device=device)
        with torch.no_grad():
            for batch in loader:
                clip = batch["context"].to(device, non_blocking=True)
                for r in range(args.n_mask_realizations):
                    # Force a different RNG state per realization, so masks differ
                    torch.manual_seed(args.seed + r + hash(label) % 10_000)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(args.seed + r + hash(label) % 10_000)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                        out = model(clip)
                    target = out["target"].to(torch.float64)  # (B,C,T,H,W)
                    recon = out["recon"].to(torch.float64)
                    # pixel_mask: (B,1,T,H,W) bool
                    mask = out["pixel_mask"].to(torch.float64)
                    sq = (recon - target) ** 2
                    # Reduce over (B,T,H,W) but keep channel
                    sq_chan = (sq * mask).sum(dim=(0, 2, 3, 4))
                    m_chan = mask.sum(dim=(0, 2, 3, 4)).clamp_min(1.0)
                    sse += sq_chan
                    msum += m_chan.expand_as(sq_chan)
        per_channel_mse = (sse / msum).cpu().numpy()
        per_encoder_per_channel[label] = per_channel_mse
        rec = {
            "checkpoint": str(ckpt_path),
            "per_channel_masked_recon_mse": {CHANNEL_NAMES[c]: float(per_channel_mse[c]) for c in range(11)},
            "mean_masked_recon_mse": float(per_channel_mse.mean()),
        }
        summary["encoders"][label] = rec
        print(f"[recon-mse] {label} mean = {rec['mean_masked_recon_mse']:.4f}", flush=True)
        for c in range(11):
            print(f"  {CHANNEL_NAMES[c]:>14}: {per_channel_mse[c]:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    save_json(out_dir / "per_channel_recon_mse.json", summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None and per_encoder_per_channel:
        n = len(per_encoder_per_channel)
        width = 0.8 / n
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(11)
        for i, (label, vals) in enumerate(per_encoder_per_channel.items()):
            ax.bar(x + i * width, vals, width=width, label=label)
        ax.set_xticks(x + (n - 1) * width / 2)
        ax.set_xticklabels(CHANNEL_NAMES, rotation=35, ha="right", fontsize="small")
        ax.set_ylabel("masked-position recon MSE (per-tube z-scored target)")
        ax.set_title(f"VideoMAE per-channel reconstruction MSE (split={args.split})")
        ax.legend(fontsize="small")
        fig.tight_layout()
        pdf_path = out_dir / "per_channel_recon_mse.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print({"pdf": str(pdf_path)}, flush=True)


if __name__ == "__main__":
    main()
