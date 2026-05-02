"""Reconstruction visualizations for the VideoMAE encoders.

For each VideoMAE checkpoint and a small set of sample clips, produces a
figure with rows = (input, masked, reconstructed, residual) and columns =
the 11 physical channels at a single chosen time step.

Visually shows what the SSL objective learns to reconstruct vs.\ ignore --
ties directly to the channel-importance ablation (which input channels are
informative for downstream MSE) and to the per-channel reconstruction MSE
analysis (which output channels are reconstructed sharply vs.\ smudged).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import ActiveMatterWindowDataset
from .models import VideoMAEModel
from .utils import (
    choose_device,
    configure_torch_runtime,
    ensure_dir,
    parse_int_list,
    seed_everything,
)


CHANNEL_NAMES = [
    "concentration", "vel_x", "vel_y",
    "orient_xx", "orient_xy", "orient_yx", "orient_yy",
    "strain_xx", "strain_xy", "strain_yx", "strain_yy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruction visualizations for VideoMAE encoders.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                        help="One or more last.pt or best.pt VideoMAE checkpoints (with encoder + decoder + mask_token).")
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=2, help="Number of validation clips to visualize per encoder.")
    parser.add_argument("--time-step", type=int, default=8, help="Which of the 16 frames to render.")
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
        raise SystemExit(f"checkpoint {checkpoint_path} is not a VideoMAE last.pt (need encoder+decoder+mask_token)")
    model.encoder.load_state_dict(payload["encoder"])
    model.decoder.load_state_dict(payload["decoder"])
    with torch.no_grad():
        model.mask_token.copy_(payload["mask_token"])
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    configure_torch_runtime(deterministic=False)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        raise SystemExit("--labels must match --checkpoints in length")
    labels = args.labels or [p.parent.name for p in args.checkpoints]

    # Take a fixed subset of validation samples for fair comparison
    valid_ds = ActiveMatterWindowDataset(
        root=args.data_root, split="valid", context_frames=16, target_frames=0,
        stride=1, resolution=224, max_samples=None,
        index_mode="single_clip", clip_selection="center",
    )
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(len(valid_ds), size=args.n_samples, replace=False).tolist()
    samples = []
    for idx in sample_indices:
        item = valid_ds[idx]
        samples.append({"clip": item["context"], "label": item["label"], "idx": int(idx)})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    for ckpt_path, label in zip(args.checkpoints, labels):
        print(f"[recon] {label}: loading {ckpt_path}", flush=True)
        model = _load_videomae(ckpt_path, device)
        for s in samples:
            clip_t = s["clip"].unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    out = model(clip_t)
            target = out["target"][0].float().cpu().numpy()  # (C,T,H,W) per-tube z-scored
            recon = out["recon"][0].float().cpu().numpy()
            pixel_mask = out["pixel_mask"][0, 0].float().cpu().numpy()  # (T,H,W)
            raw = clip_t[0].float().cpu().numpy()  # (C,T,H,W) raw
            # Build the masked-input visualization at the chosen time step:
            t = int(args.time_step)
            label_str = f"alpha={s['label'][0].item():.2f}, zeta={s['label'][1].item():.2f}"

            fig, axes = plt.subplots(4, 11, figsize=(2.0 * 11, 2.0 * 4))
            for c in range(11):
                # Row 0: raw input (target before normalization)
                ax = axes[0, c]
                im = ax.imshow(raw[c, t], cmap="viridis")
                ax.set_title(CHANNEL_NAMES[c], fontsize=7)
                ax.axis("off")
                # Row 1: masked input (raw with masked tubes set to mask token)
                ax = axes[1, c]
                masked_input = raw[c, t].copy()
                # mask token value is just a learned scalar per channel
                mask_val = float(model.mask_token[0, c, 0, 0, 0].item())
                masked_input[pixel_mask[t] > 0.5] = mask_val
                ax.imshow(masked_input, cmap="viridis")
                ax.axis("off")
                # Row 2: reconstruction (per-tube z-scored space, but display as-is)
                ax = axes[2, c]
                ax.imshow(recon[c, t], cmap="viridis")
                ax.axis("off")
                # Row 3: residual = recon - target (in z-scored space)
                ax = axes[3, c]
                resid = recon[c, t] - target[c, t]
                vmax = max(abs(resid.min()), abs(resid.max()), 1e-6)
                ax.imshow(resid, cmap="seismic", vmin=-vmax, vmax=vmax)
                ax.axis("off")

            axes[0, 0].set_ylabel("input", fontsize=8, rotation=0, ha="right", labelpad=20)
            axes[1, 0].set_ylabel("masked", fontsize=8, rotation=0, ha="right", labelpad=20)
            axes[2, 0].set_ylabel("recon", fontsize=8, rotation=0, ha="right", labelpad=20)
            axes[3, 0].set_ylabel("resid", fontsize=8, rotation=0, ha="right", labelpad=20)
            fig.suptitle(f"{label} - sample idx={s['idx']}, frame t={t}; {label_str}", fontsize=10)
            fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
            pdf_path = out_dir / f"recon_{label}_sample{s['idx']}_t{t}.pdf"
            fig.savefig(pdf_path, dpi=120)
            plt.close(fig)
            print(f"[recon] {label} sample {s['idx']} -> {pdf_path}", flush=True)
        del model
        torch.cuda.empty_cache()
    print({"out_dir": str(out_dir), "n_encoders": len(labels), "n_samples": len(samples)})


if __name__ == "__main__":
    main()
