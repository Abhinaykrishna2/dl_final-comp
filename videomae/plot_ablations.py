"""Plot channel-importance and frame-budget ablations from the JSON output of run_ablations.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation results.")
    parser.add_argument("--ablations-dir", type=Path, required=True,
                        help="Directory containing per-encoder subdirectories with channel_ablation.json and frame_ablation.json.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <ablations-dir>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablations_dir = args.ablations_dir.expanduser().resolve()
    out_dir = (args.out_dir or ablations_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    encoders: dict[str, dict] = {}
    for sub in sorted(ablations_dir.iterdir()):
        if not sub.is_dir():
            continue
        ch_path = sub / "channel_ablation.json"
        fr_path = sub / "frame_ablation.json"
        if not ch_path.exists() and not fr_path.exists():
            continue
        encoders[sub.name] = {}
        if ch_path.exists():
            with open(ch_path, "r", encoding="utf-8") as f:
                encoders[sub.name]["channel"] = json.load(f)
        if fr_path.exists():
            with open(fr_path, "r", encoding="utf-8") as f:
                encoders[sub.name]["frame"] = json.load(f)

    if not encoders:
        raise SystemExit(f"no ablation data found under {ablations_dir}")

    # 1. Channel ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(encoders))
    channel_names: list[str] | None = None
    for i, (label, payload) in enumerate(encoders.items()):
        ch = payload.get("channel")
        if ch is None:
            continue
        per_channel = ch["per_channel"]
        baseline_mse = ch["baseline"]["test_normalized_mean_mse"]
        # Skip the baseline row in per_channel (channel == -1)
        deltas = [r["test_normalized_mean_mse"] - baseline_mse for r in per_channel if r.get("channel", -1) >= 0]
        names = [r["channel_name"] for r in per_channel if r.get("channel", -1) >= 0]
        if channel_names is None:
            channel_names = names
        x = np.arange(len(deltas)) + i * width
        ax.bar(x, deltas, width=width, label=f"{label} (baseline test MSE = {baseline_mse:.3f})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(channel_names or [])) + (len(encoders) - 1) * width / 2)
    ax.set_xticklabels(channel_names or [], rotation=35, ha="right", fontsize="small")
    ax.set_ylabel(r"$\Delta$ test normalized MSE (zero - baseline)")
    ax.set_title("Channel-importance ablation: MSE delta when each input channel is zeroed")
    ax.legend(fontsize="small")
    fig.tight_layout()
    ch_pdf = out_dir / "channel_ablation.pdf"
    fig.savefig(ch_pdf)
    plt.close(fig)

    # 2. Frame budget curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, payload in encoders.items():
        fr = payload.get("frame")
        if fr is None:
            continue
        rows = fr.get("per_frames") or []
        rows.sort(key=lambda r: r["frames"])
        xs = [r["frames"] for r in rows]
        ys = [r["test_normalized_mean_mse"] for r in rows]
        ax.plot(xs, ys, "o-", label=label)
    ax.set_xlabel("frames at evaluation time")
    ax.set_ylabel("test normalized MSE")
    ax.set_title("Frame-budget ablation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fr_pdf = out_dir / "frame_ablation.pdf"
    fig.savefig(fr_pdf)
    plt.close(fig)

    print({"channel_pdf": str(ch_pdf), "frame_pdf": str(fr_pdf), "n_encoders": len(encoders)})


if __name__ == "__main__":
    main()
