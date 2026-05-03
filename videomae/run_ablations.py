"""Channel-importance and frame-budget ablations for any frozen encoder.

For every encoder in ``--encoder-checkpoints``, this script:

1. **Channel ablation** -- for each of the 11 input channels, exports embeddings
   on train/valid/test with that channel zeroed at the input, then fits a single
   linear probe and reports the test-MSE delta vs the no-ablation baseline.

2. **Frame-budget ablation** -- exports embeddings with ``num_frames in {4, 8,
   12, 16}`` (the last 16 = no ablation), fits a linear probe each, plots the
   saturation curve.

All output is written to the ``--out-dir`` (default: a NEW
``videomae/artifacts/ablations/<encoder_label>/`` directory). Existing
``videomae/artifacts/<run>/`` artifacts are NEVER overwritten.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ActiveMatterWindowDataset
from .models import ConvEncoder
from .utils import (
    LabelNormalizer,
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
    Define the CLI for ``python -m videomae.run_ablations``: data root, list of frozen-encoder checkpoints, optional matching labels, output directory (default ``videomae/artifacts/ablations``), data-loader knobs, AMP/seed/device knobs, linear-probe hyperparameters, ablation toggles (``--skip-channel``, ``--skip-frame``), and frame budgets to sweep.

    Assumptions
    -----------
    Designed to be called once at the start of ``main``. Each ``--encoder-checkpoints`` entry must be a frozen ``encoder_best.pt`` produced by ``train_videomae.py`` / ``train_supervised.py`` / colleague's JEPA trainers, containing a ``state_dict`` (or ``encoder``) key plus ``config``.

    Notes
    -----
    Linear-probe hyperparameters are intentionally fixed (lr 1e-3, wd 1e-4, 200 epochs) rather than swept, so the ablations are directly comparable across encoders.
    """
    parser = argparse.ArgumentParser(description="Run channel + frame ablations on frozen encoders.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoints", type=Path, nargs="+", required=True,
                        help="One or more encoder_best.pt paths.")
    parser.add_argument("--encoder-labels", type=str, nargs="+", default=None,
                        help="Optional labels matching the order of --encoder-checkpoints. "
                             "Defaults to the parent directory name.")
    parser.add_argument("--out-dir", type=Path, default=Path("videomae/artifacts/ablations"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    # Linear probe hyperparameters (kept simple, single config to keep ablations comparable)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-wd", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--probe-feature-norm", choices=["none", "zscore", "l2", "zscore_l2"], default="zscore")
    # Ablation toggles
    parser.add_argument("--skip-channel", action="store_true")
    parser.add_argument("--skip-frame", action="store_true")
    parser.add_argument("--frame-budgets", type=int, nargs="+", default=[4, 8, 12, 16])
    return parser.parse_args()


@dataclass
class EncoderBundle:
    """
    Parameters
    ----------
    **Input parameter 1:** label - Human-readable name for this encoder; appears in JSON output and stdout logs.
    **Input parameter 2:** checkpoint - Filesystem path to the ``encoder_best.pt`` (or equivalent) checkpoint that was loaded.
    **Input parameter 3:** encoder - The instantiated ``ConvEncoder`` with its weights already loaded onto ``device`` and ``.eval()`` set.
    **Input parameter 4:** config - The training-config dict that was packaged with the checkpoint (used to re-derive ``in_chans``, ``num_frames``, etc.).
    **Input parameter 5:** device - The device the encoder lives on (typically a single CUDA device for B200 evaluation).

    Output
    ------
    Output returned: A dataclass instance bundling everything the channel and frame ablations need to evaluate one encoder.

    Purpose
    -------
    Plain dataclass returned by ``_load_encoder``. Decoupling "load encoder" from "run ablation" keeps the helper functions free of unbundled positional args.

    Assumptions
    -----------
    Designed for the checkpoint format saved by ``train_videomae.py`` and ``train_supervised.py``: a payload dict with either ``encoder`` or ``state_dict`` keying the actual weights, plus a ``config`` dict with at least ``dims``, ``num_res_blocks``, and ``in_chans`` (or ``context_frames`` / ``num_frames``).

    Notes
    -----
    The ``encoder`` is already on ``device`` and in eval mode, so callers don't need to remember to move or eval-flag it.
    """

    label: str
    checkpoint: Path
    encoder: ConvEncoder
    config: dict
    device: torch.device


def _load_encoder(checkpoint_path: Path, device: torch.device) -> EncoderBundle:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    dims = parse_int_list(cfg["dims"]) if isinstance(cfg["dims"], str) else list(cfg["dims"])
    blocks = parse_int_list(cfg["num_res_blocks"]) if isinstance(cfg["num_res_blocks"], str) else list(cfg["num_res_blocks"])
    in_chans = int(cfg.get("in_chans", 11))
    num_frames = int(cfg.get("context_frames", cfg.get("num_frames", 16)))
    encoder = ConvEncoder(
        in_chans=in_chans, dims=dims, num_res_blocks=blocks, num_frames=num_frames,
        stem_patch_size=int(cfg.get("stem_patch_size", 1)),
        stem_kernel_size=int(cfg.get("stem_kernel_size", 4)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.0)),
        layer_scale_init_value=float(cfg.get("layer_scale_init_value", 1e-6)),
    )
    state_dict = payload["encoder"] if "encoder" in payload else payload["state_dict"]
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    return EncoderBundle(label=checkpoint_path.parent.name, checkpoint=checkpoint_path,
                         encoder=encoder, config=cfg, device=device)


def _build_dataset(data_root: Path, split: str, num_frames: int, max_samples: int | None = None) -> ActiveMatterWindowDataset:
    return ActiveMatterWindowDataset(
        root=data_root, split=split, context_frames=num_frames, target_frames=0, stride=1,
        resolution=224, max_samples=max_samples, index_mode="single_clip", clip_selection="center",
    )


def _build_loader(dataset: ActiveMatterWindowDataset, *, batch_size: int, num_workers: int,
                  prefetch_factor: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_torch_generator(seed),
    )


def _export_embeddings(bundle: EncoderBundle, *, data_root: Path, num_frames: int, splits: list[str],
                       channel_zero: int | None, batch_size: int, num_workers: int, prefetch_factor: int,
                       amp: bool, amp_dtype: torch.dtype, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns {split: (embeddings, labels)}."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in splits:
        dataset = _build_dataset(data_root, split, num_frames)
        loader = _build_loader(dataset, batch_size=batch_size, num_workers=num_workers,
                               prefetch_factor=prefetch_factor, seed=seed)
        all_emb: list[np.ndarray] = []
        all_lab: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                ctx = batch["context"].to(bundle.device, non_blocking=True)
                if channel_zero is not None:
                    ctx = ctx.clone()
                    ctx[:, channel_zero:channel_zero + 1] = 0.0
                with torch.autocast(device_type=bundle.device.type, dtype=amp_dtype,
                                    enabled=amp and bundle.device.type == "cuda"):
                    features = bundle.encoder(ctx)
                pooled = pool_features(features, "avg")
                all_emb.append(pooled.float().cpu().numpy())
                all_lab.append(batch["label"].numpy().astype(np.float32))
        out[split] = (np.concatenate(all_emb, axis=0), np.concatenate(all_lab, axis=0))
    return out


def _fit_probe(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray,
               test_x: np.ndarray, test_y: np.ndarray, *, label_norm: LabelNormalizer,
               feature_norm: str, lr: float, wd: float, batch_size: int, epochs: int,
               device: torch.device, seed: int) -> dict[str, float]:
    """Returns dict with valid_normalized_mean_mse and test_normalized_mean_mse + per-target MSEs."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    def _norm(x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None):
        if feature_norm == "none":
            return x, mean, std
        if mean is None:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
        z = (x - mean) / std
        if feature_norm == "zscore":
            return z, mean, std
        if feature_norm == "l2":
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n = np.where(n < 1e-6, 1.0, n)
            return x / n, None, None
        if feature_norm == "zscore_l2":
            n = np.linalg.norm(z, axis=1, keepdims=True)
            n = np.where(n < 1e-6, 1.0, n)
            return z / n, mean, std
        raise ValueError(f"unknown feature_norm: {feature_norm}")

    train_x_n, mean, std = _norm(train_x, None, None)
    valid_x_n, _, _ = _norm(valid_x, mean, std)
    test_x_n, _, _ = _norm(test_x, mean, std)
    train_y_n = label_norm.transform(train_y).astype(np.float32)
    valid_y_n = label_norm.transform(valid_y).astype(np.float32)
    test_y_n = label_norm.transform(test_y).astype(np.float32)

    in_dim = int(train_x_n.shape[1])
    out_dim = int(train_y_n.shape[1])
    model = torch.nn.Linear(in_dim, out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    schd = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=0.0)
    train_x_t = torch.from_numpy(train_x_n.astype(np.float32)).to(device)
    train_y_t = torch.from_numpy(train_y_n).to(device)
    valid_x_t = torch.from_numpy(valid_x_n.astype(np.float32)).to(device)
    test_x_t = torch.from_numpy(test_x_n.astype(np.float32)).to(device)

    n = train_x_t.shape[0]
    best_valid = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(train_x_t[idx])
            loss = torch.nn.functional.mse_loss(pred, train_y_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        schd.step()
        model.eval()
        with torch.no_grad():
            vp = model(valid_x_t).cpu().numpy()
        v = float(((vp - valid_y_n) ** 2).mean())
        if v < best_valid:
            best_valid = v
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = model(test_x_t).cpu().numpy()
        vp = model(valid_x_t).cpu().numpy()
    t_norm = mse_report(tp, test_y_n)
    v_norm = mse_report(vp, valid_y_n)
    return {
        "valid_normalized_mean_mse": v_norm["mean_mse"],
        "valid_normalized_alpha_mse": v_norm["alpha_mse"],
        "valid_normalized_zeta_mse": v_norm["zeta_mse"],
        "test_normalized_mean_mse": t_norm["mean_mse"],
        "test_normalized_alpha_mse": t_norm["alpha_mse"],
        "test_normalized_zeta_mse": t_norm["zeta_mse"],
    }


def _channel_ablation(bundle: EncoderBundle, *, data_root: Path, args: argparse.Namespace,
                      label_norm: LabelNormalizer, num_frames: int, amp_dtype: torch.dtype) -> dict:
    print(f"[channel] === {bundle.label} ===", flush=True)
    results: list[dict] = []
    # baseline = no channel ablation
    print("[channel] baseline (no ablation)", flush=True)
    splits = _export_embeddings(
        bundle, data_root=data_root, num_frames=num_frames, splits=["train", "valid", "test"],
        channel_zero=None, batch_size=args.batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, amp=args.amp, amp_dtype=amp_dtype, seed=args.seed,
    )
    train_x, train_y = splits["train"]
    valid_x, valid_y = splits["valid"]
    test_x, test_y = splits["test"]
    baseline = _fit_probe(
        train_x, train_y, valid_x, valid_y, test_x, test_y,
        label_norm=label_norm, feature_norm=args.probe_feature_norm,
        lr=args.probe_lr, wd=args.probe_wd, batch_size=args.probe_batch_size,
        epochs=args.probe_epochs, device=bundle.device, seed=args.seed,
    )
    results.append({"channel": -1, "channel_name": "baseline", **baseline})
    print(f"[channel] baseline test_MSE = {baseline['test_normalized_mean_mse']:.4f}", flush=True)

    # 11 channels, one zero at a time
    in_chans = int(bundle.config.get("in_chans", 11))
    channel_names = [
        "concentration", "vel_x", "vel_y",
        "orient_xx", "orient_xy", "orient_yx", "orient_yy",
        "strain_xx", "strain_xy", "strain_yx", "strain_yy",
    ][:in_chans]
    for c, name in enumerate(channel_names):
        print(f"[channel] {c}: {name} zeroed", flush=True)
        splits = _export_embeddings(
            bundle, data_root=data_root, num_frames=num_frames, splits=["train", "valid", "test"],
            channel_zero=c, batch_size=args.batch_size, num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor, amp=args.amp, amp_dtype=amp_dtype, seed=args.seed,
        )
        tx, ty = splits["train"]
        vx, vy = splits["valid"]
        ttx, tty = splits["test"]
        m = _fit_probe(
            tx, ty, vx, vy, ttx, tty, label_norm=label_norm, feature_norm=args.probe_feature_norm,
            lr=args.probe_lr, wd=args.probe_wd, batch_size=args.probe_batch_size,
            epochs=args.probe_epochs, device=bundle.device, seed=args.seed,
        )
        m["test_mse_delta_vs_baseline"] = m["test_normalized_mean_mse"] - baseline["test_normalized_mean_mse"]
        results.append({"channel": c, "channel_name": name, **m})
        print(f"[channel] {c}={name} test_MSE = {m['test_normalized_mean_mse']:.4f}  delta = {m['test_mse_delta_vs_baseline']:+.4f}", flush=True)
    return {"baseline": baseline, "per_channel": results, "channel_names": channel_names}


def _frame_ablation(bundle: EncoderBundle, *, data_root: Path, args: argparse.Namespace,
                    label_norm: LabelNormalizer, amp_dtype: torch.dtype) -> dict:
    print(f"[frame] === {bundle.label} ===", flush=True)
    results: list[dict] = []
    encoder_native_frames = int(bundle.config.get("context_frames", bundle.config.get("num_frames", 16)))
    for nf in args.frame_budgets:
        if nf > encoder_native_frames:
            print(f"[frame] {nf} > native {encoder_native_frames}, skipping", flush=True)
            continue
        # NB: we feed the encoder fewer time steps; the temporal pooling in
        # ConvEncoder collapses time at the last stage anyway. The encoder
        # was trained at 16 frames so reducing should change spatial features
        # only via the temporal-stride downsample chain. Best-effort: pad with
        # the last frame so the encoder shape stays valid for the spatial path,
        # but we set the inputs from frame 0..nf-1 to the actual clip and
        # repeat the last frame for the rest. This is a "frame-budget" probe
        # rather than a re-architecturing.
        print(f"[frame] frames={nf}", flush=True)
        # Custom export with frame masking
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split in ["train", "valid", "test"]:
            ds = _build_dataset(data_root, split, encoder_native_frames)
            loader = _build_loader(
                ds, batch_size=args.batch_size, num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor, seed=args.seed,
            )
            all_emb: list[np.ndarray] = []
            all_lab: list[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    ctx = batch["context"].to(bundle.device, non_blocking=True)
                    if nf < encoder_native_frames:
                        ctx = ctx.clone()
                        # Replace frames nf..end with the last available frame at index nf-1.
                        last_valid = ctx[:, :, nf - 1:nf].clone()
                        ctx[:, :, nf:] = last_valid
                    with torch.autocast(device_type=bundle.device.type, dtype=amp_dtype,
                                        enabled=args.amp and bundle.device.type == "cuda"):
                        feat = bundle.encoder(ctx)
                    pooled = pool_features(feat, "avg")
                    all_emb.append(pooled.float().cpu().numpy())
                    all_lab.append(batch["label"].numpy().astype(np.float32))
            out[split] = (np.concatenate(all_emb, axis=0), np.concatenate(all_lab, axis=0))
        tx, ty = out["train"]
        vx, vy = out["valid"]
        ttx, tty = out["test"]
        m = _fit_probe(
            tx, ty, vx, vy, ttx, tty, label_norm=label_norm, feature_norm=args.probe_feature_norm,
            lr=args.probe_lr, wd=args.probe_wd, batch_size=args.probe_batch_size,
            epochs=args.probe_epochs, device=bundle.device, seed=args.seed,
        )
        results.append({"frames": nf, **m})
        print(f"[frame] frames={nf} test_MSE = {m['test_normalized_mean_mse']:.4f}", flush=True)
    return {"per_frames": results, "encoder_native_frames": encoder_native_frames}


def main() -> None:
    """
    Parameters
    ----------
    (No input parameters; CLI args come from ``parse_args``.)

    Output
    ------
    Output returned: ``None``. Side effect: writes one ``<encoder_label>/`` subdirectory per encoder under ``--out-dir``, each containing ``encoder_config.json``, ``channel_ablation.json`` (unless ``--skip-channel``), and ``frame_ablation.json`` (unless ``--skip-frame``); also writes a top-level ``label_norm.json`` for reproducibility.

    Purpose
    -------
    For every encoder in ``--encoder-checkpoints``: (1) load the frozen encoder, (2) run the channel-importance ablation (zero each of the 11 input channels in turn, fit a fresh linear probe, report test-MSE delta), (3) run the frame-budget ablation (replay the encoder on ``--frame-budgets`` last-frame-padded clips, fit a fresh probe each, plot the saturation curve), saving a JSON per ablation per encoder.

    Assumptions
    -----------
    Designed for frozen-encoder checkpoints whose ``ConvEncoder`` config (``dims``, ``num_res_blocks``, ``stem_*``) is recoverable from the saved ``config`` dict. Existing ablation outputs in ``<out-dir>/<encoder_label>/`` are overwritten on re-run; pre-trained encoder artifacts in ``videomae/artifacts/<run>/`` are NEVER overwritten.

    Notes
    -----
    The label normalizer is fit on the train labels of the dataset once and reused across all encoders so the linear probes target the same z-scored (alpha, zeta). ``torch.cuda.empty_cache`` is called between encoders so peak memory stays bounded across many checkpoints.
    """
    args = parse_args()
    seed_everything(args.seed)
    configure_torch_runtime(deterministic=False)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    if args.encoder_labels is not None and len(args.encoder_labels) != len(args.encoder_checkpoints):
        raise SystemExit("--encoder-labels must match --encoder-checkpoints length")
    labels = args.encoder_labels or [p.parent.name for p in args.encoder_checkpoints]

    # Fit label normalizer on TRAIN labels (no leakage)
    train_ds = _build_dataset(args.data_root, "train", num_frames=16)
    train_labels = np.asarray(
        [train_ds.file_infos[fi].labels[si] for fi, si, _ in train_ds.index],
        dtype=np.float32,
    )
    label_norm = LabelNormalizer.fit(train_labels)
    save_json(out_dir / "label_norm.json", label_norm.to_dict())

    for ckpt, label in zip(args.encoder_checkpoints, labels):
        bundle = _load_encoder(ckpt, device)
        bundle.label = label
        bundle_dir = ensure_dir(out_dir / label)
        save_json(bundle_dir / "encoder_config.json", {
            "checkpoint": str(ckpt),
            "label": label,
            "in_chans": int(bundle.config.get("in_chans", 11)),
            "context_frames": int(bundle.config.get("context_frames", 16)),
        })
        if not args.skip_channel:
            ch = _channel_ablation(bundle, data_root=args.data_root, args=args,
                                   label_norm=label_norm, num_frames=int(bundle.config.get("context_frames", 16)),
                                   amp_dtype=amp_dtype)
            save_json(bundle_dir / "channel_ablation.json", ch)
        if not args.skip_frame:
            fr = _frame_ablation(bundle, data_root=args.data_root, args=args,
                                 label_norm=label_norm, amp_dtype=amp_dtype)
            save_json(bundle_dir / "frame_ablation.json", fr)
        del bundle
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print({"out_dir": str(out_dir), "encoders": labels})


if __name__ == "__main__":
    main()
