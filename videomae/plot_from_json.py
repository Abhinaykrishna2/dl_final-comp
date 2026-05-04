"""Regenerate the 2x2 comparison figure from JSON logs, no W&B required.

Reads ``history.json`` (training-loss curves) and ``analysis.json`` (collapse /
isotropy diagnostics) for every encoder we care about and produces a single
``figures.pdf`` with:

* Panel 1: train + valid loss curves (normalized to relative epoch).
* Panel 2: per-dim embedding std on a log y-axis (overlay across encoders).
* Panel 3: bar chart of effective rank, participation ratio, mean |off-diag corr|,
  Epps-Pulley distance to N(0, I).
* Panel 4: linear-probe and kNN test MSE bar chart (read from
  ``artifacts/{linear_probe,knn}/<encoder>/metrics.json``).

Usage::

    python -m videomae.plot_from_json \
        --runs videomae/artifacts/videomae_main:VideoMAE \
               videomae/artifacts/supervised:Supervised \
        --out-dir videomae/artifacts/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """
    Parameters
    ----------
    (No input parameters; reads from ``sys.argv`` via ``argparse``.)

    Output
    ------
    Output returned: An ``argparse.Namespace`` with attributes ``runs`` (list of "path:label" specs), ``out_dir``, ``analysis_subdir``, ``linear_probe_dir``, ``knn_dir``.

    Purpose
    -------
    Define the CLI for ``python -m videomae.plot_from_json`` so ``figures.pdf`` can be regenerated entirely from local JSON logs (no W&B access required).

    Assumptions
    -----------
    Each ``--runs`` path must point at a directory produced by ``train_videomae.py`` / ``train_supervised.py`` (with ``history.json``) and optionally containing ``analysis/analysis.json``, ``linear_probe/metrics.json``, and ``knn/metrics.json`` subdirectories.

    Notes
    -----
    ``--analysis-subdir`` defaults to ``analysis``; the override exists in case the analysis was written elsewhere by an earlier run of ``analyze_representations.py``.
    """
    parser = argparse.ArgumentParser(description="Regenerate comparison figures from JSON logs.")
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="One or more 'path:label' pairs pointing at run output directories.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--analysis-subdir", type=str, default="analysis",
                        help="Subdirectory inside each run dir containing analysis.json.")
    parser.add_argument("--linear-probe-dir", type=str, default=None,
                        help="Optional override directory for linear probe metrics (else read run/linear_probe/).")
    parser.add_argument("--knn-dir", type=str, default=None,
                        help="Optional override directory for kNN metrics.")
    return parser.parse_args()


def _parse_runs(runs: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in runs:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, Path(spec).name
        out.append((label, Path(path).expanduser().resolve()))
    return out


def _load_json_safe(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _gather(runs: list[tuple[str, Path]], analysis_subdir: str) -> dict[str, dict]:
    gathered: dict[str, dict] = {}
    for label, run_dir in runs:
        gathered[label] = {
            "run_dir": str(run_dir),
            "history": _load_json_safe(run_dir / "history.json"),
            "analysis": _load_json_safe(run_dir / analysis_subdir / "analysis.json"),
            "linear_probe": _load_json_safe(run_dir / "linear_probe" / "metrics.json"),
            "knn": _load_json_safe(run_dir / "knn" / "metrics.json"),
        }
    return gathered


def _plot_loss_curves(ax, gathered: dict[str, dict]) -> None:
    found_any = False
    for label, payload in gathered.items():
        history = payload.get("history")
        if not history:
            continue
        epochs = history.get("epochs") or []
        if not epochs:
            continue
        ep = [r["epoch"] for r in epochs]
        if "valid_loss" in epochs[0]:
            ax.plot(ep, [r.get("valid_loss") for r in epochs], "-", label=f"{label} (valid)")
            found_any = True
        if "train_loss" in epochs[0]:
            ax.plot(ep, [r.get("train_loss") for r in epochs], "--", alpha=0.5, label=f"{label} (train)")
            found_any = True
    ax.set_title("Training / validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if found_any:
        ax.legend(fontsize="small")


def _plot_per_dim_std(ax, gathered: dict[str, dict]) -> None:
    found_any = False
    for label, payload in gathered.items():
        analysis = payload.get("analysis") or {}
        for split, stats in (analysis.get("splits") or {}).items():
            if split != "valid":
                continue
            stds = stats.get("per_dim_std_sorted")
            if not stds:
                continue
            ax.plot(stds, label=f"{label} ({split})")
            found_any = True
    ax.set_title("Per-dim embedding std (sorted, valid split)")
    ax.set_xlabel("dim index (sorted)")
    ax.set_ylabel("std")
    ax.set_yscale("log")
    if found_any:
        ax.legend(fontsize="small")


def _plot_isotropy_bars(ax, gathered: dict[str, dict]) -> None:
    rows = []
    for label, payload in gathered.items():
        analysis = (payload.get("analysis") or {}).get("splits") or {}
        s = analysis.get("valid") or analysis.get("train") or {}
        if not s:
            continue
        rows.append((
            label,
            s.get("effective_rank", 0.0),
            s.get("mean_abs_offdiag_corr", 0.0),
            s.get("epps_pulley_zscore", 0.0),
        ))
    if not rows:
        ax.text(0.5, 0.5, "no analysis data", ha="center", va="center")
        return
    labels = [r[0] for r in rows]
    eff_rank = [r[1] for r in rows]
    off_corr = [r[2] for r in rows]
    eppspu = [r[3] for r in rows]
    x = np.arange(len(labels))
    w = 0.27
    eff_rank_norm = np.asarray(eff_rank) / max(max(eff_rank), 1e-9)
    ax.bar(x - w, eff_rank_norm, width=w, label="effective rank (norm)")
    ax.bar(x, off_corr, width=w, label="mean |off-diag corr|")
    ax.bar(x + w, eppspu, width=w, label="Epps-Pulley (z-scored)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize="small")
    ax.set_title("Isotropy / collapse metrics (lower for last two = more isotropic)")
    ax.legend(fontsize="x-small")


def _plot_mse_bars(ax, gathered: dict[str, dict]) -> None:
    rows = []
    for label, payload in gathered.items():
        lp = payload.get("linear_probe") or {}
        kn = payload.get("knn") or {}
        lp_test = ((lp.get("best") or {}).get("test_normalized") or {}).get("mean_mse")
        kn_test = ((kn.get("best") or {}).get("test_normalized") or {}).get("mean_mse")
        if lp_test is None and kn_test is None:
            continue
        rows.append((label, lp_test, kn_test))
    if not rows:
        ax.text(0.5, 0.5, "no probe/knn data", ha="center", va="center")
        return
    labels = [r[0] for r in rows]
    lp_vals = [r[1] if r[1] is not None else 0.0 for r in rows]
    kn_vals = [r[2] if r[2] is not None else 0.0 for r in rows]
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, lp_vals, width=w, label="linear probe test MSE (normalized)")
    ax.bar(x + w / 2, kn_vals, width=w, label="kNN test MSE (normalized)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize="small")
    ax.set_title("Frozen-encoder evaluation (test, z-scored MSE)")
    ax.legend(fontsize="x-small")


def main() -> None:
    """
    Parameters
    ----------
    (No input parameters; CLI args come from ``parse_args``.)

    Output
    ------
    Output returned: ``None``. Side effect: writes ``figures.pdf`` (a 2x2 panel figure: loss curves, per-dim std, isotropy bars, frozen-eval MSE bars) and ``figures_summary.json`` (which JSON sources were available per run) to the output directory.

    Purpose
    -------
    Read every per-run JSON log (``history.json``, ``analysis/analysis.json``, ``linear_probe/metrics.json``, ``knn/metrics.json``) and assemble a one-page comparison figure. Designed to be idempotent: re-running it always produces the same PDF for the same JSON inputs.

    Assumptions
    -----------
    Requires ``matplotlib``. Missing JSON sources are tolerated: each panel falls back to a placeholder if it has no data, so partial runs still produce a usable figure.

    Notes
    -----
    The 2x2 layout fits a single printable page; the four panels were selected as the minimum set that conveys (a) optimization dynamics, (b) representation collapse, (c) isotropy, and (d) downstream regression accuracy.
    """
    args = parse_args()
    runs = _parse_runs(args.runs)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib not available: {exc}")

    gathered = _gather(runs, args.analysis_subdir)
    pdf_path = out_dir / "figures.pdf"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _plot_loss_curves(axes[0, 0], gathered)
    _plot_per_dim_std(axes[0, 1], gathered)
    _plot_isotropy_bars(axes[1, 0], gathered)
    _plot_mse_bars(axes[1, 1], gathered)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    summary_path = out_dir / "figures_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": [{"label": label, "run_dir": str(p)} for label, p in runs],
                "pdf": str(pdf_path),
                "data": {
                    label: {
                        k: bool(v) for k, v in payload.items()
                        if k in ("history", "analysis", "linear_probe", "knn")
                    }
                    for label, payload in gathered.items()
                },
            },
            f, indent=2, sort_keys=True,
        )
    print({"pdf": str(pdf_path), "summary": str(summary_path)}, flush=True)


if __name__ == "__main__":
    main()
