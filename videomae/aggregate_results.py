"""Aggregate per-encoder eval results into a single JSON for the report.

Reads from videomae/artifacts/<run>/{linear_probe,knn,analysis}/metrics.json
or analysis.json, and produces:

* ``aggregated_results.json`` -- machine-readable per-run table
* ``aggregated_results.md``   -- human-readable Markdown report-ready table

Optionally accepts an external ``--colleague-json`` path with a payload of the
form ``{"jepa_best": {...}, "vicreg_best": {...}, ...}`` containing the
colleague's reported numbers, which are merged into the table for direct
comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    label: str
    run_dir: Path
    has_linear_probe: bool = False
    has_knn: bool = False
    has_analysis: bool = False
    history: dict | None = None
    config: dict | None = None
    linear_probe_best: dict | None = None
    knn_best: dict | None = None
    analysis: dict | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate eval results across runs.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("videomae/artifacts"))
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Run directory names under artifacts-dir. Default: auto-detect (any dir with encoder_best.pt).",
    )
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <artifacts-dir>/aggregated.")
    parser.add_argument("--colleague-json", type=Path, default=None,
                        help="Optional external JSON with colleague's SIGReg/VICReg numbers.")
    return parser.parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _summarize_run(run_dir: Path) -> RunSummary:
    rs = RunSummary(label=run_dir.name, run_dir=run_dir)
    rs.history = _load_json(run_dir / "history.json")
    rs.config = _load_json(run_dir / "train_config.json")
    lp = _load_json(run_dir / "linear_probe" / "metrics.json")
    if lp and "best" in lp:
        rs.has_linear_probe = True
        rs.linear_probe_best = lp["best"]
    kn = _load_json(run_dir / "knn" / "metrics.json")
    if kn and "best" in kn:
        rs.has_knn = True
        rs.knn_best = kn["best"]
    an = _load_json(run_dir / "analysis" / "analysis.json")
    if an and "splits" in an:
        rs.has_analysis = True
        rs.analysis = an["splits"]
    return rs


def _to_row(rs: RunSummary) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": rs.label,
        "run_dir": str(rs.run_dir),
        "has_linear_probe": rs.has_linear_probe,
        "has_knn": rs.has_knn,
        "has_analysis": rs.has_analysis,
    }
    cfg = rs.config or {}
    row["total_params"] = cfg.get("total_params")
    row["encoder_params"] = cfg.get("encoder_params")
    row["epochs_trained"] = (rs.history or {}).get("epochs", [])
    row["epochs_count"] = len(row["epochs_trained"])
    if rs.history is not None:
        ep_records = rs.history.get("epochs") or []
        if ep_records:
            row["last_train_loss"] = ep_records[-1].get("train_loss")
            row["last_valid_loss"] = ep_records[-1].get("valid_loss")
            row["best_valid"] = rs.history.get("best_valid_loss") or rs.history.get("best_valid_mean_mse_normalized")
    if rs.linear_probe_best is not None:
        b = rs.linear_probe_best
        row["linear_probe"] = {
            "feature_norm": b.get("feature_norm"),
            "lr": b.get("lr"),
            "weight_decay": b.get("weight_decay"),
            "batch_size": b.get("batch_size"),
            "valid_normalized_mean_mse": b.get("best_valid_mean_mse_normalized"),
            "test_normalized_mean_mse": (b.get("test_normalized") or {}).get("mean_mse"),
            "test_normalized_alpha_mse": (b.get("test_normalized") or {}).get("alpha_mse"),
            "test_normalized_zeta_mse": (b.get("test_normalized") or {}).get("zeta_mse"),
            "test_raw_mean_mse": (b.get("test") or {}).get("mean_mse"),
        }
    if rs.knn_best is not None:
        b = rs.knn_best
        row["knn"] = {
            "n_neighbors": b.get("n_neighbors"),
            "weights": b.get("weights"),
            "metric": b.get("metric"),
            "feature_norm": b.get("feature_norm"),
            "valid_normalized_mean_mse": (b.get("valid_normalized") or {}).get("mean_mse"),
            "test_normalized_mean_mse": (b.get("test_normalized") or {}).get("mean_mse"),
            "test_normalized_alpha_mse": (b.get("test_normalized") or {}).get("alpha_mse"),
            "test_normalized_zeta_mse": (b.get("test_normalized") or {}).get("zeta_mse"),
            "test_raw_mean_mse": (b.get("test") or {}).get("mean_mse"),
        }
    if rs.analysis is not None:
        out: dict[str, Any] = {}
        for split, stats in rs.analysis.items():
            if not isinstance(stats, dict):
                continue
            out[split] = {
                "n_samples": stats.get("n_samples"),
                "embedding_dim": stats.get("embedding_dim"),
                "effective_rank": stats.get("effective_rank"),
                "participation_ratio": stats.get("participation_ratio"),
                "condition_number": stats.get("condition_number"),
                "mean_abs_offdiag_corr": stats.get("mean_abs_offdiag_corr"),
                "epps_pulley_zscore": stats.get("epps_pulley_zscore"),
                "feature_std_mean": stats.get("feature_std_mean"),
            }
        row["analysis"] = out
    return row


def _format_md_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table summarizing the headline numbers."""
    lines = []
    lines.append("# Aggregated Results\n")
    lines.append(
        "| Encoder | Params | Linear MSE | Linear alpha | Linear zeta | "
        "kNN MSE | kNN alpha | kNN zeta | EffRank (valid) | Epps-Pulley (valid) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        params = r.get("total_params")
        params_s = f"{params/1e6:.2f}M" if isinstance(params, (int, float)) else "n/a"
        lp = r.get("linear_probe") or {}
        kn = r.get("knn") or {}
        analysis = r.get("analysis") or {}
        valid = analysis.get("valid") or {}
        def f(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "---"
        lines.append(
            f"| {r['label']} | {params_s} | "
            f"{f(lp.get('test_normalized_mean_mse'))} | "
            f"{f(lp.get('test_normalized_alpha_mse'))} | "
            f"{f(lp.get('test_normalized_zeta_mse'))} | "
            f"{f(kn.get('test_normalized_mean_mse'))} | "
            f"{f(kn.get('test_normalized_alpha_mse'))} | "
            f"{f(kn.get('test_normalized_zeta_mse'))} | "
            f"{f(valid.get('effective_rank'))} | "
            f"{f(valid.get('epps_pulley_zscore'))} |"
        )
    lines.append("\nAll MSE values are on the test split, computed against z-scored (alpha, zeta) targets.")
    lines.append("Effective rank and Epps-Pulley distance are computed on validation embeddings.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    out_dir = (args.out_dir or (artifacts_dir / "aggregated")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = args.runs
    if runs is None:
        runs = sorted(
            d.name for d in artifacts_dir.iterdir()
            if d.is_dir() and (d / "encoder_best.pt").exists()
        )
        print(f"[aggregate] auto-detected runs: {runs}")
    summaries = [_summarize_run(artifacts_dir / r) for r in runs]
    rows = [_to_row(s) for s in summaries]

    if args.colleague_json is not None and args.colleague_json.exists():
        with open(args.colleague_json, "r", encoding="utf-8") as f:
            colleague = json.load(f)
        rows.append({"label": "colleague_extra", **colleague})

    payload = {"runs": rows}
    json_path = out_dir / "aggregated_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    md_path = out_dir / "aggregated_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_format_md_table(rows))
    print({"json": str(json_path), "md": str(md_path), "rows": len(rows)})
    print()
    print(_format_md_table(rows))


if __name__ == "__main__":
    main()
