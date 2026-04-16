from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a std_coeff sweep for JEPA warm-started from an existing checkpoint."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True, help="Usually artifacts/jepa/best.pt")
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/std_coeff_sweep"))
    parser.add_argument("--std-coeffs", type=float, nargs="+", default=[20.0, 30.0, 40.0])
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--standalone", action="store_true", help="Use torch distributed standalone launch mode.")
    parser.add_argument("--run-name-prefix", type=str, default="std-sweep")
    parser.add_argument(
        "--resume-if-exists",
        action="store_true",
        help="If a sweep run already has out-dir/last.pt, resume it instead of re-initializing from init-checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to physrl.train_jepa after '--'.",
    )
    return parser.parse_args()


def _slugify_coeff(value: float) -> str:
    text = f"{value:g}"
    text = text.replace("-", "neg")
    return text.replace(".", "p")


def _clean_train_args(train_args: list[str]) -> list[str]:
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    disallowed = {"--data-root", "--out-dir", "--std-coeff", "--init-checkpoint"}
    overlapping = [arg for arg in train_args if arg in disallowed]
    if overlapping:
        raise SystemExit(
            f"Do not pass {sorted(set(overlapping))} in the forwarded train args; the sweep launcher sets them."
        )
    return train_args


def _build_command(
    *,
    nproc_per_node: int,
    standalone: bool,
    data_root: Path,
    init_checkpoint: Path,
    out_dir: Path,
    std_coeff: float,
    run_name: str,
    resume_if_exists: bool,
    train_args: list[str],
) -> list[str]:
    if nproc_per_node > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
        ]
        if standalone:
            command.append("--standalone")
        command.extend(
            [
                "-m",
                "physrl.train_jepa",
            ]
        )
    else:
        command = [sys.executable, "-m", "physrl.train_jepa"]

    command.extend(
        [
            "--data-root",
            str(data_root),
            "--out-dir",
            str(out_dir),
            "--init-checkpoint",
            str(init_checkpoint),
            "--std-coeff",
            str(std_coeff),
            "--wandb-run-name",
            run_name,
        ]
    )
    if resume_if_exists and (out_dir / "last.pt").exists():
        command.append("--resume")
    command.extend(train_args)
    return command


def main() -> None:
    args = parse_args()
    train_args = _clean_train_args(args.train_args)
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    init_checkpoint = args.init_checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()

    for std_coeff in args.std_coeffs:
        coeff_slug = _slugify_coeff(std_coeff)
        out_dir = out_root / f"std_{coeff_slug}"
        run_name = f"{args.run_name_prefix}-std{coeff_slug}"
        command = _build_command(
            nproc_per_node=args.nproc_per_node,
            standalone=args.standalone,
            data_root=data_root,
            init_checkpoint=init_checkpoint,
            out_dir=out_dir,
            std_coeff=std_coeff,
            run_name=run_name,
            resume_if_exists=args.resume_if_exists,
            train_args=train_args,
        )
        print({"std_coeff": std_coeff, "out_dir": str(out_dir), "command": command}, flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
