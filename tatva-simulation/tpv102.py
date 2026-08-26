#!/usr/bin/env python3
"""SCEC TPV102 rate-and-state half-space benchmark using Tatva operators."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tpv101 import (
    TPV101Config,
    estimate_problem_size,
    preset_config as tpv101_preset_config,
    run_simulation,
    write_scec_dump,
)


@dataclass(frozen=True)
class TPV102Config(TPV101Config):
    problem: str = "TPV102"
    free_surface_y_min: bool = True


def preset_config(name: str) -> TPV102Config:
    """Build TPV102 on y>=0 while retaining the validated TPV101 physics."""
    base = tpv101_preset_config(name)
    values = asdict(base)
    values["y_min"] = 0.0
    # Kaneko's public TPV102 histories obey this fault-plane parity exactly:
    # tangential fields are odd in z and the normal field is even.
    values["symmetry_reduced"] = True

    if name == "smoke":
        values["z_extent"] = 12_000.0
    elif name in {"coarse", "hpc-500m", "hpc-200m", "hpc-100m"}:
        values["z_extent"] = 12_000.0
    elif name == "hpc-150m":
        values["y_max"] = 181.0 * base.mesh_size
        values["z_extent"] = 80.0 * base.mesh_size
    elif name == "hpc-160m":
        values["y_max"] = 170.0 * base.mesh_size
        values["z_extent"] = 76.0 * base.mesh_size
    else:  # pragma: no cover - guarded by tpv101_preset_config
        raise ValueError(f"Unknown preset: {name}")
    return TPV102Config(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=(
            "smoke",
            "coarse",
            "hpc-500m",
            "hpc-200m",
            "hpc-150m",
            "hpc-160m",
            "hpc-100m",
        ),
        default="coarse",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mesh-size", type=float, default=None)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--z-extent", type=float, default=None)
    parser.add_argument("--graded-mesh", action="store_true")
    parser.add_argument("--fine-x-min", type=float, default=None)
    parser.add_argument("--fine-x-max", type=float, default=None)
    parser.add_argument("--fine-y-min", type=float, default=None)
    parser.add_argument("--fine-y-max", type=float, default=None)
    parser.add_argument("--fine-z-extent", type=float, default=None)
    parser.add_argument("--max-mesh-size", type=float, default=None)
    parser.add_argument("--mesh-growth-ratio", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output-dt", type=float, default=None)
    parser.add_argument("--operator-batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--uguca-dump-interval", type=float, default=None)
    parser.add_argument("--uguca-dump-dir", type=Path, default=None)
    parser.add_argument("--uguca-dump-name", type=str, default=None)
    parser.add_argument("--uguca-dump-x-min", type=float, default=None)
    parser.add_argument("--uguca-dump-x-max", type=float, default=None)
    parser.add_argument("--uguca-dump-y-min", type=float, default=None)
    parser.add_argument("--uguca-dump-y-max", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = preset_config(args.preset)
    overrides = {
        "mesh_size": args.mesh_size,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_min": args.y_min,
        "y_max": args.y_max,
        "z_extent": args.z_extent,
        "fine_x_min": args.fine_x_min,
        "fine_x_max": args.fine_x_max,
        "fine_y_min": args.fine_y_min,
        "fine_y_max": args.fine_y_max,
        "fine_z_extent": args.fine_z_extent,
        "max_mesh_size": args.max_mesh_size,
        "mesh_growth_ratio": args.mesh_growth_ratio,
        "duration": args.duration,
        "output_dt": args.output_dt,
        "operator_batch_size": args.operator_batch_size,
        "uguca_dump_interval": args.uguca_dump_interval,
        "uguca_dump_x_min": args.uguca_dump_x_min,
        "uguca_dump_x_max": args.uguca_dump_x_max,
        "uguca_dump_y_min": args.uguca_dump_y_min,
        "uguca_dump_y_max": args.uguca_dump_y_max,
    }
    config = replace(
        config,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    if args.graded_mesh:
        config = replace(config, graded_mesh=True)
    size = estimate_problem_size(config)
    print(json.dumps({"config": asdict(config), "problem_size": size}, indent=2))
    if args.dry_run:
        return 0

    output_dir = args.output_dir or Path("output") / f"tpv102_{args.preset}"
    uguca_dump_dir = args.uguca_dump_dir
    if config.uguca_dump_interval and uguca_dump_dir is None:
        uguca_dump_dir = output_dir
    result = run_simulation(
        config,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval_s=args.checkpoint_interval,
        resume=args.resume,
        uguca_dump_dir=uguca_dump_dir,
        uguca_dump_name=args.uguca_dump_name or output_dir.name,
    )
    paths = write_scec_dump(result, output_dir)
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
