#!/usr/bin/env python3
"""Summarize nvidia-smi samples without treating JAX preallocation as GPU work."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.metrics.open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    utilization = [float(row[6]) for row in rows]
    memory = [float(row[5]) for row in rows]
    power = [float(row[8]) for row in rows]
    kernel_active = [value for value in utilization if value > 0.0]
    saturated = [value for value in utilization if value >= 90.0]

    report = {
        "samples": len(rows),
        "gpu_utilization_overall_mean_percent": (
            sum(utilization) / len(utilization) if utilization else 0.0
        ),
        "kernel_active_samples": len(kernel_active),
        "gpu_utilization_kernel_active_mean_percent": (
            sum(kernel_active) / len(kernel_active) if kernel_active else 0.0
        ),
        "gpu_utilization_kernel_active_p90_percent": percentile(
            kernel_active, 0.90
        ),
        "gpu_utilization_saturated_samples": len(saturated),
        "gpu_utilization_saturated_fraction": (
            len(saturated) / len(rows) if rows else 0.0
        ),
        "gpu_utilization_max_percent": max(utilization, default=0.0),
        "memory_used_max_mib": max(memory, default=0.0),
        "power_draw_max_w": max(power, default=0.0),
    }
    output = args.output or args.metrics.with_name("gpu_utilization_summary.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
