#!/usr/bin/env python3
"""Compare a Tatva TPV101/TPV102 dump with a public SCEC submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tpv101 import (
    STATIONS,
    SURFACE_STATIONS,
    SURFACE_TIME_SERIES_FIELDS,
    TIME_SERIES_FIELDS,
)
from validate_scec_dump import _read_ascii_table


COMPARISON_FIELDS = {
    "horizontal_slip": 1,
    "horizontal_slip_rate": 2,
    "horizontal_shear_stress": 3,
    "log_theta": 8,
}
SURFACE_COMPARISON_FIELDS = {
    "horizontal_displacement": 1,
    "horizontal_velocity": 2,
    "vertical_displacement": 3,
    "vertical_velocity": 4,
    "normal_displacement": 5,
    "normal_velocity": 6,
}


def _threshold_crossing(time: np.ndarray, rate: np.ndarray) -> float | None:
    speed = np.abs(rate)
    crossings = np.flatnonzero((speed[:-1] < 1.0e-3) & (speed[1:] >= 1.0e-3))
    if crossings.size == 0:
        return None
    index = int(crossings[0])
    fraction = (1.0e-3 - speed[index]) / (speed[index + 1] - speed[index])
    return float(time[index] + fraction * (time[index + 1] - time[index]))


def _station_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate_time = candidate[:, 0]
    reference_time = reference[:, 0]
    maximum_time = min(candidate_time[-1], reference_time[-1])
    sample_mask = candidate_time <= maximum_time + 1.0e-12
    sample_time = candidate_time[sample_mask]
    metrics: dict[str, float | None] = {}

    for name, column in COMPARISON_FIELDS.items():
        candidate_values = candidate[sample_mask, column]
        reference_values = np.interp(sample_time, reference_time, reference[:, column])
        difference = candidate_values - reference_values
        scale = np.ptp(reference_values)
        metrics[f"{name}_rms"] = float(np.sqrt(np.mean(difference**2)))
        metrics[f"{name}_normalized_rms_percent"] = (
            float(100.0 * np.sqrt(np.mean(difference**2)) / scale)
            if scale > 0.0
            else None
        )

    candidate_arrival = _threshold_crossing(candidate_time, candidate[:, 2])
    reference_arrival = _threshold_crossing(reference_time, reference[:, 2])
    metrics["candidate_arrival_s"] = candidate_arrival
    metrics["reference_arrival_s"] = reference_arrival
    metrics["arrival_difference_s"] = (
        candidate_arrival - reference_arrival
        if candidate_arrival is not None and reference_arrival is not None
        else None
    )
    return metrics


def _surface_station_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate_time = candidate[:, 0]
    reference_time = reference[:, 0]
    maximum_time = min(candidate_time[-1], reference_time[-1])
    sample_mask = candidate_time <= maximum_time + 1.0e-12
    sample_time = candidate_time[sample_mask]
    metrics: dict[str, float | None] = {}
    for name, column in SURFACE_COMPARISON_FIELDS.items():
        candidate_values = candidate[sample_mask, column]
        reference_values = np.interp(sample_time, reference_time, reference[:, column])
        difference = candidate_values - reference_values
        scale = np.ptp(reference_values)
        rms = float(np.sqrt(np.mean(difference**2)))
        metrics[f"{name}_rms"] = rms
        metrics[f"{name}_normalized_rms_percent"] = (
            float(100.0 * rms / scale) if scale > 0.0 else None
        )
    return metrics


def _contour_metrics(
    candidate: np.ndarray, reference: np.ndarray, comparison_time: float
) -> dict:
    reference_x = np.unique(reference[:, 0])
    reference_y = np.unique(reference[:, 1])
    reference_grid = np.full(
        (reference_x.size, reference_y.size), np.nan, dtype=np.float64
    )
    x_indices = np.searchsorted(reference_x, reference[:, 0])
    y_indices = np.searchsorted(reference_y, reference[:, 1])
    reference_grid[x_indices, y_indices] = reference[:, 2]
    if np.isnan(reference_grid).any():
        raise ValueError("Reference rupture contour is not a complete regular grid")

    def interpolate(x: float, y: float) -> float | None:
        if not (
            reference_x[0] <= x <= reference_x[-1]
            and reference_y[0] <= y <= reference_y[-1]
        ):
            return None
        ix = int(np.clip(np.searchsorted(reference_x, x) - 1, 0, reference_x.size - 2))
        iy = int(np.clip(np.searchsorted(reference_y, y) - 1, 0, reference_y.size - 2))
        tx = (x - reference_x[ix]) / (reference_x[ix + 1] - reference_x[ix])
        ty = (y - reference_y[iy]) / (reference_y[iy + 1] - reference_y[iy])
        values = reference_grid[ix : ix + 2, iy : iy + 2]
        return float(
            (1.0 - tx) * (1.0 - ty) * values[0, 0]
            + tx * (1.0 - ty) * values[1, 0]
            + (1.0 - tx) * ty * values[0, 1]
            + tx * ty * values[1, 1]
        )

    pairs = []
    missing = 0
    candidate_ruptured = 0
    reference_ruptured = 0
    candidate_only = 0
    reference_only = 0
    for x, y, arrival in candidate:
        reference_arrival = interpolate(float(x), float(y))
        if reference_arrival is None:
            missing += 1
            continue
        candidate_has_ruptured = arrival <= comparison_time
        reference_has_ruptured = reference_arrival <= comparison_time
        candidate_ruptured += int(candidate_has_ruptured)
        reference_ruptured += int(reference_has_ruptured)
        candidate_only += int(candidate_has_ruptured and not reference_has_ruptured)
        reference_only += int(reference_has_ruptured and not candidate_has_ruptured)
        if candidate_has_ruptured and reference_has_ruptured:
            pairs.append((arrival, reference_arrival))

    values = np.asarray(pairs, dtype=np.float64)
    if values.size:
        differences = values[:, 0] - values[:, 1]
        rms = float(np.sqrt(np.mean(differences**2)))
        bias = float(np.mean(differences))
        maximum = float(np.max(np.abs(differences)))
    else:
        rms = bias = maximum = None
    return {
        "candidate_nodes": int(candidate.shape[0]),
        "matched_coordinates": int(candidate.shape[0] - missing),
        "comparison_time_s": comparison_time,
        "candidate_ruptured_nodes": candidate_ruptured,
        "reference_ruptured_nodes": reference_ruptured,
        "both_ruptured_nodes": int(len(pairs)),
        "candidate_only_ruptured_nodes": candidate_only,
        "reference_only_ruptured_nodes": reference_only,
        "rupture_time_rms_s": rms,
        "rupture_time_bias_s": bias,
        "rupture_time_max_abs_s": maximum,
    }


def compare(
    candidate_dir: Path, reference_dir: Path, *, problem: str | None = None
) -> dict:
    if problem is None:
        summary_path = candidate_dir / "summary.json"
        if summary_path.exists():
            problem = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "problem", "TPV101"
            )
        else:
            problem = "TPV102" if (candidate_dir / "tpv102_rupture_time.txt").exists() else "TPV101"
    problem = problem.upper()
    station_metrics = {}
    comparison_time: float | None = None
    for station_name, _x, _y in STATIONS:
        candidate_fields, candidate = _read_ascii_table(
            candidate_dir / f"{station_name}.txt"
        )
        reference_fields, reference = _read_ascii_table(
            reference_dir / f"{station_name}.txt"
        )
        if candidate_fields != list(TIME_SERIES_FIELDS):
            raise ValueError(f"Unexpected candidate fields at {station_name}")
        if reference_fields != list(TIME_SERIES_FIELDS):
            raise ValueError(f"Unexpected reference fields at {station_name}")
        station_metrics[station_name] = _station_metrics(candidate, reference)
        station_comparison_time = float(min(candidate[-1, 0], reference[-1, 0]))
        if comparison_time is None:
            comparison_time = station_comparison_time
        elif not np.isclose(comparison_time, station_comparison_time):
            raise ValueError("Station files do not share one comparison duration")

    surface_metrics = {}
    surface_missing_from_reference = []
    if problem == "TPV102":
        for station_name, _z, _x in SURFACE_STATIONS:
            candidate_path = candidate_dir / f"{station_name}.txt"
            reference_path = reference_dir / f"{station_name}.txt"
            if not reference_path.exists():
                surface_missing_from_reference.append(station_name)
                continue
            if not candidate_path.exists():
                raise FileNotFoundError(f"Missing candidate surface station: {candidate_path}")
            candidate_fields, candidate = _read_ascii_table(candidate_path)
            reference_fields, reference = _read_ascii_table(reference_path)
            if candidate_fields != list(SURFACE_TIME_SERIES_FIELDS):
                raise ValueError(f"Unexpected candidate fields at {station_name}")
            if reference_fields != list(SURFACE_TIME_SERIES_FIELDS):
                raise ValueError(f"Unexpected reference fields at {station_name}")
            surface_metrics[station_name] = _surface_station_metrics(candidate, reference)

    contour_name = f"{problem.lower()}_rupture_time.txt"
    candidate_fields, candidate_contour = _read_ascii_table(
        candidate_dir / contour_name
    )
    reference_fields, reference_contour = _read_ascii_table(
        reference_dir / contour_name
    )
    if candidate_fields != ["j", "k", "t"] or reference_fields != ["j", "k", "t"]:
        raise ValueError("Unexpected rupture contour fields")

    return {
        "problem": problem,
        "candidate": str(candidate_dir.resolve()),
        "reference": str(reference_dir.resolve()),
        "stations": station_metrics,
        "surface_stations": surface_metrics,
        "surface_stations_missing_from_reference": surface_missing_from_reference,
        "contour": _contour_metrics(
            candidate_contour,
            reference_contour,
            comparison_time if comparison_time is not None else 0.0,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--problem", choices=("TPV101", "TPV102"), default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare(args.candidate_dir, args.reference_dir, problem=args.problem)
    output = args.output or args.candidate_dir / "reference_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
