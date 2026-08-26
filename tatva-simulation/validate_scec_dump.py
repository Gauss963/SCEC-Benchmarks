#!/usr/bin/env python3
"""Validate Tatva TPV101/TPV102 ASCII files against the SCEC specification."""

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


def _read_ascii_table(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="ascii").splitlines()
    content = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not content:
        raise ValueError(f"No field list or data in {path}")
    fields = content[0].split()
    rows = []
    for line_number, line in enumerate(content[1:], start=2):
        try:
            rows.append([float(value) for value in line.split()])
        except ValueError as error:
            raise ValueError(f"Non-numeric data in {path}, content line {line_number}") from error
    return fields, np.asarray(rows, dtype=np.float64)


def _has_uniform_time_step(times: np.ndarray) -> bool:
    """Allow roundoff introduced by the SCEC 12-digit ASCII time format."""
    increments = np.diff(times)
    return bool(
        increments.size
        and np.allclose(increments, np.median(increments), rtol=0.0, atol=2.0e-11)
    )


def validate_dump(output_dir: Path, *, require_full_duration: bool = True) -> dict:
    errors: list[str] = []
    station_stats: dict[str, dict[str, float | int]] = {}
    expected_fields = list(TIME_SERIES_FIELDS)
    expected_theta_log = np.log10(1.606238999213454e9)
    expected_duration = 12.0
    expected_normal_stress_mpa = 120.0
    problem = "TPV102" if (output_dir / "tpv102_rupture_time.txt").exists() else "TPV101"
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        problem = str(summary.get("problem", problem)).upper()
        expected_duration = float(summary.get("config", {}).get("duration", 12.0))
        expected_normal_stress_mpa = (
            float(summary.get("config", {}).get("normal_stress", 120.0e6)) / 1.0e6
        )
    reference_times: np.ndarray | None = None

    for station_name, _x, _y in STATIONS:
        path = output_dir / f"{station_name}.txt"
        if not path.exists():
            errors.append(f"Missing station file: {path.name}")
            continue
        fields, data = _read_ascii_table(path)
        if fields != expected_fields:
            errors.append(f"{path.name}: wrong field list {fields}")
        if data.ndim != 2 or data.shape[1] != 9:
            errors.append(f"{path.name}: expected 9 numeric columns, got {data.shape}")
            continue
        if not np.isfinite(data).all():
            errors.append(f"{path.name}: contains non-finite values")
        times = data[:, 0]
        if times.size < 2 or not np.all(np.diff(times) > 0.0):
            errors.append(f"{path.name}: times are not strictly increasing")
        else:
            if not _has_uniform_time_step(times):
                errors.append(f"{path.name}: time step is not uniform")
        if reference_times is None:
            reference_times = times
        elif not np.array_equal(times, reference_times):
            errors.append(f"{path.name}: time vector differs from other stations")
        if not np.isclose(times[0], 0.0, atol=1e-14):
            errors.append(f"{path.name}: first time is not zero")
        if require_full_duration and not np.isclose(
            times[-1], expected_duration, atol=1e-11
        ):
            errors.append(
                f"{path.name}: final time is {times[-1]}, "
                f"expected {expected_duration:g} s"
            )

        initial = data[0]
        expected_initial = np.array(
            [0.0, 0.0, 1.0e-12, 75.0, 0.0, 0.0, 0.0, 120.0, expected_theta_log]
        )
        tolerances = np.array([1e-14, 1e-14, 1e-17, 1e-5, 1e-14, 1e-14, 1e-12, 1e-5, 1e-5])
        if not np.all(np.abs(initial - expected_initial) <= tolerances):
            errors.append(
                f"{path.name}: initial row is inconsistent with {problem}: {initial.tolist()}"
            )
        normal_stress_error = np.abs(data[:, 7] - expected_normal_stress_mpa)
        maximum_normal_stress_error = float(np.max(normal_stress_error))
        if maximum_normal_stress_error > 1.0e-4:
            errors.append(
                f"{path.name}: normal stress varies by up to "
                f"{maximum_normal_stress_error:.6g} MPa; {problem} prescribes "
                f"{expected_normal_stress_mpa:g} MPa"
            )
        station_stats[station_name] = {
            "rows": int(data.shape[0]),
            "final_time": float(times[-1]),
            "peak_horizontal_slip_rate": float(np.max(np.abs(data[:, 2]))),
            "peak_vertical_slip_rate": float(np.max(np.abs(data[:, 5]))),
            "maximum_normal_stress_error_mpa": maximum_normal_stress_error,
        }

    surface_station_stats: dict[str, dict[str, float | int]] = {}
    if problem == "TPV102":
        expected_surface_fields = list(SURFACE_TIME_SERIES_FIELDS)
        for station_name, station_z, _station_x in SURFACE_STATIONS:
            path = output_dir / f"{station_name}.txt"
            if not path.exists():
                errors.append(f"Missing free-surface station file: {path.name}")
                continue
            fields, data = _read_ascii_table(path)
            if fields != expected_surface_fields:
                errors.append(f"{path.name}: wrong field list {fields}")
            if data.ndim != 2 or data.shape[1] != len(expected_surface_fields):
                errors.append(
                    f"{path.name}: expected {len(expected_surface_fields)} numeric columns, "
                    f"got {data.shape}"
                )
                continue
            if not np.isfinite(data).all():
                errors.append(f"{path.name}: contains non-finite values")
            times = data[:, 0]
            if times.size < 2 or not np.all(np.diff(times) > 0.0):
                errors.append(f"{path.name}: times are not strictly increasing")
            elif not _has_uniform_time_step(times):
                errors.append(f"{path.name}: time step is not uniform")
            if reference_times is not None and not np.array_equal(times, reference_times):
                errors.append(f"{path.name}: time vector differs from fault stations")
            if require_full_duration and not np.isclose(
                times[-1], expected_duration, atol=1e-11
            ):
                errors.append(
                    f"{path.name}: final time is {times[-1]}, expected {expected_duration:g} s"
                )
            initial = data[0]
            expected_h_velocity = 0.5e-12 if station_z > 0.0 else -0.5e-12
            expected_initial = np.array(
                [0.0, 0.0, expected_h_velocity, 0.0, 0.0, 0.0, 0.0]
            )
            tolerances = np.array([1e-14, 1e-14, 1e-17, 1e-14, 1e-17, 1e-14, 1e-17])
            if not np.all(np.abs(initial - expected_initial) <= tolerances):
                errors.append(
                    f"{path.name}: initial free-surface row is inconsistent: "
                    f"{initial.tolist()}"
                )
            surface_station_stats[station_name] = {
                "rows": int(data.shape[0]),
                "final_time": float(times[-1]),
                "peak_horizontal_velocity": float(np.max(np.abs(data[:, 2]))),
                "peak_vertical_velocity": float(np.max(np.abs(data[:, 4]))),
                "peak_normal_velocity": float(np.max(np.abs(data[:, 6]))),
            }

    contour_path = output_dir / f"{problem.lower()}_rupture_time.txt"
    contour_stats: dict[str, float | int] = {}
    if not contour_path.exists():
        errors.append(f"Missing rupture contour: {contour_path.name}")
    else:
        fields, contour = _read_ascii_table(contour_path)
        if fields != ["j", "k", "t"]:
            errors.append(f"{contour_path.name}: wrong field list {fields}")
        if contour.ndim != 2 or contour.shape[1] != 3:
            errors.append(f"{contour_path.name}: expected 3 numeric columns, got {contour.shape}")
        else:
            x, y, arrival = contour.T
            if not (
                np.all((-15_000.0 < x) & (x < 15_000.0))
                and np.all((0.0 < y) & (y < 15_000.0))
            ):
                errors.append(f"{contour_path.name}: coordinates extend outside the VW region")
            maximum_arrival = expected_duration if require_full_duration else np.inf
            valid_arrival = (
                (arrival >= 0.0) & (arrival <= maximum_arrival)
            ) | np.isclose(arrival, 1.0e9)
            if not np.all(valid_arrival):
                errors.append(f"{contour_path.name}: invalid rupture arrival values")
            contour_stats = {
                "nodes": int(contour.shape[0]),
                "ruptured_nodes": int(np.count_nonzero(arrival < 1.0e9)),
                "earliest_arrival": float(np.min(arrival)),
                "latest_finite_arrival": float(
                    np.max(arrival[arrival < 1.0e9])
                    if np.any(arrival < 1.0e9)
                    else 1.0e9
                ),
            }

    report = {
        "problem": problem,
        "valid": not errors,
        "require_full_duration": require_full_duration,
        "expected_duration": expected_duration,
        "errors": errors,
        "stations": station_stats,
        "surface_stations": surface_station_stats,
        "contour": contour_stats,
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_dump(
        args.output_dir, require_full_duration=not args.allow_partial
    )
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
