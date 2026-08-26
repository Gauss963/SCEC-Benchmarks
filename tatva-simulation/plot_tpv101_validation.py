#!/usr/bin/env python3
"""Draw reproducible TPV101/TPV102 validation figures from SCEC submissions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from compare_scec_reference import _contour_metrics, _station_metrics
from tpv101 import (
    STATIONS,
    SURFACE_STATIONS,
    SURFACE_TIME_SERIES_FIELDS,
    TIME_SERIES_FIELDS,
)
from validate_scec_dump import _read_ascii_table


PUBLIC_LABELS = {
    "barall": "FaultMod (Barall)",
    "dalguer2": "DFM (Dalguer & Day)",
    "dunham": "MDSBI (Dunham)",
    "kaneko": "SPECFEM3D (Kaneko et al.)",
    "ke": "UGUCA (User: ke, Chun-Yu Ke)",
    "liu": "BI (Lapusta & Liu)",
}
PUBLIC_MESH_SIZES = {"dalguer2": 50.0}
REFERENCE_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
)
CANDIDATE_COLORS = ("#4D4D4D", "#7A5195", "#C62828", "#003F5C")
PROBLEM = "TPV101"

FIELD_SPECS = (
    ("h-slip", 1, "Horizontal slip [m]"),
    ("h-slip-rate", 2, "Horizontal slip rate [m/s]"),
    ("h-shear-stress", 3, "Horizontal shear stress [MPa]"),
    ("v-slip", 4, "Vertical slip [m]"),
    ("v-slip-rate", 5, "Vertical slip rate [m/s]"),
    ("v-shear-stress", 6, "Vertical shear stress [MPa]"),
    ("n-stress", 7, "Normal stress [MPa]"),
    ("log-theta", 8, r"$\log_{10}(\theta\,[\mathrm{s}])$"),
)
SURFACE_FIELD_SPECS = (
    ("h-disp", 1, "Horizontal displacement [m]"),
    ("h-vel", 2, "Horizontal velocity [m/s]"),
    ("v-disp", 3, "Vertical displacement [m]"),
    ("v-vel", 4, "Vertical velocity [m/s]"),
    ("n-disp", 5, "Fault-normal displacement [m]"),
    ("n-vel", 6, "Fault-normal velocity [m/s]"),
)
DIFFERENCE_METRIC_SPECS = (
    ("horizontal_slip_normalized_rms_percent", "Horizontal slip"),
    ("horizontal_slip_rate_normalized_rms_percent", "Horizontal slip rate"),
    ("horizontal_shear_stress_normalized_rms_percent", "Horizontal shear stress"),
    ("log_theta_normalized_rms_percent", r"$\log_{10}(\theta)$"),
)


@dataclass(frozen=True)
class Submission:
    label: str
    path: Path
    color: str
    kind: str
    mesh_size_m: float | None = None


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, dpi in (("pdf", None), ("png", 240)):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def _station_title(station_name: str) -> str:
    station = next(item for item in STATIONS if item[0] == station_name)
    return f"x = {station[1] / 1000:g} km, depth = {station[2] / 1000:g} km"


def _station_axis_label(station: tuple[str, float, float]) -> str:
    return f"{station[1] / 1000:g}/{station[2] / 1000:g}"


def _short_reference_label(label: str) -> str:
    if label.startswith("UGUCA"):
        return "UGUCA"
    if label.startswith("SPECFEM3D"):
        return "SPECFEM3D"
    return label


def _load_station(submission: Submission, station_name: str) -> np.ndarray:
    fields, values = _read_ascii_table(submission.path / f"{station_name}.txt")
    if fields != list(TIME_SERIES_FIELDS) or values.ndim != 2 or values.shape[1] != 9:
        raise ValueError(f"Unexpected station format: {submission.path}/{station_name}.txt")
    values = values.copy()
    # CVWS requires compression-positive normal stress. Some historical public
    # submissions use the opposite sign despite the uploaded field name.
    if np.nanmedian(values[: min(10, len(values)), 7]) < 0.0:
        values[:, 7] *= -1.0
    return values


def _load_contour(submission: Submission) -> np.ndarray:
    fields, values = _read_ascii_table(
        submission.path / f"{PROBLEM.lower()}_rupture_time.txt"
    )
    if fields != ["j", "k", "t"] or values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Unexpected contour format: {submission.path}")
    return values


def _reference_submissions(
    reference_root: Path, users: set[str] | None = None
) -> list[Submission]:
    references = []
    for index, path in enumerate(sorted(reference_root.glob("*_100m"))):
        user = path.name.removesuffix("_100m")
        if user not in PUBLIC_LABELS or (users is not None and user not in users):
            continue
        references.append(
            Submission(
                label=PUBLIC_LABELS[user],
                path=path,
                color=REFERENCE_COLORS[index % len(REFERENCE_COLORS)],
                kind="reference",
                mesh_size_m=PUBLIC_MESH_SIZES.get(user, 100.0),
            )
        )
    if not references:
        raise FileNotFoundError(f"No public submissions found below {reference_root}")
    return references


def _candidate_submission(value: str, index: int) -> Submission:
    if "=" in value:
        label, raw_path = value.split("=", maxsplit=1)
    else:
        raw_path = value
        label = Path(raw_path).name
    path = Path(raw_path)
    summary_path = path / "summary.json"
    mesh_size = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mesh_size = float(summary["config"]["mesh_size"])
        if "=" not in value:
            label = f"Tatva {mesh_size:g} m"
    return Submission(
        label=label,
        path=path,
        color=CANDIDATE_COLORS[index % len(CANDIDATE_COLORS)],
        kind="candidate",
        mesh_size_m=mesh_size,
    )


def _plot_curves(
    ax: plt.Axes,
    submissions: list[Submission],
    station_name: str,
    column: int,
    maximum_time: float,
) -> None:
    for submission in submissions:
        values = _load_station(submission, station_name)
        mask = values[:, 0] <= maximum_time + 1.0e-12
        is_reference = submission.kind == "reference"
        ax.plot(
            values[mask, 0],
            values[mask, column],
            color=submission.color,
            lw=0.9 if is_reference else 1.8,
            alpha=0.72 if is_reference else 1.0,
            zorder=1 if is_reference else 3,
        )
    ax.grid(color="#D9D9D9", lw=0.5, alpha=0.7)
    ax.set_xlim(0.0, maximum_time)


def _legend_handles(submissions: list[Submission]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=item.color,
            lw=1.0 if item.kind == "reference" else 2.0,
            label=item.label,
        )
        for item in submissions
    ]


def plot_field_overviews(
    submissions: list[Submission],
    output_dir: Path,
    *,
    prefix: str,
    maximum_time: float,
) -> list[Path]:
    paths = []
    for field_name, column, ylabel in FIELD_SPECS:
        fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.0), sharex=True)
        for ax, (station_name, _x, _y) in zip(axes.flat, STATIONS, strict=True):
            _plot_curves(ax, submissions, station_name, column, maximum_time)
            ax.set_title(_station_title(station_name))
        for ax in axes[-1, :]:
            ax.set_xlabel("Time [s]")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        fig.suptitle(f"{PROBLEM} {ylabel}", fontsize=13, fontweight="bold")
        if field_name == "n-stress":
            fig.text(
                0.99,
                0.01,
                "Normal stress normalized to SCEC compression-positive convention",
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#555555",
            )
        fig.legend(
            handles=_legend_handles(submissions),
            loc="outside lower center",
            ncol=min(4, len(submissions)),
        )
        fig.subplots_adjust(bottom=0.13, top=0.92, hspace=0.30, wspace=0.24)
        paths.extend(_save(fig, output_dir, f"{prefix}_{field_name}"))
    return paths


def _load_surface_station(submission: Submission, station_name: str) -> np.ndarray:
    fields, values = _read_ascii_table(submission.path / f"{station_name}.txt")
    if (
        fields != list(SURFACE_TIME_SERIES_FIELDS)
        or values.ndim != 2
        or values.shape[1] != len(SURFACE_TIME_SERIES_FIELDS)
    ):
        raise ValueError(
            f"Unexpected surface-station format: {submission.path}/{station_name}.txt"
        )
    return values


def plot_surface_field_overviews(
    submissions: list[Submission], output_dir: Path, maximum_time: float
) -> list[Path]:
    available = [
        submission
        for submission in submissions
        if any((submission.path / f"{station[0]}.txt").exists() for station in SURFACE_STATIONS)
    ]
    if not available:
        return []
    paths = []
    for field_name, column, ylabel in SURFACE_FIELD_SPECS:
        fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.0), sharex=True)
        used_submissions: list[Submission] = []
        for ax, (station_name, station_z, station_x) in zip(
            axes.flat, SURFACE_STATIONS, strict=True
        ):
            for submission in available:
                path = submission.path / f"{station_name}.txt"
                if not path.exists():
                    continue
                values = _load_surface_station(submission, station_name)
                mask = values[:, 0] <= maximum_time + 1.0e-12
                ax.plot(
                    values[mask, 0],
                    values[mask, column],
                    color=submission.color,
                    lw=0.9 if submission.kind == "reference" else 1.8,
                    alpha=0.75 if submission.kind == "reference" else 1.0,
                )
                if submission not in used_submissions:
                    used_submissions.append(submission)
            side = "far" if station_z > 0.0 else "near"
            ax.set_title(
                f"x={station_x / 1000:g} km, |z|={abs(station_z) / 1000:g} km ({side})"
            )
            ax.grid(color="#D9D9D9", lw=0.5, alpha=0.7)
            ax.set_xlim(0.0, maximum_time)
        for ax in axes[-1, :]:
            ax.set_xlabel("Time [s]")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        fig.suptitle(f"TPV102 free surface: {ylabel}", fontsize=13, fontweight="bold")
        fig.legend(
            handles=_legend_handles(used_submissions),
            loc="outside lower center",
            ncol=min(4, len(used_submissions)),
        )
        fig.subplots_adjust(bottom=0.15, top=0.90, hspace=0.30, wspace=0.24)
        paths.extend(_save(fig, output_dir, f"surface_comparison_{field_name}"))
    return paths


def plot_station_details(
    submissions: list[Submission], output_dir: Path, maximum_time: float
) -> list[Path]:
    paths = []
    for station_name, _x, _y in STATIONS:
        fig, axes = plt.subplots(4, 2, figsize=(10.5, 10.0), sharex=True)
        for ax, (field_name, column, ylabel) in zip(axes.flat, FIELD_SPECS, strict=True):
            _plot_curves(ax, submissions, station_name, column, maximum_time)
            ax.set_ylabel(ylabel)
            ax.set_title(field_name)
        for ax in axes[-1, :]:
            ax.set_xlabel("Time [s]")
        fig.suptitle(
            f"{PROBLEM} station comparison: {_station_title(station_name)}",
            fontsize=13,
            fontweight="bold",
        )
        fig.legend(
            handles=_legend_handles(submissions),
            loc="outside lower center",
            ncol=min(4, len(submissions)),
        )
        fig.subplots_adjust(bottom=0.11, top=0.93, hspace=0.39, wspace=0.27)
        paths.extend(_save(fig, output_dir, f"station_{station_name}_all_fields"))
    return paths


def _contour_grid(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.unique(values[:, 0])
    y = np.unique(values[:, 1])
    grid = np.full((y.size, x.size), np.nan)
    ix = np.searchsorted(x, values[:, 0])
    iy = np.searchsorted(y, values[:, 1])
    grid[iy, ix] = values[:, 2]
    grid[grid >= 1.0e8] = np.nan
    return x / 1000.0, y / 1000.0, grid


def _draw_contour(
    ax: plt.Axes,
    submission: Submission,
    levels: np.ndarray,
    *,
    color: str | None = None,
) -> None:
    x, y, grid = _contour_grid(_load_contour(submission))
    finite = np.isfinite(grid)
    if not np.any(finite):
        return
    usable_levels = levels[(levels >= np.nanmin(grid)) & (levels <= np.nanmax(grid))]
    if usable_levels.size == 0:
        return
    contours = ax.contour(
        x,
        y,
        grid,
        levels=usable_levels,
        colors=color,
        cmap=None if color else "viridis",
        linewidths=1.0,
    )
    if color is None:
        ax.clabel(contours, fmt="%.1f", fontsize=7, inline_spacing=2)


def _format_fault_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-15.0, 15.0)
    ax.set_ylim(15.0, 0.0)
    ax.set_xlabel("Along strike, x [km]")
    ax.set_ylabel("Depth [km]")
    ax.grid(color="#E0E0E0", lw=0.45, alpha=0.7)


def plot_contours(
    references: list[Submission],
    candidates: list[Submission],
    output_dir: Path,
    comparison_time: float,
) -> list[Path]:
    paths = []
    # CVWS constructs its official contour plots at 0.5 s intervals.
    full_levels = np.arange(0.5, 8.5, 0.5)
    columns = min(3, len(references))
    rows = int(np.ceil(len(references) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.35 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, submission in zip(axes.flat, references):
        _draw_contour(ax, submission, full_levels)
        _format_fault_axes(ax)
        ax.set_title(submission.label)
    for ax in axes.flat[len(references) :]:
        ax.axis("off")
    fig.suptitle(f"{PROBLEM} public rupture-time contours [s]", fontsize=13, fontweight="bold")
    paths.extend(_save(fig, output_dir, "reference_rupture_time_contours"))

    all_submissions = references + candidates
    columns = 3
    rows = int(np.ceil(len(all_submissions) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11.5, 3.35 * rows), squeeze=False)
    early_levels = np.arange(0.6, comparison_time + 0.001, 0.2)
    for ax, submission in zip(axes.flat, all_submissions):
        _draw_contour(ax, submission, early_levels)
        _format_fault_axes(ax)
        ax.set_title(submission.label)
    for ax in axes.flat[len(all_submissions) :]:
        ax.axis("off")
    fig.suptitle(
        f"{PROBLEM} early rupture-time contours, t <= {comparison_time:g} s",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths.extend(_save(fig, output_dir, "rupture_time_contours_early_comparison"))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    overlay_levels = np.arange(0.75, comparison_time + 0.001, 0.25)
    for submission in all_submissions:
        _draw_contour(ax, submission, overlay_levels, color=submission.color)
    _format_fault_axes(ax)
    ax.set_title(
        f"{PROBLEM} rupture-front overlay, t <= {comparison_time:g} s",
        fontweight="bold",
    )
    ax.legend(
        handles=_legend_handles(all_submissions),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.subplots_adjust(right=0.73)
    paths.extend(_save(fig, output_dir, "rupture_time_contour_overlay"))
    return paths


def _write_metrics(
    references: list[Submission],
    candidates: list[Submission],
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    station_rows = []
    contour_rows = []
    for candidate in candidates:
        for reference in references:
            candidate_contour = _load_contour(candidate)
            reference_contour = _load_contour(reference)
            station_durations = []
            for station_name, _x, _y in STATIONS:
                candidate_data = _load_station(candidate, station_name)
                reference_data = _load_station(reference, station_name)
                metrics = _station_metrics(candidate_data, reference_data)
                station_durations.append(min(candidate_data[-1, 0], reference_data[-1, 0]))
                for metric, value in metrics.items():
                    station_rows.append(
                        {
                            "candidate": candidate.label,
                            "mesh_size_m": candidate.mesh_size_m,
                            "reference": reference.label,
                            "station": station_name,
                            "metric": metric,
                            "value": value,
                        }
                    )
            comparison_time = float(min(station_durations))
            contour = _contour_metrics(candidate_contour, reference_contour, comparison_time)
            contour_rows.append(
                {
                    "candidate": candidate.label,
                    "mesh_size_m": candidate.mesh_size_m,
                    "reference": reference.label,
                    **contour,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    station_path = output_dir / "station_comparison_metrics.csv"
    with station_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(station_rows[0]))
        writer.writeheader()
        writer.writerows(station_rows)
    contour_path = output_dir / "rupture_contour_metrics.csv"
    with contour_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contour_rows[0]))
        writer.writeheader()
        writer.writerows(contour_rows)
    summary_rows = _summarize_station_metrics(station_rows)
    summary_csv_path = output_dir / "station_metric_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_path = output_dir / "validation_metrics.json"
    summary_path.write_text(
        json.dumps(
            {
                "stations": station_rows,
                "station_summary": summary_rows,
                "contours": contour_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return station_rows, contour_rows


def _summarize_station_metrics(station_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    metadata: dict[tuple, tuple] = {}
    for row in station_rows:
        value = row["value"]
        if value is None or not np.isfinite(value):
            continue
        key = (row["candidate"], row["reference"], row["metric"])
        groups.setdefault(key, []).append(float(value))
        metadata[key] = (row["mesh_size_m"],)

    summaries = []
    for key in sorted(groups):
        values = np.asarray(groups[key], dtype=np.float64)
        candidate, reference, metric = key
        summaries.append(
            {
                "candidate": candidate,
                "mesh_size_m": metadata[key][0],
                "reference": reference,
                "metric": metric,
                "station_count": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "standard_deviation": float(np.std(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    return summaries


def _station_metric_matrix(
    station_rows: list[dict],
    candidate: Submission,
    references: list[Submission],
    metric: str,
) -> np.ndarray:
    lookup = {
        (row["reference"], row["station"]): row["value"]
        for row in station_rows
        if row["candidate"] == candidate.label and row["metric"] == metric
    }
    return np.asarray(
        [
            [lookup.get((reference.label, station[0]), np.nan) for station in STATIONS]
            for reference in references
        ],
        dtype=np.float64,
    )


def plot_difference_statistics(
    references: list[Submission],
    candidates: list[Submission],
    station_rows: list[dict],
    contour_rows: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Plot station waveform and rupture-time errors against each reference."""
    paths = []
    station_labels = [_station_axis_label(station) for station in STATIONS]
    reference_labels = [_short_reference_label(item.label) for item in references]

    for candidate_index, candidate in enumerate(candidates):
        suffix = "" if len(candidates) == 1 else f"_candidate_{candidate_index + 1}"
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 5.9), constrained_layout=True)
        for ax, (metric, title) in zip(
            axes.flat, DIFFERENCE_METRIC_SPECS, strict=True
        ):
            values = _station_metric_matrix(
                station_rows, candidate, references, metric
            )
            finite = values[np.isfinite(values)]
            color_max = float(np.max(finite)) if finite.size else 1.0
            image = ax.imshow(
                values,
                aspect="auto",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=max(color_max, 1.0e-12),
            )
            for row_index in range(values.shape[0]):
                for column_index in range(values.shape[1]):
                    value = values[row_index, column_index]
                    if np.isfinite(value):
                        ax.text(
                            column_index,
                            row_index,
                            f"{value:.1f}",
                            ha="center",
                            va="center",
                            fontsize=7.0,
                            color="white" if value > 0.58 * color_max else "#222222",
                        )
            ax.set_xticks(range(len(STATIONS)), station_labels, rotation=45, ha="right")
            ax.set_yticks(range(len(references)), reference_labels)
            ax.set_title(title)
            fig.colorbar(image, ax=ax, label="Normalized RMS difference [%]", shrink=0.82)
        fig.suptitle(
            f"{PROBLEM} station waveform differences: {candidate.label}\n"
            "station labels are x/depth [km]",
            fontsize=12,
            fontweight="bold",
        )
        paths.extend(
            _save(
                fig,
                output_dir,
                f"station_normalized_rms_difference{suffix}",
            )
        )

        fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), constrained_layout=True)
        positions = np.arange(len(STATIONS), dtype=np.float64)
        bar_width = 0.8 / max(len(references), 1)
        arrival_metric = "arrival_difference_s"
        for reference_index, reference in enumerate(references):
            values = _station_metric_matrix(
                station_rows, candidate, [reference], arrival_metric
            )[0]
            offset = (reference_index - (len(references) - 1) / 2.0) * bar_width
            axes[0].bar(
                positions + offset,
                values * 1000.0,
                width=bar_width,
                color=reference.color,
                label=_short_reference_label(reference.label),
            )
        axes[0].axhline(0.0, color="#333333", lw=0.8)
        axes[0].set_xticks(positions, station_labels)
        axes[0].set_ylabel("Tatva - reference arrival [ms]")
        axes[0].set_title("Local rupture-arrival difference")
        axes[0].grid(axis="y", color="#D9D9D9", lw=0.5)
        axes[0].legend(ncol=min(3, len(references)))

        contour_specs = (
            ("rupture_time_rms_s", "RMS"),
            ("rupture_time_bias_s", "|Bias|"),
            ("rupture_time_max_abs_s", "Maximum"),
        )
        metric_positions = np.arange(len(contour_specs), dtype=np.float64)
        for reference_index, reference in enumerate(references):
            matching = next(
                row
                for row in contour_rows
                if row["candidate"] == candidate.label
                and row["reference"] == reference.label
            )
            values = np.asarray(
                [
                    abs(float(matching[name])) * 1000.0
                    if matching[name] is not None
                    else np.nan
                    for name, _label in contour_specs
                ]
            )
            offset = (reference_index - (len(references) - 1) / 2.0) * bar_width
            axes[1].bar(
                metric_positions + offset,
                values,
                width=bar_width,
                color=reference.color,
                label=_short_reference_label(reference.label),
            )
        axes[1].set_xticks(
            metric_positions, [label for _name, label in contour_specs]
        )
        axes[1].set_ylabel("Rupture-time difference [ms]")
        axes[1].set_title("Fault-wide rupture-time statistics")
        axes[1].grid(axis="y", color="#D9D9D9", lw=0.5)
        axes[1].legend(ncol=min(3, len(references)))
        fig.suptitle(
            f"{PROBLEM} rupture-time differences: {candidate.label}",
            fontsize=12,
            fontweight="bold",
        )
        paths.extend(
            _save(fig, output_dir, f"rupture_time_difference_statistics{suffix}")
        )
    return paths


def plot_convergence(
    references: list[Submission],
    candidates: list[Submission],
    station_rows: list[dict],
    contour_rows: list[dict],
    output_dir: Path,
) -> list[Path]:
    usable = [item for item in candidates if item.mesh_size_m is not None]
    if len(usable) < 2:
        return []
    usable.sort(key=lambda item: item.mesh_size_m, reverse=True)
    meshes = np.asarray([item.mesh_size_m for item in usable])
    center_station = "faultst000dp075"
    arrival_values = []
    contour_values = []
    contour_bias = []
    for candidate in usable:
        candidate_arrivals = []
        for reference in references:
            rows = [
                row
                for row in station_rows
                if row["candidate"] == candidate.label
                and row["reference"] == reference.label
                and row["station"] == center_station
                and row["metric"] == "arrival_difference_s"
                and row["value"] is not None
            ]
            candidate_arrivals.extend(abs(float(row["value"])) for row in rows)
        matching_contours = [
            row for row in contour_rows if row["candidate"] == candidate.label
        ]
        arrival_values.append(float(np.median(candidate_arrivals)))
        contour_values.append(
            float(
                np.median(
                    [row["rupture_time_rms_s"] for row in matching_contours if row["rupture_time_rms_s"] is not None]
                )
            )
        )
        contour_bias.append(
            float(
                np.median(
                    [row["rupture_time_bias_s"] for row in matching_contours if row["rupture_time_bias_s"] is not None]
                )
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].loglog(meshes, np.asarray(arrival_values) * 1000.0, "o-", color="#0072B2")
    axes[0].set_ylabel("Median |center arrival error| [ms]")
    axes[1].loglog(meshes, np.asarray(contour_values) * 1000.0, "o-", color="#D55E00", label="RMS")
    axes[1].plot(meshes, np.abs(contour_bias) * 1000.0, "s--", color="#009E73", label="|bias|")
    axes[1].set_ylabel("Rupture-time error [ms]")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Tatva mesh size [m]")
        ax.invert_xaxis()
        ax.grid(which="both", color="#D9D9D9", lw=0.5)
    reference_names = ", ".join(item.label for item in references)
    fig.suptitle(
        f"{PROBLEM} Tatva mesh convergence against {reference_names}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, output_dir, "tatva_mesh_convergence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("reference"))
    parser.add_argument(
        "--reference-user",
        action="append",
        choices=sorted(PUBLIC_LABELS),
        default=[],
        help="Limit plots and metrics to this CVWS user; may be repeated.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="[LABEL=]DIRECTORY",
        help="Tatva output directory; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Plot/tpv101_validation"))
    parser.add_argument("--comparison-time", type=float, default=None)
    parser.add_argument("--reference-time", type=float, default=12.0)
    parser.add_argument("--problem", choices=("TPV101", "TPV102"), default="TPV101")
    return parser.parse_args()


def main() -> int:
    global PROBLEM
    args = parse_args()
    PROBLEM = args.problem
    _set_style()
    selected_users = set(args.reference_user) if args.reference_user else None
    references = _reference_submissions(args.reference_root, selected_users)
    candidates = [_candidate_submission(value, index) for index, value in enumerate(args.candidate)]
    if not candidates:
        raise ValueError("At least one --candidate is required")
    candidate_end_times = [
        _load_station(candidate, STATIONS[0][0])[-1, 0] for candidate in candidates
    ]
    comparison_time = args.comparison_time or float(min(candidate_end_times))

    selected_prefix = "_".join(args.reference_user) if args.reference_user else "all_codes"
    reference_prefix = f"reference_{selected_prefix}"
    comparison_prefix = f"tatva_{selected_prefix}_comparison"
    generated = []
    generated.extend(
        plot_field_overviews(
            references,
            args.output_dir,
            prefix=reference_prefix,
            maximum_time=args.reference_time,
        )
    )
    generated.extend(
        plot_field_overviews(
            references + candidates,
            args.output_dir,
            prefix=comparison_prefix,
            maximum_time=comparison_time,
        )
    )
    generated.extend(
        plot_station_details(references + candidates, args.output_dir, comparison_time)
    )
    if PROBLEM == "TPV102":
        generated.extend(
            plot_surface_field_overviews(
                references + candidates, args.output_dir, comparison_time
            )
        )
    generated.extend(plot_contours(references, candidates, args.output_dir, comparison_time))
    station_rows, contour_rows = _write_metrics(
        references, candidates, args.output_dir / "stats"
    )
    generated.extend(
        plot_difference_statistics(
            references,
            candidates,
            station_rows,
            contour_rows,
            args.output_dir,
        )
    )
    generated.extend(
        plot_convergence(
            references,
            candidates,
            station_rows,
            contour_rows,
            args.output_dir,
        )
    )
    manifest = {
        "problem": PROBLEM,
        "references": [
            {
                "label": item.label,
                "path": str(item.path),
                "native_mesh_size_m": item.mesh_size_m,
            }
            for item in references
        ],
        "candidates": [
            {
                "label": item.label,
                "path": str(item.path),
                "mesh_size_m": item.mesh_size_m,
            }
            for item in candidates
        ],
        "comparison_time_s": comparison_time,
        "selected_cvws_users": args.reference_user or list(PUBLIC_LABELS),
        "files": [str(path) for path in generated],
    }
    manifest_path = args.output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(generated)} figure files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
