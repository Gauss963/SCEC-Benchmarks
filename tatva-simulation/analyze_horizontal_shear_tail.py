#!/usr/bin/env python3
"""Compare late-time SCEC horizontal shear stress across Tatva domains."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validate_scec_dump import _read_ascii_table


STATION_PATTERN = re.compile(r"faultst(?P<strike>-?\d{3})dp(?P<depth>\d{3})$")


@dataclass(frozen=True)
class Submission:
    label: str
    path: Path
    kind: str


def _parse_submission(value: str, kind: str) -> Submission:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(raw_path).name
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Submission directory not found: {path}")
    return Submission(label=label.strip(), path=path, kind=kind)


def _station_names(submissions: list[Submission]) -> list[str]:
    station_sets = [
        {path.stem for path in submission.path.glob("faultst*.txt")}
        for submission in submissions
    ]
    common = set.intersection(*station_sets)

    def station_key(name: str) -> tuple[int, int]:
        match = STATION_PATTERN.fullmatch(name)
        if match is None:
            return (10_000, 10_000)
        return (int(match.group("depth")), int(match.group("strike")))

    names = sorted((name for name in common if STATION_PATTERN.fullmatch(name)), key=station_key)
    if not names:
        raise ValueError("No common on-fault station histories were found.")
    return names


def _load_station(submission: Submission, station: str) -> np.ndarray:
    _, values = _read_ascii_table(submission.path / f"{station}.txt")
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError(f"Invalid station history: {submission.path / f'{station}.txt'}")
    return values[:, [0, 3]]


def _window_mean(time: np.ndarray, values: np.ndarray, lower: float, upper: float) -> float:
    mask = (time >= lower) & (time <= upper)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"Fewer than two samples in [{lower}, {upper}] s.")
    return float(np.mean(values[mask]))


def _tail_metrics(history: np.ndarray, start: float, end: float, window: float) -> dict[str, float]:
    time = history[:, 0]
    stress = history[:, 1]
    mask = (time >= start) & (time <= end)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"History does not cover the requested tail interval [{start}, {end}] s.")
    start_mean = _window_mean(time, stress, start, start + window)
    end_mean = _window_mean(time, stress, end - window, end)
    slope = float(np.polyfit(time[mask], stress[mask], 1)[0])
    return {
        "start_mean_mpa": start_mean,
        "end_mean_mpa": end_mean,
        "tail_change_mpa": end_mean - start_mean,
        "tail_slope_mpa_per_s": slope,
    }


def _comparison_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    start: float,
    end: float,
    window: float,
) -> dict[str, float]:
    candidate_mask = (candidate[:, 0] >= start) & (candidate[:, 0] <= end)
    time = candidate[candidate_mask, 0]
    candidate_stress = candidate[candidate_mask, 1]
    reference_stress = np.interp(time, reference[:, 0], reference[:, 1])
    candidate_tail = _tail_metrics(candidate, start, end, window)
    reference_tail = _tail_metrics(reference, start, end, window)
    difference = candidate_stress - reference_stress
    return {
        **candidate_tail,
        "reference_tail_change_mpa": reference_tail["tail_change_mpa"],
        "reference_tail_slope_mpa_per_s": reference_tail["tail_slope_mpa_per_s"],
        "tail_rmse_mpa": float(np.sqrt(np.mean(difference**2))),
        "late_bias_mpa": (
            candidate_tail["end_mean_mpa"] - reference_tail["end_mean_mpa"]
        ),
    }


def _station_title(name: str) -> str:
    match = STATION_PATTERN.fullmatch(name)
    assert match is not None
    strike = int(match.group("strike")) / 10.0
    depth = int(match.group("depth")) / 10.0
    return rf"$x={strike:g}$ km, $y={depth:g}$ km"


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _plot_histories(
    submissions: list[Submission],
    histories: dict[tuple[str, str], np.ndarray],
    stations: list[str],
    start: float,
    end: float,
    output_dir: Path,
) -> list[Path]:
    columns = 3
    rows = int(np.ceil(len(stations) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11.0, 2.75 * rows), sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    candidate_colors = ("#1479b8", "#d1492e", "#6a3d9a", "#00876c")
    reference_styles = (("#333333", "--"), ("#888888", ":"))
    candidate_index = 0
    reference_index = 0
    styles: dict[str, tuple[str, str, float]] = {}
    for submission in submissions:
        if submission.kind == "candidate":
            styles[submission.label] = (
                candidate_colors[candidate_index % len(candidate_colors)],
                "-",
                1.7,
            )
            candidate_index += 1
        else:
            color, linestyle = reference_styles[reference_index % len(reference_styles)]
            styles[submission.label] = (color, linestyle, 1.25)
            reference_index += 1

    for axis, station in zip(axes_array, stations, strict=False):
        for submission in submissions:
            history = histories[(submission.label, station)]
            color, linestyle, linewidth = styles[submission.label]
            axis.plot(
                history[:, 0],
                history[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=submission.label,
            )
        axis.axvspan(start, end, color="#dceaf3", alpha=0.35, linewidth=0)
        axis.set_xlim(max(0.0, start - 1.0), end)
        axis.set_title(_station_title(station))
        axis.grid(color="#d8d8d8", linewidth=0.55, alpha=0.7)

    for axis in axes_array[len(stations) :]:
        axis.set_visible(False)
    for axis in axes_array[-columns:]:
        if axis.get_visible():
            axis.set_xlabel("Time (s)")
    for row_index in range(rows):
        axes_array[row_index * columns].set_ylabel("Horizontal shear stress (MPa)")

    handles, labels = axes_array[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    figure.suptitle(
        f"Late-time horizontal shear stress ({start:g}-{end:g} s)",
        y=0.995,
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    paths = [
        output_dir / "horizontal_shear_stress_tail_comparison.png",
        output_dir / "horizontal_shear_stress_tail_comparison.pdf",
    ]
    for path in paths:
        figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return paths


def _plot_rmse(
    rows: list[dict[str, object]],
    candidates: list[Submission],
    stations: list[str],
    output_dir: Path,
) -> list[Path]:
    x = np.arange(len(stations), dtype=float)
    width = 0.8 / len(candidates)
    colors = ("#1479b8", "#d1492e", "#6a3d9a", "#00876c")
    figure, axis = plt.subplots(figsize=(10.5, 4.4))
    for index, candidate in enumerate(candidates):
        values = []
        for station in stations:
            matches = [
                float(row["tail_rmse_mpa"])
                for row in rows
                if row["candidate"] == candidate.label and row["station"] == station
            ]
            values.append(float(np.mean(matches)))
        offset = (index - (len(candidates) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            color=colors[index % len(colors)],
            label=candidate.label,
        )
    axis.set_xticks(x, [_station_title(station).replace("$", "") for station in stations], rotation=35, ha="right")
    axis.set_ylabel("Mean tail RMSE vs references (MPa)")
    axis.set_title("Horizontal shear stress agreement over the late-time interval")
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.6)
    axis.legend(frameon=False)
    figure.tight_layout()
    paths = [
        output_dir / "horizontal_shear_stress_tail_rmse.png",
        output_dir / "horizontal_shear_stress_tail_rmse.pdf",
    ]
    for path in paths:
        figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return paths


def analyze(args: argparse.Namespace) -> dict[str, object]:
    candidates = [_parse_submission(value, "candidate") for value in args.candidate]
    references = [_parse_submission(value, "reference") for value in args.reference]
    if not candidates or not references:
        raise ValueError("At least one --candidate and one --reference are required.")
    submissions = references + candidates
    stations = _station_names(submissions)
    histories = {
        (submission.label, station): _load_station(submission, station)
        for submission in submissions
        for station in stations
    }

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        for reference in references:
            for station in stations:
                metrics = _comparison_metrics(
                    histories[(candidate.label, station)],
                    histories[(reference.label, station)],
                    args.tail_start,
                    args.tail_end,
                    args.window,
                )
                rows.append(
                    {
                        "candidate": candidate.label,
                        "reference": reference.label,
                        "station": station,
                        **metrics,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = args.output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = stats_dir / "horizontal_shear_stress_tail_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for candidate in candidates:
        candidate_rows = [row for row in rows if row["candidate"] == candidate.label]
        aggregate.append(
            {
                "candidate": candidate.label,
                "median_tail_rmse_mpa": float(
                    np.median([float(row["tail_rmse_mpa"]) for row in candidate_rows])
                ),
                "median_absolute_late_bias_mpa": float(
                    np.median([abs(float(row["late_bias_mpa"])) for row in candidate_rows])
                ),
                "median_tail_change_mpa": float(
                    np.median([float(row["tail_change_mpa"]) for row in candidate_rows])
                ),
            }
        )
    improvement = None
    if len(candidates) == 2:
        baseline_rmse = aggregate[0]["median_tail_rmse_mpa"]
        expanded_rmse = aggregate[1]["median_tail_rmse_mpa"]
        improvement = {
            "baseline": candidates[0].label,
            "expanded": candidates[1].label,
            "median_tail_rmse_reduction_percent": float(
                100.0 * (baseline_rmse - expanded_rmse) / baseline_rmse
            ),
        }

    generated = _plot_histories(
        submissions,
        histories,
        stations,
        args.tail_start,
        args.tail_end,
        args.output_dir,
    )
    generated.extend(_plot_rmse(rows, candidates, stations, args.output_dir))
    report = {
        "tail_interval_s": [args.tail_start, args.tail_end],
        "endpoint_window_s": args.window,
        "stations": stations,
        "candidates": [str(item.path) for item in candidates],
        "references": [str(item.path) for item in references],
        "aggregate": aggregate,
        "domain_expansion_improvement": improvement,
        "metrics_csv": str(csv_path),
        "figures": [str(path) for path in generated],
    }
    json_path = stats_dir / "horizontal_shear_stress_tail_summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["summary_json"] = str(json_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=DIRECTORY",
        help="Tatva output; repeat with the original domain first and expanded domain second.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="LABEL=DIRECTORY",
        help="Reference output directory; may be repeated.",
    )
    parser.add_argument("--tail-start", type=float, default=10.0)
    parser.add_argument("--tail-end", type=float, default=15.0)
    parser.add_argument("--window", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    _set_style()
    report = analyze(parse_args())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
