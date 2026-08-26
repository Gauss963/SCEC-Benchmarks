#!/usr/bin/env python3
"""Animate a Tatva FEM TPV run: rupture front plus on-fault shear stress.

Unlike the UGUCA dumps handled by ``TPVShearAnimation.py``, the FEM output
stores no time-resolved shear-stress field on the fault. What it does store is

* ``<problem>_internal_diagnostics.npz``: the fault node grid (``fault_x``,
  ``fault_y``), the rate-and-state direct effect ``a`` and, per fault node, the
  rupture arrival time (1e9 where the node never ruptured);
* ``faultst<strike>dp<dip>.txt``: the nine SCEC on-fault stations sampled every
  ``output_dt``, whose column 4 is the horizontal shear stress in MPa.

The animation therefore shows the ruptured area growing with time (filled grey,
current front outlined) together with the nine stations coloured by their
instantaneous horizontal shear stress, and a time-series panel with a cursor.

Example
-------
python Analysis/code/FEMRuptureAnimation.py \
    /Volumes/Gauss-T7/SCEC-Code-Verification-Finished/SCEC-Code-Validation/tatva-simulation/output/tpv101_100m_15s_xy_expanded_gpu1_h200 \
    --output-dir Analysis/TPV101_FEM --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


DEFAULT_WIDTH = 2400
DEFAULT_HEIGHT = 1800
DEFAULT_DPI = 100
DEFAULT_FPS = 12
DEFAULT_FRAME_INTERVAL = 0.1
UNRUPTURED = 1.0e8
SHEAR_COLUMN = 3  # column #4: horizontal shear stress (MPa)
STATION_NAME = re.compile(r"faultst(-?\d+)dp(\d+)\.txt$")


def find_diagnostics(run_dir: Path) -> tuple[Path, str]:
    """Locate ``<problem>_internal_diagnostics.npz`` inside a run directory."""
    candidates = sorted(run_dir.glob("*_internal_diagnostics.npz"))
    if not candidates:
        raise FileNotFoundError(f"No *_internal_diagnostics.npz in {run_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Several diagnostics files in {run_dir}: {candidates}")
    problem = candidates[0].name[: -len("_internal_diagnostics.npz")]
    return candidates[0], problem.upper()


def read_config(run_dir: Path) -> dict:
    """Read ``summary.json`` if the run wrote one."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    with summary_path.open(encoding="utf-8") as summary_file:
        return json.load(summary_file).get("config", {})


def read_fault_grid(
    diagnostics_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_km, y_km, arrival[y, x], direct_effect[y, x]) on the fault grid.

    Fault nodes are stored with y varying fastest, so the (nx, ny) reshape is
    transposed into the [y, x] layout expected by ``imshow``.
    """
    with np.load(diagnostics_path) as diagnostics:
        fault_x = np.asarray(diagnostics["fault_x"], dtype=float)
        fault_y = np.asarray(diagnostics["fault_y"], dtype=float)
        arrival = np.asarray(diagnostics["rupture_arrival"], dtype=float)
        direct_effect = np.asarray(diagnostics["direct_effect"], dtype=float)

    x_values = np.unique(fault_x)
    y_values = np.unique(fault_y)
    if x_values.size * y_values.size != fault_x.size:
        raise ValueError(
            f"Fault nodes in {diagnostics_path} do not form a regular grid"
        )
    if not np.allclose(fault_y[: y_values.size], y_values):
        raise ValueError(
            f"Unexpected fault node ordering in {diagnostics_path}; "
            "expected y to vary fastest"
        )
    shape = (x_values.size, y_values.size)
    return (
        1.0e-3 * x_values,
        1.0e-3 * y_values,
        arrival.reshape(shape).T,
        direct_effect.reshape(shape).T,
    )


def station_coordinates(station_path: Path) -> tuple[float, float]:
    """Return (along-strike km, down-dip km) encoded in a station file name."""
    match = STATION_NAME.search(station_path.name)
    if match is None:
        raise ValueError(f"Unexpected station file name: {station_path.name}")
    return 0.1 * float(match.group(1)), 0.1 * float(match.group(2))


def read_station_table(station_path: Path) -> np.ndarray:
    """Read a SCEC station file, skipping its ``#`` header and column-name row."""
    rows: list[list[float]] = []
    with station_path.open(encoding="utf-8") as station_file:
        for line in station_file:
            if line.startswith("#") or not line.strip():
                continue
            try:
                rows.append([float(value) for value in line.split()])
            except ValueError:  # the column-name row
                continue
    if not rows:
        raise ValueError(f"Station file contains no data: {station_path}")
    return np.asarray(rows, dtype=float)


def read_stations(run_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Read every on-fault station time series of a run."""
    station_paths = sorted(run_dir.glob("faultst*.txt"))
    if not station_paths:
        raise FileNotFoundError(f"No faultst*.txt station files in {run_dir}")

    stations: list[dict] = []
    times: np.ndarray | None = None
    for station_path in station_paths:
        table = read_station_table(station_path)
        if table.shape[1] <= SHEAR_COLUMN:
            raise ValueError(f"Station file has too few columns: {station_path}")
        station_time = table[:, 0]
        if times is None:
            times = station_time
        elif station_time.shape != times.shape or not np.allclose(
            station_time, times
        ):
            raise ValueError(f"Station time base differs in {station_path}")
        strike_km, dip_km = station_coordinates(station_path)
        stations.append(
            {
                "name": f"{strike_km:+.0f} km / {dip_km:.1f} km",
                "strike_km": strike_km,
                "dip_km": dip_km,
                "shear_mpa": table[:, SHEAR_COLUMN],
            }
        )
    assert times is not None
    stations.sort(key=lambda station: (station["dip_km"], station["strike_km"]))
    return stations, times


def select_frame_indices(times: np.ndarray, interval: float) -> np.ndarray:
    """Sub-sample the station time base at a fixed physical interval."""
    if interval <= 0.0:
        raise ValueError("frame-interval must be positive")
    wanted = np.arange(times[0], times[-1] + 0.5 * interval, interval)
    return np.unique(np.searchsorted(times, wanted).clip(0, times.size - 1))


def render_frames(
    run_dir: Path,
    output_dir: Path,
    *,
    case_name: str,
    width: int,
    height: int,
    dpi: int,
    frame_interval: float,
    vmin: float | None,
    vmax: float | None,
    cmap_name: str,
    overwrite: bool,
) -> tuple[int, float, float]:
    """Render the rupture-front/shear-stress snapshots as 4:3 PNG images."""
    if width * 3 != height * 4:
        raise ValueError(f"PNG size must be 4:3, got {width}x{height}")

    diagnostics_path, problem = find_diagnostics(run_dir)
    config = read_config(run_dir)
    x_km, y_km, arrival, direct_effect = read_fault_grid(diagnostics_path)
    stations, times = read_stations(run_dir)
    frame_indices = select_frame_indices(times, frame_interval)

    ruptured = arrival < UNRUPTURED
    if not ruptured.any():
        raise ValueError(f"No node ever ruptured in {diagnostics_path}")

    shear_stack = np.stack([station["shear_mpa"] for station in stations])
    color_min = float(vmin) if vmin is not None else float(np.floor(shear_stack.min() / 5.0) * 5.0)
    color_max = float(vmax) if vmax is not None else float(np.ceil(shear_stack.max() / 5.0) * 5.0)
    if color_max <= color_min:
        raise ValueError("vmax must be greater than vmin")

    # View window: the ruptured area padded by 3 km, clipped to the fault grid.
    ruptured_x = x_km[np.any(ruptured, axis=0)]
    ruptured_y = y_km[np.any(ruptured, axis=1)]
    view_x = (
        max(x_km[0], ruptured_x.min() - 3.0),
        min(x_km[-1], ruptured_x.max() + 3.0),
    )
    view_y = (
        max(y_km[0], ruptured_y.min() - 3.0),
        min(y_km[-1], ruptured_y.max() + 3.0),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_frames = sorted(output_dir.glob("frame_*.png"))
    if existing_frames and not overwrite:
        raise FileExistsError(
            f"Found {len(existing_frames)} existing frames in {output_dir}; "
            "use --overwrite to replace them"
        )
    for existing_frame in existing_frames:
        existing_frame.unlink()

    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    map_axis = figure.add_axes((0.10, 0.42, 0.62, 0.47))
    colorbar_axis = figure.add_axes((0.755, 0.42, 0.022, 0.47))
    series_axis = figure.add_axes((0.10, 0.08, 0.62, 0.24))

    font_scale = min(width / 800.0, height / 600.0)
    title_font_size = 13.0 * font_scale
    label_font_size = 10.0 * font_scale
    tick_font_size = 9.0 * font_scale

    extent = (
        x_km[0] - 0.5 * (x_km[1] - x_km[0]),
        x_km[-1] + 0.5 * (x_km[1] - x_km[0]),
        y_km[0] - 0.5 * (y_km[1] - y_km[0]),
        y_km[-1] + 0.5 * (y_km[1] - y_km[0]),
    )
    ruptured_mask = np.zeros(arrival.shape, dtype=float)
    ruptured_image = map_axis.imshow(
        ruptured_mask,
        origin="lower",
        extent=extent,
        cmap=plt.get_cmap("Greys"),
        norm=Normalize(vmin=0.0, vmax=1.0),
        interpolation="nearest",
        aspect="equal",
        alpha=0.55,
    )
    front_contour: list = []

    # Velocity-weakening patch (a at its minimum) and hypocenter.
    weakening = direct_effect <= direct_effect.min() + 1.0e-12
    patch_x = x_km[np.any(weakening, axis=0)]
    patch_y = y_km[np.any(weakening, axis=1)]
    map_axis.add_patch(
        Rectangle(
            (patch_x.min(), patch_y.min()),
            patch_x.max() - patch_x.min(),
            patch_y.max() - patch_y.min(),
            fill=False,
            edgecolor="0.25",
            linestyle="--",
            linewidth=1.0 * font_scale,
        )
    )
    hypocenter = (
        1.0e-3 * float(config.get("hypocenter_x", 0.0)),
        1.0e-3 * float(config.get("hypocenter_y", 7500.0)),
    )
    map_axis.plot(
        *hypocenter,
        marker="*",
        markersize=10.0 * font_scale,
        color="crimson",
        markeredgecolor="white",
        markeredgewidth=0.8 * font_scale,
        linestyle="none",
        zorder=6,
    )

    station_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(stations)))
    normalization = Normalize(vmin=color_min, vmax=color_max, clip=True)
    scatter = map_axis.scatter(
        [station["strike_km"] for station in stations],
        [station["dip_km"] for station in stations],
        c=[station["shear_mpa"][0] for station in stations],
        cmap=plt.get_cmap(cmap_name),
        norm=normalization,
        s=(9.0 * font_scale) ** 2,
        edgecolors=station_colors,
        linewidths=1.6 * font_scale,
        zorder=5,
    )

    map_axis.set_xlim(*view_x)
    map_axis.set_ylim(view_y[1], view_y[0])  # down-dip increases downward
    map_axis.set_xlabel(r"along strike $x$ (km)", fontsize=label_font_size)
    map_axis.set_ylabel(r"down dip $y$ (km)", fontsize=label_font_size)
    map_axis.tick_params(labelsize=tick_font_size)

    colorbar = figure.colorbar(scatter, cax=colorbar_axis)
    colorbar.set_label(
        r"Horizontal shear stress $\tau_x$ (MPa)", fontsize=label_font_size
    )
    colorbar.ax.tick_params(labelsize=tick_font_size)

    for station, color in zip(stations, station_colors):
        series_axis.plot(
            times,
            station["shear_mpa"],
            color=color,
            linewidth=0.9 * font_scale,
            label=station["name"],
        )
    cursor = series_axis.axvline(
        times[0], color="0.2", linewidth=1.0 * font_scale, zorder=3
    )
    cursor_dots = series_axis.scatter(
        np.full(len(stations), times[0]),
        shear_stack[:, 0],
        s=(4.0 * font_scale) ** 2,
        facecolors=station_colors,
        edgecolors="none",
        zorder=4,
    )
    series_axis.set_xlim(times[0], times[-1])
    series_padding = 0.05 * (shear_stack.max() - shear_stack.min())
    series_axis.set_ylim(
        shear_stack.min() - series_padding, shear_stack.max() + series_padding
    )
    series_axis.set_xlabel(r"time $t$ (s)", fontsize=label_font_size)
    series_axis.set_ylabel(r"$\tau_x$ (MPa)", fontsize=label_font_size)
    series_axis.tick_params(labelsize=tick_font_size)
    series_axis.grid(alpha=0.25, linewidth=0.6 * font_scale)
    series_axis.legend(
        title="station: along strike / down dip",
        fontsize=0.62 * label_font_size,
        title_fontsize=0.62 * label_font_size,
        ncol=3,
        loc="upper right",
        framealpha=0.85,
    )

    legend_handles = [
        Line2D([], [], marker="s", linestyle="none", color="0.55", label="ruptured area"),
        Line2D([], [], linestyle="-", color="crimson", label="rupture front"),
        Line2D([], [], linestyle="--", color="0.25", label="velocity-weakening patch"),
        Line2D([], [], marker="*", linestyle="none", color="crimson", label="hypocenter"),
    ]
    map_axis.legend(
        handles=legend_handles,
        fontsize=0.62 * label_font_size,
        loc="lower right",
        framealpha=0.85,
    )

    for frame_number, time_index in enumerate(frame_indices):
        time_seconds = float(times[time_index])
        ruptured_now = ruptured & (arrival <= time_seconds)
        ruptured_image.set_data(ruptured_now.astype(float))
        for collection in front_contour:
            collection.remove()
        front_contour.clear()
        if ruptured_now.any() and not ruptured_now.all():
            contour_set = map_axis.contour(
                x_km,
                y_km,
                np.where(ruptured, arrival, np.nan),
                levels=[time_seconds],
                colors="crimson",
                linewidths=1.2 * font_scale,
            )
            front_contour.extend(
                getattr(contour_set, "collections", [contour_set])
            )
        scatter.set_array(shear_stack[:, time_index])
        cursor.set_xdata([time_seconds, time_seconds])
        cursor_dots.set_offsets(
            np.column_stack(
                (np.full(len(stations), time_seconds), shear_stack[:, time_index])
            )
        )
        map_axis.set_title(
            f"{case_name} — rupture front & on-fault $\\tau_x$    "
            f"t = {time_seconds:.3f} s",
            fontsize=title_font_size,
            pad=6.0 * font_scale,
        )
        frame_path = output_dir / f"frame_{frame_number:04d}.png"
        figure.savefig(frame_path, dpi=dpi, facecolor="white")
        if frame_number == 0 or (frame_number + 1) % 10 == 0 or frame_number + 1 == len(frame_indices):
            print(f"Rendered {frame_number + 1:>3}/{len(frame_indices)}: {frame_path.name}")

    plt.close(figure)
    print(f"Problem: {problem}")
    print(f"Diagnostics: {diagnostics_path}")
    print(f"Stations: {len(stations)} on-fault time series, {times.size} samples")
    print(
        f"Fault grid: {x_km.size} x {y_km.size} nodes; view "
        f"x [{view_x[0]:g}, {view_x[1]:g}] km, y [{view_y[0]:g}, {view_y[1]:g}] km"
    )
    print(f"Ruptured nodes: {int(ruptured.sum())} of {arrival.size}")
    print(f"Fixed color range: {color_min:g} to {color_max:g} MPa")
    return len(frame_indices), float(times[frame_indices[0]]), float(times[frame_indices[-1]])


def encode_video(
    output_dir: Path,
    output_path: Path,
    *,
    frame_count: int,
    fps: int,
    overwrite: bool,
) -> None:
    """Assemble sequential PNGs into an H.264 MP4 using FFmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg is not available on PATH")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Video already exists: {output_path}")

    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(output_dir / "frame_%04d.png"),
        "-frames:v", str(frame_count),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-r", str(fps),
        "-movflags", "+faststart",
        str(output_path),
    ]
    print("Encoding video with FFmpeg...")
    subprocess.run(command, check=True)
    print(f"Saved video: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate a Tatva FEM TPV run (rupture front + on-fault shear stress)."
    )
    parser.add_argument("run_dir", type=Path, help="FEM output directory of one run")
    parser.add_argument("--output-dir", type=Path, help="PNG/video directory")
    parser.add_argument("--video", type=Path, help="MP4 path")
    parser.add_argument("--case-name", help="Title prefix (default: <PROBLEM> FEM)")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Video frame rate")
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=DEFAULT_FRAME_INTERVAL,
        help="Physical time between frames (s)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--vmin", type=float, help="Fixed lower color limit (MPa)")
    parser.add_argument("--vmax", type=float, help="Fixed upper color limit (MPa)")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument("--png-only", action="store_true", help="Skip FFmpeg")
    parser.add_argument("--keep-png", action="store_true", help="Keep frame_*.png")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_dir = arguments.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory does not exist: {run_dir}")
    _, problem = find_diagnostics(run_dir)
    case_name = arguments.case_name or f"{problem} (Tatva FEM)"

    default_analysis_dir = Path(__file__).resolve().parent.parent
    output_dir = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else default_analysis_dir / f"{problem}_FEM"
    )
    video_path = (
        arguments.video.expanduser().resolve()
        if arguments.video is not None
        else output_dir / f"{run_dir.name}_rupture_{arguments.fps}fps.mp4"
    )

    frame_count, first_time, last_time = render_frames(
        run_dir,
        output_dir,
        case_name=case_name,
        width=arguments.width,
        height=arguments.height,
        dpi=arguments.dpi,
        frame_interval=arguments.frame_interval,
        vmin=arguments.vmin,
        vmax=arguments.vmax,
        cmap_name=arguments.cmap,
        overwrite=arguments.overwrite,
    )
    print(f"Physical time: {first_time:.6f} to {last_time:.6f} s ({frame_count} frames)")

    if not arguments.png_only:
        encode_video(
            output_dir,
            video_path,
            frame_count=frame_count,
            fps=arguments.fps,
            overwrite=arguments.overwrite,
        )
        print(f"Playback duration: {frame_count / arguments.fps:.3f} s")
        if not arguments.keep_png:
            rendered_frames = sorted(output_dir.glob("frame_*.png"))
            for rendered_frame in rendered_frames:
                rendered_frame.unlink()
            print(f"Deleted {len(rendered_frames)} rendered PNG frames")


if __name__ == "__main__":
    main()
