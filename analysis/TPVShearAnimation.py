#!/usr/bin/env python3
"""Animate the horizontal shear stress on a TPV rupture plane.

The same UGUCA binary dump layout is produced by

* `benchmarks/TPV10x` in UGUCA, and
* `tpv10x.py --uguca-dump-interval ...` in the Tatva FEM validation
  (https://github.com/Gauss963/TPV101),

so one script animates both and the two can be compared frame by frame.

Everything is drawn in the SCEC TPV101/TPV102 fault convention: distance along
strike `j` on the horizontal axis, distance down dip `k` on the vertical axis
increasing downward, with `k = 0` at the top of the 30 x 15 km rupture patch
(the free surface for TPV102) and the hypocenter at `j = 0`, `k = 7.5 km`.
Each dump's own coordinates are mapped into that frame by `--convention`:

===================== ==================================================
`tatva`               already SCEC: `j = x`, `k = z` (z holds down dip)
`uguca-fullspace`     `j = x - Lx/2`, `k = z - (Lz/2 - patch half width)`
`uguca-freesurface`   `j = x - Lx/2`, `k = Lz/2 - z`; the mirror image the
                      benchmark uses to make the free surface is dropped
===================== ==================================================

The figure follows the rupture videos: the shear-stress field on the fault, the
velocity-weakening patch and hypocenter, the nine SCEC on-fault stations, and a
time-series panel with a cursor. Station curves come from `faultst*.txt` when
they sit next to the dump, and are sampled from the dumped field otherwise.

Example
-------
python Analysis/code/TPVShearAnimation.py \
    /Volumes/Gauss-T7/UGUCA-Dump/TPV101/TPV101_Nx1440_Nz720_s2.00_tf0.35_npc1 \
    --output-dir Analysis/TPV101 --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


DEFAULT_WIDTH = 2400
DEFAULT_HEIGHT = 1800
DEFAULT_DPI = 100
DEFAULT_PLOT_NODES = 512
DEFAULT_FPS = 12
SHEAR_FIELD = "cohesion_0"

# SCEC TPV101/TPV102 fault geometry, in km.
PATCH_HALF_LENGTH = 15.0
PATCH_HALF_WIDTH = 7.5
HYPOCENTER = (0.0, 7.5)
STATIONS = (
    ("faultst-120dp030", -12.0, 3.0),
    ("faultst000dp030", 0.0, 3.0),
    ("faultst120dp030", 12.0, 3.0),
    ("faultst-090dp075", -9.0, 7.5),
    ("faultst000dp075", 0.0, 7.5),
    ("faultst090dp075", 9.0, 7.5),
    ("faultst-120dp120", -12.0, 12.0),
    ("faultst000dp120", 0.0, 12.0),
    ("faultst120dp120", 12.0, 12.0),
)
STATION_SHEAR_COLUMN = 3  # column #4 of a SCEC on-fault file, in MPa


def normalize_dump_base(path: Path) -> Path:
    """Return the dump prefix, accepting a prefix or any metadata path."""
    resolved = path.expanduser().resolve()
    for suffix in (".info", ".time", ".fields", ".coord"):
        if resolved.name.endswith(suffix):
            return resolved.with_name(resolved.name[: -len(suffix)])
    if resolved.name.endswith("-DataFiles"):
        return resolved.with_name(resolved.name[: -len("-DataFiles")])
    return resolved


def read_info(dump_base: Path) -> dict[str, str]:
    """Read the UGUCA ``.info`` file into a key/value dictionary."""
    info_path = Path(f"{dump_base}.info")
    if not info_path.is_file():
        raise FileNotFoundError(f"Dump description does not exist: {info_path}")
    info: dict[str, str] = {}
    with info_path.open(encoding="utf-8") as info_file:
        for line in info_file:
            parts = line.split()
            if len(parts) == 2:
                info[parts[0]] = parts[1]
    output_format = info.get("output_format", "binary")
    if output_format != "binary":
        raise ValueError(f"Only binary dumps are supported, got {output_format!r}")
    return info


def field_file_from_metadata(dump_base: Path, field_name: str) -> Path:
    """Resolve a field output using the UGUCA ``.fields`` description."""
    fields_path = Path(f"{dump_base}.fields")
    if not fields_path.is_file():
        raise FileNotFoundError(f"Field description does not exist: {fields_path}")
    fields: dict[str, str] = {}
    with fields_path.open(encoding="utf-8") as fields_file:
        for line in fields_file:
            parts = line.split()
            if len(parts) == 2:
                fields[parts[0]] = parts[1]
    if field_name not in fields:
        raise KeyError(
            f"Field {field_name!r} is not dumped; available: "
            + ", ".join(sorted(fields))
        )
    field_path = Path(f"{dump_base}-DataFiles") / fields[field_name]
    if not field_path.is_file():
        raise FileNotFoundError(f"Field data does not exist: {field_path}")
    return field_path


def read_times(dump_base: Path) -> np.ndarray:
    """Read the physical time column from a UGUCA ``.time`` file."""
    time_path = Path(f"{dump_base}.time")
    if not time_path.is_file():
        raise FileNotFoundError(f"Time description does not exist: {time_path}")
    time_table = np.loadtxt(time_path, ndmin=2)
    if len(time_table) == 0 or time_table.shape[1] < 2:
        raise ValueError(f"Invalid UGUCA time file: {time_path}")
    return np.asarray(time_table[:, 1], dtype=float)


def read_mesh(dump_base: Path, nb_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover the dump's x and z node coordinates in km from ``.coord``.

    Nodes are written with z varying fastest, so the number of z nodes is the
    number of leading lines that share the same x coordinate.
    """
    coord_path = Path(f"{dump_base}.coord")
    if not coord_path.is_file():
        raise FileNotFoundError(f"Coordinate description does not exist: {coord_path}")

    x_values: list[float] = []
    z_values: list[float] = []
    with coord_path.open(encoding="utf-8") as coord_file:
        for line in coord_file:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid coordinate line in {coord_path}: {line!r}")
            x_values.append(float(parts[0]))
            z_values.append(float(parts[2]))
            if len(x_values) > 1 and x_values[-1] != x_values[0]:
                break
    if len(x_values) < 3 or x_values[-1] == x_values[0]:
        raise ValueError(f"Could not detect the grid layout from {coord_path}")

    nb_nodes_z = len(x_values) - 1
    if nb_nodes % nb_nodes_z != 0:
        raise ValueError(
            f"Node count {nb_nodes} is not a multiple of nb_nodes_z={nb_nodes_z}"
        )
    nb_nodes_x = nb_nodes // nb_nodes_z
    spacing_z = z_values[1] - z_values[0]
    spacing_x = x_values[-1] - x_values[0]
    if spacing_x <= 0.0 or spacing_z <= 0.0:
        raise ValueError(f"Non-positive grid spacing detected in {coord_path}")
    x_km = 1.0e-3 * (x_values[0] + spacing_x * np.arange(nb_nodes_x))
    z_km = 1.0e-3 * (z_values[0] + spacing_z * np.arange(nb_nodes_z))
    return x_km, z_km


def detect_convention(dump_base: Path, x_km: np.ndarray) -> str:
    """Guess the coordinate convention of a dump."""
    if float(x_km.min()) < 0.0:
        return "tatva"
    return "uguca-freesurface" if "102" in dump_base.name else "uguca-fullspace"


def scec_axes(
    convention: str, x_km: np.ndarray, z_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Map a dump's own axes onto SCEC (j, k), in km.

    Returns the along-strike axis, the down-dip axis, whether the down-dip axis
    had to be reversed relative to the stored order, and whether the dump holds
    a mirror image that must be cropped away.
    """
    spacing_x = float(x_km[1] - x_km[0])
    spacing_z = float(z_km[1] - z_km[0])
    length_x = x_km.size * spacing_x
    length_z = z_km.size * spacing_z

    if convention == "tatva":
        return x_km, z_km, False, False
    if convention == "uguca-fullspace":
        j = x_km - 0.5 * length_x
        k = z_km - (0.5 * length_z - PATCH_HALF_WIDTH)
        return j, k, False, False
    if convention == "uguca-freesurface":
        j = x_km - 0.5 * length_x
        k = 0.5 * length_z - z_km
        return j, k[::-1], True, True
    raise ValueError(f"Unknown convention: {convention}")


def read_station_table(station_path: Path) -> np.ndarray:
    """Read a SCEC on-fault station file, skipping its header rows."""
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


def station_series(
    stations_dir: Path | None,
    times: np.ndarray,
    sample: "callable",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (station times, shear stress [station, time], its source).

    SCEC ``faultst*.txt`` files are preferred because they carry the solver's
    own output cadence; otherwise the dumped field is sampled at the station
    positions.
    """
    if stations_dir is not None:
        paths = {name: stations_dir / f"{name}.txt" for name, _, _ in STATIONS}
        if all(path.is_file() for path in paths.values()):
            tables = [read_station_table(paths[name]) for name, _, _ in STATIONS]
            station_times = tables[0][:, 0]
            shear = np.stack([table[:, STATION_SHEAR_COLUMN] for table in tables])
            return station_times, shear, f"{stations_dir}/faultst*.txt"
    shear = np.stack([sample(index) for index in range(times.size)], axis=1)
    return times, shear, "sampled from the dumped field"


def render_frames(
    dump_base: Path,
    output_dir: Path,
    *,
    field_name: str,
    convention: str,
    case_name: str,
    stations_dir: Path | None,
    window: tuple[float, float, float, float],
    width: int,
    height: int,
    dpi: int,
    plot_nodes: int,
    vmin: float | None,
    vmax: float | None,
    cmap_name: str,
    overwrite: bool,
) -> tuple[int, float, float]:
    """Render every dumped snapshot as a sequential 4:3 PNG image."""
    if width * 3 != height * 4:
        raise ValueError(f"PNG size must be 4:3, got {width}x{height}")
    if plot_nodes < 2:
        raise ValueError("plot-nodes must be at least 2")

    read_info(dump_base)
    times = read_times(dump_base)
    field_path = field_file_from_metadata(dump_base, field_name)
    raw_field = np.memmap(field_path, dtype="<f4", mode="r")
    if raw_field.size % times.size != 0:
        raise ValueError(
            f"Field size mismatch: {raw_field.size} values over {times.size} frames"
        )
    nb_nodes = raw_field.size // times.size
    x_km, z_km = read_mesh(dump_base, nb_nodes)
    frames = raw_field.reshape(times.size, x_km.size, z_km.size)

    j_axis, k_axis, k_reversed, has_mirror = scec_axes(convention, x_km, z_km)
    j_min, j_max, k_min, k_max = window
    j_selection = np.flatnonzero((j_axis >= j_min - 1e-9) & (j_axis <= j_max + 1e-9))
    k_selection = np.flatnonzero((k_axis >= k_min - 1e-9) & (k_axis <= k_max + 1e-9))
    if j_selection.size < 2 or k_selection.size < 2:
        raise ValueError(f"The window {window} keeps fewer than two nodes per axis")
    stride = max(
        1, int(np.ceil(max(j_selection.size, k_selection.size) / plot_nodes))
    )
    j_selection = j_selection[::stride]
    k_selection = k_selection[::stride]
    j_view = j_axis[j_selection]
    k_view = k_axis[k_selection]

    def frame_at(index: int) -> np.ndarray:
        """One frame in MPa, oriented [k, j] and cropped to the window."""
        frame = np.asarray(frames[index], dtype=float).T  # [z, x]
        if k_reversed:
            frame = frame[::-1]
        return frame[np.ix_(k_selection, j_selection)] * 1.0e-6

    station_rows = np.array(
        [int(np.argmin(np.abs(k_axis - k))) for _, _, k in STATIONS]
    )
    station_columns = np.array(
        [int(np.argmin(np.abs(j_axis - j))) for _, j, _ in STATIONS]
    )

    def sample(index: int) -> np.ndarray:
        """Nearest-node value at every station for one frame, in MPa."""
        frame = np.asarray(frames[index], dtype=float).T
        if k_reversed:
            frame = frame[::-1]
        return frame[station_rows, station_columns] * 1.0e-6

    station_times, station_shear, station_source = station_series(
        stations_dir, times, sample
    )

    if vmin is None or vmax is None:
        print("Scanning rendered frames for a fixed color range...")
        sampled_min, sampled_max = np.inf, -np.inf
        for index in range(times.size):
            frame = frame_at(index)
            sampled_min = min(sampled_min, float(np.min(frame)))
            sampled_max = max(sampled_max, float(np.max(frame)))
        print(f"Rendered-grid shear stress: {sampled_min:.6g} to {sampled_max:.6g} MPa")
        rounded_min = float(np.floor(sampled_min / 5.0) * 5.0)
        rounded_max = float(np.ceil(sampled_max / 5.0) * 5.0)
    else:
        rounded_min, rounded_max = 0.0, 0.0
    color_min = float(vmin) if vmin is not None else rounded_min
    color_max = float(vmax) if vmax is not None else rounded_max
    if color_max <= color_min:
        raise ValueError("vmax must be greater than vmin")

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

    spacing_j = float(j_view[1] - j_view[0])
    spacing_k = float(k_view[1] - k_view[0])
    extent = (
        j_view[0] - 0.5 * spacing_j,
        j_view[-1] + 0.5 * spacing_j,
        k_view[0] - 0.5 * spacing_k,
        k_view[-1] + 0.5 * spacing_k,
    )
    normalization = Normalize(vmin=color_min, vmax=color_max, clip=True)
    image = map_axis.imshow(
        frame_at(0),
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        cmap=plt.get_cmap(cmap_name),
        norm=normalization,
        aspect="equal",
    )
    map_axis.set_xlim(extent[0], extent[1])
    map_axis.set_ylim(extent[3], extent[2])  # down dip increases downward
    map_axis.set_xlabel(r"along strike $j$ (km)", fontsize=label_font_size)
    map_axis.set_ylabel(r"down dip $k$ (km)", fontsize=label_font_size)
    map_axis.tick_params(labelsize=tick_font_size)
    for spine in map_axis.spines.values():
        spine.set_linewidth(0.8 * font_scale)

    map_axis.add_patch(
        Rectangle(
            (-PATCH_HALF_LENGTH, 0.0),
            2.0 * PATCH_HALF_LENGTH,
            2.0 * PATCH_HALF_WIDTH,
            fill=False,
            edgecolor="white",
            linestyle="--",
            linewidth=1.0 * font_scale,
            alpha=0.85,
        )
    )
    map_axis.plot(
        *HYPOCENTER,
        marker="*",
        markersize=10.0 * font_scale,
        color="crimson",
        markeredgecolor="white",
        markeredgewidth=0.8 * font_scale,
        linestyle="none",
        zorder=6,
    )
    station_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(STATIONS)))
    for (_, j_position, k_position), color in zip(STATIONS, station_colors):
        map_axis.plot(
            j_position,
            k_position,
            marker="o",
            markersize=6.0 * font_scale,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.6 * font_scale,
            linestyle="none",
            zorder=5,
        )

    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(
        r"Horizontal shear stress $\tau_x$ (MPa)", fontsize=label_font_size
    )
    colorbar.ax.tick_params(labelsize=tick_font_size)
    colorbar.outline.set_linewidth(0.8 * font_scale)

    for position, ((_, j_position, k_position), color) in enumerate(
        zip(STATIONS, station_colors)
    ):
        series_axis.plot(
            station_times,
            station_shear[position],
            color=color,
            linewidth=0.9 * font_scale,
            label=f"{j_position:+.0f} km / {k_position:.1f} km",
        )
    cursor = series_axis.axvline(
        times[0], color="0.2", linewidth=1.0 * font_scale, zorder=3
    )
    cursor_dots = series_axis.scatter(
        np.full(len(STATIONS), times[0]),
        station_shear[:, 0],
        s=(4.0 * font_scale) ** 2,
        facecolors=station_colors,
        edgecolors="none",
        zorder=4,
    )
    padding = 0.05 * float(station_shear.max() - station_shear.min())
    series_axis.set_xlim(float(times[0]), float(times[-1]))
    series_axis.set_ylim(
        float(station_shear.min()) - padding, float(station_shear.max()) + padding
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

    for index, time_seconds in enumerate(times):
        image.set_data(frame_at(index))
        cursor.set_xdata([time_seconds, time_seconds])
        column = int(np.argmin(np.abs(station_times - time_seconds)))
        cursor_dots.set_offsets(
            np.column_stack(
                (np.full(len(STATIONS), time_seconds), station_shear[:, column])
            )
        )
        map_axis.set_title(
            f"{case_name} — horizontal shear stress    t = {time_seconds:.3f} s",
            fontsize=title_font_size,
            pad=6.0 * font_scale,
        )
        frame_path = output_dir / f"frame_{index:04d}.png"
        figure.savefig(frame_path, dpi=dpi, facecolor="white")
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == times.size:
            print(f"Rendered {index + 1:>3}/{times.size}: {frame_path.name}")

    plt.close(figure)
    print(f"Dump field: {field_path}")
    print(f"Convention: {convention}" + (" (mirror image cropped)" if has_mirror else ""))
    print(
        f"Dumped grid: {x_km.size} x {z_km.size} nodes; rendered window "
        f"j [{j_view[0]:g}, {j_view[-1]:g}] km, k [{k_view[0]:g}, {k_view[-1]:g}] km "
        f"at {j_view.size} x {k_view.size}"
    )
    print(f"Stations: {station_source}")
    print(f"Fixed color range: {color_min:g} to {color_max:g} MPa")
    return times.size, float(times[0]), float(times[-1])


def encode_video(
    output_dir: Path,
    output_path: Path,
    *,
    frame_count: int,
    fps: int,
    overwrite: bool,
) -> None:
    """Assemble sequential PNGs into an H.264 MP4 using FFmpeg."""
    if fps <= 0:
        raise ValueError("fps must be positive")
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
        description="Animate the horizontal shear stress of a UGUCA or Tatva TPV dump."
    )
    parser.add_argument(
        "dump_base",
        type=Path,
        help="Dump prefix (or one of its .info/.time/.fields/.coord paths)",
    )
    parser.add_argument("--output-dir", type=Path, help="PNG/video directory")
    parser.add_argument("--video", type=Path, help="MP4 path")
    parser.add_argument("--case-name", help="Title prefix")
    parser.add_argument(
        "--convention",
        choices=("auto", "tatva", "uguca-fullspace", "uguca-freesurface"),
        default="auto",
        help="How the dump's coordinates map onto the SCEC fault axes",
    )
    parser.add_argument(
        "--stations-dir",
        type=Path,
        help="Directory holding SCEC faultst*.txt files (default: beside the dump)",
    )
    parser.add_argument(
        "--field", default=SHEAR_FIELD, help=f"Dumped field (default: {SHEAR_FIELD})"
    )
    parser.add_argument("--j-min", type=float, default=-18.0, help="Window, km")
    parser.add_argument("--j-max", type=float, default=18.0, help="Window, km")
    parser.add_argument(
        "--k-min",
        type=float,
        help="Window, km (default: 0 with a free surface, -3 otherwise)",
    )
    parser.add_argument("--k-max", type=float, default=18.0, help="Window, km")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--plot-nodes", type=int, default=DEFAULT_PLOT_NODES)
    parser.add_argument("--vmin", type=float, help="Fixed lower color limit (MPa)")
    parser.add_argument("--vmax", type=float, help="Fixed upper color limit (MPa)")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--png-only", action="store_true")
    parser.add_argument("--keep-png", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dump_base = normalize_dump_base(arguments.dump_base)
    case_name = arguments.case_name or re.split(r"[_-]", dump_base.name)[0]

    times = read_times(dump_base)
    field_path = field_file_from_metadata(dump_base, arguments.field)
    nb_nodes = np.memmap(field_path, dtype="<f4", mode="r").size // times.size
    x_km, _ = read_mesh(dump_base, nb_nodes)
    convention = (
        detect_convention(dump_base, x_km)
        if arguments.convention == "auto"
        else arguments.convention
    )
    k_min = arguments.k_min
    if k_min is None:
        k_min = 0.0 if convention == "uguca-freesurface" else -3.0

    stations_dir = arguments.stations_dir or dump_base.parent
    default_analysis_dir = Path(__file__).resolve().parent.parent
    output_dir = (
        arguments.output_dir.expanduser().resolve()
        if arguments.output_dir is not None
        else default_analysis_dir / case_name
    )
    video_path = (
        arguments.video.expanduser().resolve()
        if arguments.video is not None
        else output_dir / f"{dump_base.name}_shear_stress_{arguments.fps}fps.mp4"
    )

    frame_count, first_time, last_time = render_frames(
        dump_base,
        output_dir,
        field_name=arguments.field,
        convention=convention,
        case_name=case_name,
        stations_dir=stations_dir,
        window=(arguments.j_min, arguments.j_max, k_min, arguments.k_max),
        width=arguments.width,
        height=arguments.height,
        dpi=arguments.dpi,
        plot_nodes=arguments.plot_nodes,
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
            rendered = sorted(output_dir.glob("frame_*.png"))
            for frame in rendered:
                frame.unlink()
            print(f"Deleted {len(rendered)} rendered PNG frames")


if __name__ == "__main__":
    main()
