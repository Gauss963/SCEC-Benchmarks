#!/usr/bin/env python3
"""Write UGUCA-compatible binary interface dumps from the Tatva TPV drivers.

UGUCA (``Dumper::Format::Binary``) describes one dump with four metadata files
next to a directory of raw field data::

    <base>.info    keys pointing at the other three files and the data folder
    <base>.fields  "<field name> <file name>" per registered field
    <base>.time    "<time step> <physical time>" per dumped frame
    <base>.coord   "x y z" per interface node, ASCII, %.10e
    <base>-DataFiles/<field>.out   little-endian float32, one frame after another

Nodes are ordered with the second in-plane axis varying fastest (in UGUCA: z
fastest, x slowest), which is exactly how the Tatva fault plane is raveled
(``np.meshgrid(x, y, indexing="ij")``), so the Tatva down-dip axis y maps onto
the UGUCA z axis.

Writing this format lets the same analysis scripts read UGUCA and Tatva output.
The field names mirror ``benchmarks/TPV101/TPV101.cc`` so the semantics match:

===============  ====================================================
``cohesion_0``   interface shear traction along strike, Pa
``top_disp_0``   top-side displacement along strike, m (half the slip)
``top_velo_0``   top-side velocity along strike, m/s (half the slip rate)
``theta``        rate-and-state state variable, s
===============  ====================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


FIELD_NAMES: tuple[str, ...] = ("cohesion_0", "top_disp_0", "top_velo_0", "theta")
COORDINATE_FORMAT = "%.10e"
UNIFORM_SPACING_TOLERANCE = 1.0e-6


def _uniform_spacing(axis: np.ndarray, name: str) -> float:
    """Return the constant spacing of ``axis`` or raise if it is graded."""
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"The {name} dump axis needs at least two nodes.")
    spacings = np.diff(axis)
    spacing = float(spacings[0])
    if spacing <= 0.0:
        raise ValueError(f"The {name} dump axis must increase monotonically.")
    if float(np.max(np.abs(spacings - spacing))) > UNIFORM_SPACING_TOLERANCE * spacing:
        raise ValueError(
            f"The {name} dump axis is not uniformly spaced; restrict the dump "
            "window to the fine part of the graded mesh."
        )
    return spacing


class UgucaDumper:
    """Append interface snapshots to a UGUCA binary dump."""

    def __init__(
        self,
        output_dir: Path,
        base_name: str,
        x: np.ndarray,
        z: np.ndarray,
        *,
        field_names: tuple[str, ...] = FIELD_NAMES,
        append: bool = False,
    ) -> None:
        self.base_name = base_name
        self.field_names = tuple(field_names)
        self.x = np.asarray(x, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        self.spacing_x = _uniform_spacing(self.x, "x")
        self.spacing_z = _uniform_spacing(self.z, "z")
        self.nb_nodes = self.x.size * self.z.size

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_path = output_dir / base_name
        self.data_dir = output_dir / f"{base_name}-DataFiles"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.time_path = Path(f"{self.base_path}.time")

        self._write_info()
        self._write_fields()
        if not append or not Path(f"{self.base_path}.coord").is_file():
            self._write_coordinates()
        if not append:
            self.time_path.write_text("", encoding="utf-8")

        mode = "ab" if append else "wb"
        self._streams = {
            name: open(self.data_dir / f"{name}.out", mode) for name in self.field_names
        }
        self.frames_written = self._count_time_rows()

    # -- metadata ----------------------------------------------------------
    def _write_info(self) -> None:
        lines = [
            f"field_description {self.base_name}.fields",
            f"time_description {self.base_name}.time",
            f"coord_description {self.base_name}.coord",
            f"folder_name {self.base_name}-DataFiles",
            "output_format binary",
        ]
        Path(f"{self.base_path}.info").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_fields(self) -> None:
        lines = [f"{name} {name}.out" for name in self.field_names]
        Path(f"{self.base_path}.fields").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _write_coordinates(self) -> None:
        """Write one ``x y z`` line per node, z varying fastest."""
        x_grid, z_grid = np.meshgrid(self.x, self.z, indexing="ij")
        coordinates = np.column_stack(
            (
                x_grid.ravel(),
                np.zeros(self.nb_nodes, dtype=np.float64),
                z_grid.ravel(),
            )
        )
        np.savetxt(
            Path(f"{self.base_path}.coord"), coordinates, fmt=COORDINATE_FORMAT
        )

    def _count_time_rows(self) -> int:
        if not self.time_path.is_file():
            return 0
        with self.time_path.open(encoding="utf-8") as time_file:
            return sum(1 for line in time_file if line.strip())

    # -- frames ------------------------------------------------------------
    def dump(self, step: int, time_seconds: float, fields: dict[str, np.ndarray]) -> None:
        """Append one frame; ``fields`` holds one flat array per field name."""
        missing = [name for name in self.field_names if name not in fields]
        if missing:
            raise KeyError(f"Missing dump fields: {', '.join(missing)}")
        for name in self.field_names:
            values = np.ascontiguousarray(fields[name], dtype="<f4")
            if values.size != self.nb_nodes:
                raise ValueError(
                    f"Field {name} has {values.size} values, expected {self.nb_nodes}"
                )
            self._streams[name].write(values.tobytes())
            self._streams[name].flush()
        with self.time_path.open("a", encoding="utf-8") as time_file:
            time_file.write(f"{int(step)} {float(time_seconds):.10e}\n")
        self.frames_written += 1

    def truncate_to_time(self, time_seconds: float) -> int:
        """Drop frames past ``time_seconds`` so a resumed run stays consistent."""
        if not self.time_path.is_file():
            return 0
        rows = [
            line
            for line in self.time_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kept = [
            line
            for line in rows
            if float(line.split()[1]) <= time_seconds + 1.0e-9
        ]
        if len(kept) != len(rows):
            self.time_path.write_text(
                "".join(f"{line}\n" for line in kept), encoding="utf-8"
            )
        frame_bytes = self.nb_nodes * 4
        for name in self.field_names:
            stream = self._streams[name]
            stream.flush()
            os.truncate(stream.fileno(), len(kept) * frame_bytes)
            stream.seek(0, os.SEEK_END)
        self.frames_written = len(kept)
        return self.frames_written

    def close(self) -> None:
        for stream in self._streams.values():
            stream.flush()
            stream.close()
        self._streams = {}

    def __enter__(self) -> "UgucaDumper":
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


def window_indices(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the fault sub-grid inside a window.

    Returns the window's x nodes, y nodes, and the flat indices into a fault
    array raveled as ``np.meshgrid(x, y, indexing="ij")``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tolerance = 1.0e-6 * max(1.0, float(np.max(np.abs(x))))
    x_selection = np.flatnonzero((x >= x_min - tolerance) & (x <= x_max + tolerance))
    y_selection = np.flatnonzero((y >= y_min - tolerance) & (y <= y_max + tolerance))
    if x_selection.size < 2 or y_selection.size < 2:
        raise ValueError(
            f"The dump window x=[{x_min}, {x_max}], y=[{y_min}, {y_max}] "
            "contains fewer than two nodes per axis."
        )
    flat = (
        x_selection[:, None] * y.size + y_selection[None, :]
    ).ravel()
    return x[x_selection], y[y_selection], flat
