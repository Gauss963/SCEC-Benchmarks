#!/usr/bin/env python3
"""SCEC TPV101 dynamic-rupture benchmark using Tatva FEM operators."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tatva import Mesh, Operator
from tatva.element import Hexahedron8
from tatva.friction import (
    regularized_rate_state_initial_state,
    regularized_rate_state_strength,
    update_ageing_state,
)

from uguca_dump import UgucaDumper, window_indices


STATIONS = (
    ("faultst-120dp030", -12_000.0, 3_000.0),
    ("faultst000dp030", 0.0, 3_000.0),
    ("faultst120dp030", 12_000.0, 3_000.0),
    ("faultst-090dp075", -9_000.0, 7_500.0),
    ("faultst000dp075", 0.0, 7_500.0),
    ("faultst090dp075", 9_000.0, 7_500.0),
    ("faultst-120dp120", -12_000.0, 12_000.0),
    ("faultst000dp120", 0.0, 12_000.0),
    ("faultst120dp120", 12_000.0, 12_000.0),
)

TIME_SERIES_FIELDS = (
    "t",
    "h-slip",
    "h-slip-rate",
    "h-shear-stress",
    "v-slip",
    "v-slip-rate",
    "v-shear-stress",
    "n-stress",
    "log-theta",
)

SURFACE_STATIONS = (
    ("body-060st-120dp000", -6_000.0, -12_000.0),
    ("body-090st000dp000", -9_000.0, 0.0),
    ("body-060st120dp000", -6_000.0, 12_000.0),
    ("body060st-120dp000", 6_000.0, -12_000.0),
    ("body090st000dp000", 9_000.0, 0.0),
    ("body060st120dp000", 6_000.0, 12_000.0),
)

SURFACE_TIME_SERIES_FIELDS = (
    "t",
    "h-disp",
    "h-vel",
    "v-disp",
    "v-vel",
    "n-disp",
    "n-vel",
)


@dataclass(frozen=True)
class TPV101Config:
    mesh_size: float = 1_000.0
    graded_mesh: bool = False
    fine_x_min: float | None = None
    fine_x_max: float | None = None
    fine_y_min: float | None = None
    fine_y_max: float | None = None
    fine_z_extent: float | None = None
    max_mesh_size: float | None = None
    mesh_growth_ratio: float = 1.0
    z_extent: float = 8_000.0
    x_min: float = -26_000.0
    x_max: float = 26_000.0
    y_min: float = -12_000.0
    y_max: float = 27_000.0
    duration: float = 12.0
    output_dt: float = 0.01
    cfl: float = 0.30
    normal_penalty_factor: float = 20.0
    operator_batch_size: int = 2048
    symmetry_reduced: bool = False
    density: float = 2670.0
    shear_wave_speed: float = 3_464.0
    pressure_wave_speed: float = 6_000.0
    normal_stress: float = 120.0e6
    initial_shear_stress: float = 75.0e6
    initial_slip_rate: float = 1.0e-12
    reference_friction: float = 0.6
    reference_velocity: float = 1.0e-6
    direct_effect_vw: float = 0.008
    direct_effect_increment: float = 0.008
    state_effect: float = 0.012
    characteristic_slip: float = 0.02
    vw_half_length: float = 15_000.0
    vw_half_width: float = 7_500.0
    transition_width: float = 3_000.0
    hypocenter_x: float = 0.0
    hypocenter_y: float = 7_500.0
    nucleation_amplitude: float = 25.0e6
    nucleation_radius: float = 3_000.0
    nucleation_time: float = 1.0
    rupture_threshold: float = 1.0e-3
    uguca_dump_interval: float | None = None
    uguca_dump_x_min: float | None = None
    uguca_dump_x_max: float | None = None
    uguca_dump_y_min: float | None = None
    uguca_dump_y_max: float | None = None


@dataclass(frozen=True)
class StructuredBlock:
    mesh: Mesh
    operator: Operator
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    interface_nodes: jax.Array
    mass: jax.Array
    damping: jax.Array


def preset_config(name: str) -> TPV101Config:
    if name == "smoke":
        return replace(
            TPV101Config(),
            mesh_size=4_000.0,
            z_extent=4_000.0,
            x_min=-20_000.0,
            x_max=20_000.0,
            y_min=-4_000.0,
            y_max=20_000.0,
            duration=0.1,
            output_dt=0.01,
            operator_batch_size=512,
        )
    if name == "coarse":
        return TPV101Config()
    if name == "hpc-500m":
        return replace(
            TPV101Config(),
            mesh_size=500.0,
            z_extent=10_000.0,
            output_dt=0.005,
            operator_batch_size=4096,
        )
    if name == "hpc-200m":
        return replace(
            TPV101Config(),
            mesh_size=200.0,
            z_extent=8_000.0,
            output_dt=0.005,
            operator_batch_size=32768,
            symmetry_reduced=True,
        )
    if name == "hpc-150m":
        # 87 cells per 13 km keeps both in-plane domain lengths exactly divisible.
        spacing = 13_000.0 / 87.0
        return replace(
            TPV101Config(),
            mesh_size=spacing,
            z_extent=53.0 * spacing,
            output_dt=0.005,
            operator_batch_size=2048,
            symmetry_reduced=True,
        )
    if name == "hpc-160m":
        # Slightly coarser fallback that reduces peak AD memory on a 24 GB host.
        spacing = 13_000.0 / 82.0
        return replace(
            TPV101Config(),
            mesh_size=spacing,
            z_extent=50.0 * spacing,
            output_dt=0.005,
            operator_batch_size=4096,
            symmetry_reduced=True,
        )
    if name == "hpc-100m":
        return replace(
            TPV101Config(),
            mesh_size=100.0,
            z_extent=8_000.0,
            output_dt=0.005,
            operator_batch_size=4096,
            symmetry_reduced=True,
        )
    raise ValueError(f"Unknown preset: {name}")


def _axis(lower: float, upper: float, spacing: float) -> np.ndarray:
    cells_float = (upper - lower) / spacing
    cells = int(round(cells_float))
    if cells <= 0 or not math.isclose(cells_float, cells, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Axis [{lower}, {upper}] is not divisible by mesh size {spacing}."
        )
    return np.linspace(lower, upper, cells + 1, dtype=np.float64)


def _graded_segment_widths(
    length: float,
    spacing: float,
    max_spacing: float,
    max_growth_ratio: float,
) -> np.ndarray:
    """Return an exact segment whose cells grow smoothly away from a fine region."""
    if math.isclose(length, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return np.empty(0, dtype=np.float64)
    if length < spacing:
        raise ValueError("A graded outer segment cannot be shorter than mesh_size.")
    if max_spacing < spacing:
        raise ValueError("max_mesh_size must be at least mesh_size.")
    if max_growth_ratio < 1.0:
        raise ValueError("mesh_growth_ratio must be at least 1.0.")

    minimum_cells = max(1, int(math.ceil(length / max_spacing)))
    maximum_cells = int(math.floor(length / spacing + 1.0e-12))

    def widths(cell_count: int, growth: float) -> np.ndarray:
        indices = np.arange(cell_count, dtype=np.float64)
        return np.minimum(spacing * np.power(growth, indices), max_spacing)

    for cell_count in range(minimum_cells, maximum_cells + 1):
        upper_widths = widths(cell_count, max_growth_ratio)
        if float(np.sum(upper_widths)) + 1.0e-9 < length:
            continue

        low, high = 1.0, max_growth_ratio
        for _ in range(80):
            middle = 0.5 * (low + high)
            if float(np.sum(widths(cell_count, middle))) < length:
                low = middle
            else:
                high = middle
        result = widths(cell_count, 0.5 * (low + high))
        result[-1] += length - float(np.sum(result))
        if np.min(result) < spacing * (1.0 - 1.0e-10):
            raise RuntimeError("Graded mesh construction produced a cell below mesh_size.")
        return result

    raise ValueError(
        "Unable to grade the requested segment with the configured spacing and growth."
    )


def _graded_axis(
    lower: float,
    upper: float,
    spacing: float,
    fine_lower: float,
    fine_upper: float,
    max_spacing: float,
    max_growth_ratio: float,
) -> np.ndarray:
    if not lower <= fine_lower < fine_upper <= upper:
        raise ValueError(
            f"Fine interval [{fine_lower}, {fine_upper}] must lie inside "
            f"axis [{lower}, {upper}]."
        )
    fine = _axis(fine_lower, fine_upper, spacing)
    lower_widths = _graded_segment_widths(
        fine_lower - lower, spacing, max_spacing, max_growth_ratio
    )
    upper_widths = _graded_segment_widths(
        upper - fine_upper, spacing, max_spacing, max_growth_ratio
    )
    lower_nodes = np.concatenate(
        ([fine_lower], fine_lower - np.cumsum(lower_widths))
    )[::-1]
    upper_nodes = np.concatenate(
        ([fine_upper], fine_upper + np.cumsum(upper_widths))
    )
    axis = np.concatenate((lower_nodes[:-1], fine, upper_nodes[1:]))
    axis[0], axis[-1] = lower, upper
    return axis


def _block_axes(
    config: TPV101Config, *, plus_side: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not config.graded_mesh:
        x = _axis(config.x_min, config.x_max, config.mesh_size)
        y = _axis(config.y_min, config.y_max, config.mesh_size)
        z = _axis(0.0, config.z_extent, config.mesh_size)
    else:
        required = {
            "fine_x_min": config.fine_x_min,
            "fine_x_max": config.fine_x_max,
            "fine_y_min": config.fine_y_min,
            "fine_y_max": config.fine_y_max,
            "fine_z_extent": config.fine_z_extent,
            "max_mesh_size": config.max_mesh_size,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Graded mesh requires: " + ", ".join(sorted(missing))
            )
        x = _graded_axis(
            config.x_min,
            config.x_max,
            config.mesh_size,
            float(config.fine_x_min),
            float(config.fine_x_max),
            float(config.max_mesh_size),
            config.mesh_growth_ratio,
        )
        y = _graded_axis(
            config.y_min,
            config.y_max,
            config.mesh_size,
            float(config.fine_y_min),
            float(config.fine_y_max),
            float(config.max_mesh_size),
            config.mesh_growth_ratio,
        )
        z = _graded_axis(
            0.0,
            config.z_extent,
            config.mesh_size,
            0.0,
            float(config.fine_z_extent),
            float(config.max_mesh_size),
            config.mesh_growth_ratio,
        )
    if not plus_side:
        z = -z[::-1]
    return x, y, z


def _structured_hex_mesh(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    ix, iy, iz = np.meshgrid(
        np.arange(nx, dtype=np.int32),
        np.arange(ny, dtype=np.int32),
        np.arange(nz, dtype=np.int32),
        indexing="ij",
    )
    n000 = ((ix * (ny + 1) + iy) * (nz + 1) + iz).ravel()
    stride_x = (ny + 1) * (nz + 1)
    stride_y = nz + 1
    elements = np.stack(
        [
            n000,
            n000 + stride_x,
            n000 + stride_x + stride_y,
            n000 + stride_y,
            n000 + 1,
            n000 + stride_x + 1,
            n000 + stride_x + stride_y + 1,
            n000 + stride_y + 1,
        ],
        axis=1,
    ).astype(np.int32)
    return coords, elements


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    widths = np.diff(axis)
    if np.any(widths <= 0.0):
        raise ValueError("Mesh axes must be strictly increasing.")
    weights = np.empty(axis.shape, dtype=np.float64)
    weights[0] = 0.5 * widths[0]
    weights[-1] = 0.5 * widths[-1]
    weights[1:-1] = 0.5 * (widths[:-1] + widths[1:])
    return weights


def _build_damping(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: TPV101Config,
    *,
    plus_side: bool,
) -> np.ndarray:
    """Build axis-aligned Lysmer dashpot coefficients [N s/m]."""
    shape = (len(x), len(y), len(z), 3)
    damping = np.zeros(shape, dtype=np.float64)
    wx = _trapezoid_weights(x)
    wy = _trapezoid_weights(y)
    wz = _trapezoid_weights(z)
    rho_cp = config.density * config.pressure_wave_speed
    rho_cs = config.density * config.shear_wave_speed

    yz_area = wy[:, None] * wz[None, :]
    for index in (0, -1):
        damping[index, :, :, 0] += rho_cp * yz_area
        damping[index, :, :, 1] += rho_cs * yz_area
        damping[index, :, :, 2] += rho_cs * yz_area

    xz_area = wx[:, None] * wz[None, :]
    y_boundary_indices = (-1,) if getattr(config, "free_surface_y_min", False) else (0, -1)
    for index in y_boundary_indices:
        damping[:, index, :, 0] += rho_cs * xz_area
        damping[:, index, :, 1] += rho_cp * xz_area
        damping[:, index, :, 2] += rho_cs * xz_area

    xy_area = wx[:, None] * wy[None, :]
    outer_z_index = -1 if plus_side else 0
    damping[:, :, outer_z_index, 0] += rho_cs * xy_area
    damping[:, :, outer_z_index, 1] += rho_cs * xy_area
    damping[:, :, outer_z_index, 2] += rho_cp * xy_area
    return damping.reshape(-1, 3)


def build_block(config: TPV101Config, *, plus_side: bool) -> StructuredBlock:
    x, y, z = _block_axes(config, plus_side=plus_side)
    coords, elements = _structured_hex_mesh(x, y, z)
    mesh = Mesh(jnp.asarray(coords), jnp.asarray(elements))
    operator = Operator(
        mesh,
        Hexahedron8(),
        batch_size=min(config.operator_batch_size, elements.shape[0]),
        cache_weights=True,
    )

    wx = _trapezoid_weights(x)
    wy = _trapezoid_weights(y)
    wz = _trapezoid_weights(z)
    nodal_volume = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    mass = config.density * nodal_volume.reshape(-1)

    nx_nodes, ny_nodes, nz_nodes = len(x), len(y), len(z)
    ix, iy = np.meshgrid(
        np.arange(nx_nodes), np.arange(ny_nodes), indexing="ij"
    )
    iz = 0 if plus_side else nz_nodes - 1
    interface_nodes = ((ix * ny_nodes + iy) * nz_nodes + iz).ravel()
    damping = _build_damping(x, y, z, config, plus_side=plus_side)
    return StructuredBlock(
        mesh=mesh,
        operator=operator,
        x=x,
        y=y,
        z=z,
        interface_nodes=jnp.asarray(interface_nodes, dtype=jnp.int32),
        mass=jnp.asarray(mass),
        damping=jnp.asarray(damping),
    )


def smooth_boxcar(coordinate: jax.Array, half_width: float, transition: float) -> jax.Array:
    absolute = jnp.abs(coordinate)
    transition_value = 0.5 * (
        1.0
        + jnp.tanh(
            transition / (absolute - half_width - transition)
            + transition / (absolute - half_width)
        )
    )
    return jnp.where(
        absolute <= half_width,
        1.0,
        jnp.where(absolute < half_width + transition, transition_value, 0.0),
    )


def direct_effect_profile(
    x: jax.Array, y: jax.Array, config: TPV101Config
) -> jax.Array:
    bx = smooth_boxcar(x - config.hypocenter_x, config.vw_half_length, config.transition_width)
    by = smooth_boxcar(y - config.hypocenter_y, config.vw_half_width, config.transition_width)
    return config.direct_effect_vw + config.direct_effect_increment * (1.0 - bx * by)


def nucleation_perturbation(
    x: jax.Array, y: jax.Array, time: jax.Array, config: TPV101Config
) -> jax.Array:
    radius = jnp.sqrt(
        (x - config.hypocenter_x) ** 2 + (y - config.hypocenter_y) ** 2
    )
    spatial = jnp.where(
        radius < config.nucleation_radius,
        jnp.exp(radius**2 / (radius**2 - config.nucleation_radius**2)),
        0.0,
    )
    safe_time = jnp.maximum(time, jnp.finfo(time.dtype).tiny)
    temporal_ramp = jnp.exp(
        (time - config.nucleation_time) ** 2
        / (safe_time * (time - 2.0 * config.nucleation_time))
    )
    temporal = jnp.where(
        time <= 0.0,
        0.0,
        jnp.where(time < config.nucleation_time, temporal_ramp, 1.0),
    )
    return config.nucleation_amplitude * spatial * temporal


def _station_interpolation(
    x: np.ndarray, y: np.ndarray
) -> tuple[jax.Array, jax.Array]:
    indices = []
    weights = []
    ny = len(y)
    for _name, station_x, station_y in STATIONS:
        ix = int(np.clip(np.searchsorted(x, station_x) - 1, 0, len(x) - 2))
        iy = int(np.clip(np.searchsorted(y, station_y) - 1, 0, len(y) - 2))
        tx = (station_x - x[ix]) / (x[ix + 1] - x[ix])
        ty = (station_y - y[iy]) / (y[iy + 1] - y[iy])
        indices.append(
            [ix * ny + iy, (ix + 1) * ny + iy, ix * ny + iy + 1, (ix + 1) * ny + iy + 1]
        )
        weights.append(
            [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty]
        )
    return jnp.asarray(indices, dtype=jnp.int32), jnp.asarray(weights)


def _surface_station_interpolation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Interpolate TPV102 surface stations from the positive-z symmetry block."""
    if not np.isclose(y[0], 0.0):
        raise ValueError("TPV102 surface interpolation requires y=0 at the first node.")
    indices = []
    weights = []
    parity = []
    ny = len(y)
    nz = len(z)
    for _name, station_z, station_x in SURFACE_STATIONS:
        sample_z = abs(station_z)
        if not (x[0] <= station_x <= x[-1] and z[0] <= sample_z <= z[-1]):
            raise ValueError(
                f"Surface station ({station_x:g}, {station_z:g}) lies outside the mesh."
            )
        ix = int(np.clip(np.searchsorted(x, station_x) - 1, 0, len(x) - 2))
        iz = int(np.clip(np.searchsorted(z, sample_z) - 1, 0, len(z) - 2))
        tx = (station_x - x[ix]) / (x[ix + 1] - x[ix])
        tz = (sample_z - z[iz]) / (z[iz + 1] - z[iz])

        def node(node_x: int, node_z: int) -> int:
            return (node_x * ny) * nz + node_z

        indices.append(
            [node(ix, iz), node(ix + 1, iz), node(ix, iz + 1), node(ix + 1, iz + 1)]
        )
        weights.append(
            [(1.0 - tx) * (1.0 - tz), tx * (1.0 - tz), (1.0 - tx) * tz, tx * tz]
        )
        tangential_sign = 1.0 if station_z > 0.0 else -1.0
        parity.append([tangential_sign, tangential_sign, 1.0])
    return (
        jnp.asarray(indices, dtype=jnp.int32),
        jnp.asarray(weights),
        jnp.asarray(parity),
    )


def estimate_problem_size(config: TPV101Config) -> dict[str, int | float | bool]:
    x, y, z = _block_axes(config, plus_side=True)
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    nodes_per_half = (nx + 1) * (ny + 1) * (nz + 1)
    domain_count = 1 if config.symmetry_reduced else 2
    elements_total = domain_count * nx * ny * nz
    return {
        "nx": nx,
        "ny": ny,
        "nz_per_half": nz,
        "nodes_total": domain_count * nodes_per_half,
        "dofs_total": 3 * domain_count * nodes_per_half,
        "elements_total": elements_total,
        "fault_nodes": (nx + 1) * (ny + 1),
        "symmetry_reduced": config.symmetry_reduced,
        "graded_mesh": config.graded_mesh,
        "min_cell_size": min(np.min(np.diff(axis)) for axis in (x, y, z)),
        "max_cell_size": max(np.max(np.diff(axis)) for axis in (x, y, z)),
    }


def _write_checkpoint(
    path: Path,
    config: TPV101Config,
    carry: tuple[jax.Array | None, ...],
    station_history: list[np.ndarray],
    surface_history: list[np.ndarray],
    output_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "output_index": np.asarray(output_index, dtype=np.int64),
        "station_history": np.stack(station_history, axis=0),
        "surface_history": np.stack(surface_history, axis=0),
        "u_plus": np.asarray(carry[0]),
        "v_half_plus": np.asarray(carry[2]),
        "state": np.asarray(carry[4]),
        "arrival": np.asarray(carry[5]),
        "previous_speed": np.asarray(carry[6]),
        "time": np.asarray(carry[7]),
    }
    if not config.symmetry_reduced:
        arrays["u_minus"] = np.asarray(carry[1])
        arrays["v_half_minus"] = np.asarray(carry[3])

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path, config: TPV101Config
) -> tuple[tuple[jax.Array | None, ...], list[np.ndarray], list[np.ndarray], int]:
    with np.load(path, allow_pickle=False) as checkpoint:
        stored_config = json.loads(str(checkpoint["config_json"].item()))
        if stored_config != asdict(config):
            raise ValueError(f"Checkpoint configuration does not match: {path}")
        output_index = int(checkpoint["output_index"])
        history_array = np.asarray(checkpoint["station_history"])
        if history_array.shape[0] != output_index + 1:
            raise ValueError(f"Checkpoint station history is incomplete: {path}")
        surface_history_array = (
            np.asarray(checkpoint["surface_history"])
            if "surface_history" in checkpoint
            else np.empty((output_index + 1, 0, len(SURFACE_TIME_SERIES_FIELDS)))
        )
        carry = (
            jnp.asarray(checkpoint["u_plus"]),
            None
            if config.symmetry_reduced
            else jnp.asarray(checkpoint["u_minus"]),
            jnp.asarray(checkpoint["v_half_plus"]),
            None
            if config.symmetry_reduced
            else jnp.asarray(checkpoint["v_half_minus"]),
            jnp.asarray(checkpoint["state"]),
            jnp.asarray(checkpoint["arrival"]),
            jnp.asarray(checkpoint["previous_speed"]),
            jnp.asarray(checkpoint["time"]),
        )
    return carry, list(history_array), list(surface_history_array), output_index


def _report_device_memory() -> None:
    """Print the JAX allocator's view of device memory, if it exposes one."""
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # pragma: no cover - platform dependent
        return
    gibibyte = 1024.0**3
    reported = [
        f"{key}={stats[key] / gibibyte:.2f} GiB"
        for key in (
            "bytes_in_use",
            "peak_bytes_in_use",
            "bytes_reserved",
            "peak_pool_bytes",
            "bytes_limit",
        )
        if key in stats
    ]
    if reported:
        print("device memory: " + ", ".join(reported), flush=True)


def _material(config: TPV101Config) -> tuple[float, float]:
    mu = config.density * config.shear_wave_speed**2
    lmbda = config.density * config.pressure_wave_speed**2 - 2.0 * mu
    return lmbda, mu


def run_simulation(
    config: TPV101Config,
    *,
    checkpoint_path: Path | None = None,
    checkpoint_interval_s: float | None = None,
    resume: bool = False,
    uguca_dump_dir: Path | None = None,
    uguca_dump_name: str | None = None,
) -> dict[str, Any]:
    plus = build_block(config, plus_side=True)
    minus = None if config.symmetry_reduced else build_block(config, plus_side=False)
    if minus is not None and not (
        np.array_equal(plus.x, minus.x) and np.array_equal(plus.y, minus.y)
    ):
        raise RuntimeError("The two fault traces do not match.")

    lmbda, mu = _material(config)
    fault_x, fault_y = np.meshgrid(plus.x, plus.y, indexing="ij")
    fault_x_flat = jnp.asarray(fault_x.ravel())
    fault_y_flat = jnp.asarray(fault_y.ravel())
    interface_weights = jnp.asarray(
        (_trapezoid_weights(plus.x)[:, None] * _trapezoid_weights(plus.y)[None, :]).ravel()
    )
    direct_effect = direct_effect_profile(fault_x_flat, fault_y_flat, config)
    initial_state = regularized_rate_state_initial_state(
        jnp.full(direct_effect.shape, config.initial_slip_rate),
        jnp.full(direct_effect.shape, config.initial_shear_stress),
        jnp.full(direct_effect.shape, config.normal_stress),
        reference_friction=config.reference_friction,
        direct_effect=direct_effect,
        state_effect=config.state_effect,
        reference_velocity=config.reference_velocity,
        characteristic_slip=config.characteristic_slip,
    )
    station_indices, station_weights = _station_interpolation(plus.x, plus.y)
    has_free_surface = getattr(config, "free_surface_y_min", False)
    if has_free_surface:
        surface_indices, surface_weights, surface_parity = _surface_station_interpolation(
            plus.x, plus.y, plus.z
        )
    else:
        surface_indices = jnp.empty((0, 4), dtype=jnp.int32)
        surface_weights = jnp.empty((0, 4), dtype=jnp.float64)
        surface_parity = jnp.empty((0, 3), dtype=jnp.float64)
    vw_mask = (
        (fault_x_flat > -config.vw_half_length)
        & (fault_x_flat < config.vw_half_length)
        & (fault_y_flat > 0.0)
        & (fault_y_flat < 2.0 * config.vw_half_width)
    )
    normal_penalty = config.normal_penalty_factor * mu / config.mesh_size

    min_interface_mass_per_area = float(
        jnp.min(plus.mass[plus.interface_nodes] / interface_weights)
    )
    if minus is not None:
        min_interface_mass_per_area = min(
            min_interface_mass_per_area,
            float(jnp.min(minus.mass[minus.interface_nodes] / interface_weights)),
        )
    dt_wave = config.cfl * config.mesh_size / config.pressure_wave_speed
    dt_contact = 0.25 * math.sqrt(
        min_interface_mass_per_area / normal_penalty
    )
    target_dt = min(dt_wave, dt_contact, config.output_dt)
    substeps_per_output = max(1, int(math.ceil(config.output_dt / target_dt)))
    dt = config.output_dt / substeps_per_output
    output_steps = int(round(config.duration / config.output_dt))
    if not math.isclose(
        output_steps * config.output_dt,
        config.duration,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("duration must be exactly divisible by output_dt.")
    checkpoint_outputs = None
    if checkpoint_interval_s is not None:
        checkpoint_outputs_float = checkpoint_interval_s / config.output_dt
        checkpoint_outputs = int(round(checkpoint_outputs_float))
        if checkpoint_path is None or checkpoint_outputs <= 0 or not math.isclose(
            checkpoint_outputs_float,
            checkpoint_outputs,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "checkpoint_interval_s must be positive, divisible by output_dt, "
                "and accompanied by checkpoint_path."
            )
    if resume and (checkpoint_path is None or not checkpoint_path.exists()):
        raise FileNotFoundError("--resume requires an existing checkpoint.")

    def block_energy(operator: Operator, displacement: jax.Array) -> jax.Array:
        grad = operator.grad(displacement)
        strain = 0.5 * (grad + jnp.swapaxes(grad, -1, -2))
        density = mu * jnp.einsum("...ij,...ij->...", strain, strain)
        density += 0.5 * lmbda * jnp.trace(strain, axis1=-2, axis2=-1) ** 2
        return operator.integrate(density)

    if config.symmetry_reduced:
        elastic_energy_and_force = jax.jit(
            jax.value_and_grad(lambda displacement: block_energy(plus.operator, displacement))
        )
    else:
        if minus is None:
            raise RuntimeError("Full-domain solve requires the minus block.")

        def elastic_energy(u_plus: jax.Array, u_minus: jax.Array) -> jax.Array:
            return block_energy(plus.operator, u_plus) + block_energy(
                minus.operator, u_minus
            )

        elastic_energy_and_force = jax.jit(
            jax.value_and_grad(elastic_energy, argnums=(0, 1))
        )

    def fault_fields(
        u_plus: jax.Array,
        u_minus: jax.Array | None,
        velocity_plus: jax.Array,
        velocity_minus: jax.Array | None,
        state: jax.Array,
        time: jax.Array,
    ) -> tuple[jax.Array, ...]:
        if config.symmetry_reduced:
            tangential_displacement = 2.0 * u_plus[plus.interface_nodes, :2]
            tangential_velocity = 2.0 * velocity_plus[plus.interface_nodes, :2]
            relative_displacement = jnp.column_stack(
                [tangential_displacement, jnp.zeros(tangential_displacement.shape[0])]
            )
            relative_velocity = jnp.column_stack(
                [tangential_velocity, jnp.zeros(tangential_velocity.shape[0])]
            )
        else:
            if minus is None or u_minus is None or velocity_minus is None:
                raise RuntimeError("Full-domain fault fields require both blocks.")
            relative_displacement = (
                u_plus[plus.interface_nodes] - u_minus[minus.interface_nodes]
            )
            relative_velocity = (
                velocity_plus[plus.interface_nodes]
                - velocity_minus[minus.interface_nodes]
            )
        slip_velocity = relative_velocity[:, :2]
        speed = jnp.linalg.norm(slip_velocity, axis=1)
        gap = relative_displacement[:, 2]
        normal_stress = jnp.maximum(
            config.normal_stress - normal_penalty * gap,
            1.0,
        )
        strength = regularized_rate_state_strength(
            speed,
            normal_stress,
            state,
            reference_friction=config.reference_friction,
            direct_effect=direct_effect,
            state_effect=config.state_effect,
            reference_velocity=config.reference_velocity,
            characteristic_slip=config.characteristic_slip,
        )
        default_direction = jnp.zeros_like(slip_velocity).at[:, 0].set(1.0)
        direction = jnp.where(
            (speed > 0.0)[:, None],
            slip_velocity / jnp.maximum(speed[:, None], jnp.finfo(speed.dtype).tiny),
            default_direction,
        )
        friction_traction = strength[:, None] * direction
        perturbation = nucleation_perturbation(
            fault_x_flat, fault_y_flat, time, config
        )
        driving_traction = jnp.zeros_like(friction_traction).at[:, 0].set(
            config.initial_shear_stress + perturbation
        )
        return (
            relative_displacement,
            relative_velocity,
            speed,
            normal_stress,
            friction_traction,
            driving_traction,
        )

    def free_acceleration(
        u_plus: jax.Array,
        u_minus: jax.Array | None,
        velocity_plus: jax.Array,
        velocity_minus: jax.Array | None,
        state: jax.Array,
        time: jax.Array,
    ) -> tuple[jax.Array, jax.Array | None, tuple[jax.Array, ...]]:
        if config.symmetry_reduced:
            _energy, internal_plus = elastic_energy_and_force(u_plus)
            internal_minus = None
        else:
            if u_minus is None:
                raise RuntimeError("Full-domain acceleration requires both blocks.")
            _energy, (internal_plus, internal_minus) = elastic_energy_and_force(
                u_plus, u_minus
            )
        fields = fault_fields(
            u_plus, u_minus, velocity_plus, velocity_minus, state, time
        )
        relative_displacement, _relative_velocity, _speed, _normal, _friction, drive = fields
        force_plus = -internal_plus - plus.damping * velocity_plus
        tangential_increment = interface_weights[:, None] * drive
        normal_increment = (
            -normal_penalty * relative_displacement[:, 2] * interface_weights
        )
        force_plus = force_plus.at[plus.interface_nodes, :2].add(
            tangential_increment
        )
        force_plus = force_plus.at[plus.interface_nodes, 2].add(normal_increment)
        if config.symmetry_reduced:
            return force_plus / plus.mass[:, None], None, fields
        if minus is None or velocity_minus is None or internal_minus is None:
            raise RuntimeError("Full-domain acceleration requires both blocks.")
        force_minus = -internal_minus - minus.damping * velocity_minus
        force_minus = force_minus.at[minus.interface_nodes, :2].add(
            -tangential_increment
        )
        force_minus = force_minus.at[minus.interface_nodes, 2].add(-normal_increment)
        return force_plus / plus.mass[:, None], force_minus / minus.mass[:, None], fields

    free_acceleration_jit = jax.jit(free_acceleration)

    inverse_mass_plus = 1.0 / plus.mass[plus.interface_nodes]
    inverse_mass_minus = (
        inverse_mass_plus
        if config.symmetry_reduced
        else 1.0 / minus.mass[minus.interface_nodes]
    )
    relative_impulse_factor = dt * interface_weights * (
        inverse_mass_plus + inverse_mass_minus
    )

    def apply_implicit_friction(
        velocity_plus_free: jax.Array,
        velocity_minus_free: jax.Array | None,
        u_plus: jax.Array,
        u_minus: jax.Array | None,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array | None]:
        if config.symmetry_reduced:
            free_relative = 2.0 * velocity_plus_free[plus.interface_nodes, :2]
        else:
            if minus is None or velocity_minus_free is None:
                raise RuntimeError("Full-domain friction requires both blocks.")
            free_relative = (
                velocity_plus_free[plus.interface_nodes, :2]
                - velocity_minus_free[minus.interface_nodes, :2]
            )
        free_speed = jnp.linalg.norm(free_relative, axis=1)
        direction = jnp.where(
            (free_speed > 0.0)[:, None],
            free_relative
            / jnp.maximum(free_speed[:, None], jnp.finfo(free_speed.dtype).tiny),
            jnp.array([1.0, 0.0]),
        )
        if config.symmetry_reduced:
            gap = jnp.zeros(u_plus[plus.interface_nodes, 2].shape)
        else:
            if minus is None or u_minus is None:
                raise RuntimeError("Full-domain friction requires both blocks.")
            gap = (
                u_plus[plus.interface_nodes, 2]
                - u_minus[minus.interface_nodes, 2]
            )
        normal_stress = jnp.maximum(
            config.normal_stress - normal_penalty * gap,
            1.0,
        )

        def residual(speed: jax.Array) -> jax.Array:
            strength = regularized_rate_state_strength(
                speed,
                normal_stress,
                state,
                reference_friction=config.reference_friction,
                direct_effect=direct_effect,
                state_effect=config.state_effect,
                reference_velocity=config.reference_velocity,
                characteristic_slip=config.characteristic_slip,
            )
            return speed + relative_impulse_factor * strength - free_speed

        lower = jnp.zeros_like(free_speed)
        upper = free_speed

        def bisect(_iteration: int, bounds: tuple[jax.Array, jax.Array]):
            low, high = bounds
            midpoint = 0.5 * (low + high)
            move_low = residual(midpoint) < 0.0
            return jnp.where(move_low, midpoint, low), jnp.where(
                move_low, high, midpoint
            )

        lower, upper = jax.lax.fori_loop(0, 64, bisect, (lower, upper))
        corrected_speed = 0.5 * (lower + upper)
        corrected_relative = corrected_speed[:, None] * direction
        relative_correction = free_relative - corrected_relative
        inverse_mass_sum = inverse_mass_plus + inverse_mass_minus
        plus_share = inverse_mass_plus / inverse_mass_sum
        minus_share = inverse_mass_minus / inverse_mass_sum
        velocity_plus = velocity_plus_free.at[plus.interface_nodes, :2].add(
            -plus_share[:, None] * relative_correction
        )
        if config.symmetry_reduced:
            return velocity_plus, None
        if minus is None or velocity_minus_free is None:
            raise RuntimeError("Full-domain friction requires both blocks.")
        velocity_minus = velocity_minus_free.at[minus.interface_nodes, :2].add(
            minus_share[:, None] * relative_correction
        )
        return velocity_plus, velocity_minus

    def interpolate_station(field: jax.Array) -> jax.Array:
        if field.ndim == 1:
            return jnp.einsum("sk,sk->s", station_weights, field[station_indices])
        return jnp.einsum("sk,skd->sd", station_weights, field[station_indices])

    def dump_slice(fields: tuple[jax.Array, ...], state: jax.Array) -> jax.Array:
        """Gather the UGUCA dump window out of the fields a step computed."""
        if dump_indices is None:
            return jnp.zeros((0, 4))
        (
            relative_displacement,
            relative_velocity,
            _speed,
            _normal,
            friction,
            _drive,
        ) = fields
        return jnp.column_stack(
            [
                friction[dump_indices, 0],
                0.5 * relative_displacement[dump_indices, 0],
                0.5 * relative_velocity[dump_indices, 0],
                state[dump_indices],
            ]
        )

    def station_rows(
        fields: tuple[jax.Array, ...], state: jax.Array, time: jax.Array
    ) -> jax.Array:
        relative_displacement, relative_velocity, _speed, normal, friction, _drive = fields
        slip = interpolate_station(relative_displacement[:, :2])
        slip_rate = interpolate_station(relative_velocity[:, :2])
        traction = interpolate_station(friction)
        station_normal = interpolate_station(normal)
        station_state = interpolate_station(state)
        return jnp.column_stack(
            [
                jnp.full((len(STATIONS),), time),
                slip[:, 0],
                slip_rate[:, 0],
                traction[:, 0] / 1.0e6,
                slip[:, 1],
                slip_rate[:, 1],
                traction[:, 1] / 1.0e6,
                station_normal / 1.0e6,
                jnp.log10(station_state),
            ]
        )

    def surface_rows(
        displacement_plus: jax.Array,
        velocity_plus: jax.Array,
        time: jax.Array,
    ) -> jax.Array:
        displacement = jnp.einsum(
            "sk,skd->sd", surface_weights, displacement_plus[surface_indices]
        )
        velocity = jnp.einsum(
            "sk,skd->sd", surface_weights, velocity_plus[surface_indices]
        )
        displacement = displacement * surface_parity
        velocity = velocity * surface_parity
        return jnp.column_stack(
            [
                jnp.full((len(SURFACE_STATIONS) if has_free_surface else 0,), time),
                displacement[:, 0],
                velocity[:, 0],
                displacement[:, 1],
                velocity[:, 1],
                displacement[:, 2],
                velocity[:, 2],
            ]
        )

    n_plus = plus.mesh.coords.shape[0]
    u_plus0 = jnp.zeros((n_plus, 3), dtype=jnp.float64)
    u_minus0 = (
        None
        if config.symmetry_reduced
        else jnp.zeros((minus.mesh.coords.shape[0], 3), dtype=jnp.float64)
    )
    velocity_plus0 = jnp.zeros_like(u_plus0).at[:, 0].set(
        0.5 * config.initial_slip_rate
    )
    velocity_minus0 = (
        None
        if config.symmetry_reduced
        else jnp.zeros_like(u_minus0).at[:, 0].set(
            -0.5 * config.initial_slip_rate
        )
    )
    time0 = jnp.asarray(0.0)
    initial_fields = fault_fields(
        u_plus0, u_minus0, velocity_plus0, velocity_minus0, initial_state, time0
    )
    v_half_plus0 = velocity_plus0
    v_half_minus0 = velocity_minus0
    initial_speed = initial_fields[2]
    arrival0 = jnp.full(initial_speed.shape, 1.0e9)

    def step(carry: tuple[jax.Array, ...], _unused: None):
        (
            u_plus,
            u_minus,
            v_half_plus,
            v_half_minus,
            state,
            arrival,
            previous_speed,
            time,
        ) = carry
        u_plus_new = u_plus + dt * v_half_plus
        if config.symmetry_reduced:
            u_minus_new = None
            relative_velocity_half = 2.0 * v_half_plus[plus.interface_nodes, :2]
        else:
            u_minus_new = u_minus + dt * v_half_minus
            relative_velocity_half = (
                v_half_plus[plus.interface_nodes, :2]
                - v_half_minus[minus.interface_nodes, :2]
            )
        speed_half = jnp.linalg.norm(relative_velocity_half, axis=1)
        state_new = update_ageing_state(
            state, speed_half, dt, config.characteristic_slip
        )
        time_new = time + dt
        accel_plus, accel_minus, _fields_half = free_acceleration(
            u_plus_new,
            u_minus_new,
            v_half_plus,
            v_half_minus,
            state_new,
            time_new,
        )
        v_half_plus_free = v_half_plus + dt * accel_plus
        v_half_minus_free = (
            None
            if config.symmetry_reduced
            else v_half_minus + dt * accel_minus
        )
        v_half_plus_new, v_half_minus_new = apply_implicit_friction(
            v_half_plus_free,
            v_half_minus_free,
            u_plus_new,
            u_minus_new,
            state_new,
        )
        velocity_plus = 0.5 * (v_half_plus + v_half_plus_new)
        velocity_minus = (
            None
            if config.symmetry_reduced
            else 0.5 * (v_half_minus + v_half_minus_new)
        )
        fields = fault_fields(
            u_plus_new,
            u_minus_new,
            velocity_plus,
            velocity_minus,
            state_new,
            time_new,
        )
        speed = fields[2]
        crossing = (
            (arrival >= 1.0e9)
            & (previous_speed < config.rupture_threshold)
            & (speed >= config.rupture_threshold)
        )
        fraction = jnp.clip(
            (config.rupture_threshold - previous_speed)
            / jnp.maximum(speed - previous_speed, jnp.finfo(speed.dtype).tiny),
            0.0,
            1.0,
        )
        arrival_new = jnp.where(
            crossing,
            time_new - dt + fraction * dt,
            arrival,
        )
        rows = station_rows(fields, state_new, time_new)
        body_rows = surface_rows(u_plus_new, velocity_plus, time_new)
        dump_rows = dump_slice(fields, state_new)
        return (
            u_plus_new,
            u_minus_new,
            v_half_plus_new,
            v_half_minus_new,
            state_new,
            arrival_new,
            speed,
            time_new,
        ), (rows, body_rows, dump_rows)

    @jax.jit
    def advance_output(carry: tuple[jax.Array, ...]):
        new_carry, (rows, body_rows, dump_rows) = jax.lax.scan(
            step, carry, xs=None, length=substeps_per_output
        )
        return new_carry, (rows[-1], body_rows[-1], dump_rows[-1])

    carry = (
        u_plus0,
        u_minus0,
        v_half_plus0,
        v_half_minus0,
        initial_state,
        arrival0,
        initial_speed,
        time0,
    )
    dumper = None
    dump_every = 0
    dump_indices = None
    if uguca_dump_dir is not None and config.uguca_dump_interval:
        if config.graded_mesh:
            window_default = (
                config.fine_x_min,
                config.fine_x_max,
                config.fine_y_min,
                config.fine_y_max,
            )
        else:
            window_default = (config.x_min, config.x_max, config.y_min, config.y_max)
        window = tuple(
            override if override is not None else fallback
            for override, fallback in zip(
                (
                    config.uguca_dump_x_min,
                    config.uguca_dump_x_max,
                    config.uguca_dump_y_min,
                    config.uguca_dump_y_max,
                ),
                window_default,
            )
        )
        dump_x, dump_z, dump_flat = window_indices(plus.x, plus.y, *window)
        dump_indices = jnp.asarray(dump_flat)
        dump_every = max(
            1, int(round(config.uguca_dump_interval / config.output_dt))
        )
        dumper = UgucaDumper(
            uguca_dump_dir,
            uguca_dump_name or "interface",
            dump_x,
            dump_z,
            append=resume,
        )
        print(
            f"UGUCA dump: {dumper.base_path} "
            f"({dump_x.size} x {dump_z.size} nodes, "
            f"x=[{dump_x[0]:g}, {dump_x[-1]:g}] m, z=[{dump_z[0]:g}, {dump_z[-1]:g}] m, "
            f"every {dump_every} outputs = {dump_every * config.output_dt:g} s)",
            flush=True,
        )

    def write_uguca_dump(dump_rows: jax.Array, index: int) -> None:
        """Append one UGUCA frame from the columns the step already produced."""
        values = np.asarray(dump_rows)
        dumper.dump(
            index * substeps_per_output,
            index * config.output_dt,
            {
                "cohesion_0": values[:, 0],
                "top_disp_0": values[:, 1],
                "top_velo_0": values[:, 2],
                "theta": values[:, 3],
            },
        )

    station_history = [
        np.asarray(station_rows(initial_fields, initial_state, time0))
    ]
    surface_history = [np.asarray(surface_rows(u_plus0, velocity_plus0, time0))]
    start_output_index = 0
    if resume:
        carry, station_history, surface_history, start_output_index = _load_checkpoint(
            checkpoint_path, config
        )
        if start_output_index > output_steps:
            raise ValueError("Checkpoint is beyond the requested duration.")
        print(
            f"Resuming {checkpoint_path} at "
            f"t={start_output_index * config.output_dt:.3f} s",
            flush=True,
        )
    if dumper is not None:
        if start_output_index == 0:
            write_uguca_dump(dump_slice(initial_fields, initial_state), 0)
        else:
            kept = dumper.truncate_to_time(start_output_index * config.output_dt)
            print(f"UGUCA dump resumed with {kept} frames", flush=True)
    print(
        f"{getattr(config, 'problem', 'TPV101')} mesh={config.mesh_size:g} m, dt={dt:.6e} s, "
        f"substeps/output={substeps_per_output}, outputs={output_steps + 1}",
        flush=True,
    )
    loop_started = time.monotonic()
    previous_report = loop_started
    for output_index in range(start_output_index + 1, output_steps + 1):
        carry, (rows, body_rows, dump_rows) = advance_output(carry)
        station_history.append(np.asarray(rows))
        surface_history.append(np.asarray(body_rows))
        if output_index == 1:
            jax.block_until_ready(rows)
            _report_device_memory()
        if (
            output_index == 1
            or output_index % max(1, output_steps // 20) == 0
            or output_index == output_steps
        ):
            now = time.monotonic()
            print(
                f"  {output_index:5d}/{output_steps}: "
                f"t={output_index * config.output_dt:7.3f} s "
                f"wall={now - loop_started:8.1f} s "
                f"since_report={now - previous_report:7.1f} s",
                flush=True,
            )
            previous_report = now
        if dumper is not None and (
            output_index % dump_every == 0 or output_index == output_steps
        ):
            write_uguca_dump(dump_rows, output_index)
        if checkpoint_outputs is not None and (
            output_index % checkpoint_outputs == 0 or output_index == output_steps
        ):
            _write_checkpoint(
                checkpoint_path,
                config,
                carry,
                station_history,
                surface_history,
                output_index,
            )
            print(f"  checkpoint: {checkpoint_path}", flush=True)

    if dumper is not None:
        print(
            f"UGUCA dump closed with {dumper.frames_written} frames: "
            f"{dumper.base_path}",
            flush=True,
        )
        dumper.close()

    station_history_array = np.stack(station_history, axis=0)
    final_arrival = np.asarray(carry[5])
    final_state = np.asarray(carry[4])
    return {
        "config": config,
        "dt": dt,
        "substeps_per_output": substeps_per_output,
        "station_history": station_history_array,
        "surface_history": np.stack(surface_history, axis=0),
        "fault_x": np.asarray(fault_x_flat),
        "fault_y": np.asarray(fault_y_flat),
        "direct_effect": np.asarray(direct_effect),
        "initial_state": np.asarray(initial_state),
        "final_state": final_state,
        "rupture_arrival": final_arrival,
        "vw_mask": np.asarray(vw_mask),
        "problem_size": estimate_problem_size(config),
    }


def _write_station_file(
    path: Path,
    station: tuple[str, float, float],
    rows: np.ndarray,
    config: TPV101Config,
    dt: float,
) -> None:
    name, x, y = station
    problem = getattr(config, "problem", "TPV101")
    header = [
        f"# SCEC {problem} on-fault time series",
        f"# problem={problem}",
        "# author=Tatva validation workflow",
        f"# date={date.today().isoformat()}",
        "# code=Tatva",
        f"# code_version={problem}-validation-1",
        f"# element_size={config.mesh_size:g} m",
        f"# time_step={config.output_dt:.12e}",
        f"# internal_time_step={dt:.12e}",
        f"# num_time_steps={rows.shape[0]}",
        f"# location=on fault, {x / 1000:g} km along strike, {y / 1000:g} km down-dip",
        "# Column #1 = Time (s)",
        "# Column #2 = horizontal slip (m)",
        "# Column #3 = horizontal slip rate (m/s)",
        "# Column #4 = horizontal shear stress (MPa)",
        "# Column #5 = vertical slip (m)",
        "# Column #6 = vertical slip rate (m/s)",
        "# Column #7 = vertical shear stress (MPa)",
        "# Column #8 = normal stress, positive in compression (MPa)",
        "# Column #9 = log10 of state variable (log-seconds)",
        " ".join(TIME_SERIES_FIELDS),
    ]
    with path.open("w", encoding="ascii") as stream:
        stream.write("\n".join(header) + "\n")
        for row in rows:
            stream.write(
                f"{row[0]:20.12E} "
                + " ".join(f"{value:14.6E}" for value in row[1:])
                + "\n"
            )


def _write_surface_station_file(
    path: Path,
    station: tuple[str, float, float],
    rows: np.ndarray,
    config: TPV101Config,
    dt: float,
) -> None:
    name, z, x = station
    side = "far" if z > 0.0 else "near"
    header = [
        "# SCEC TPV102 free-surface time series",
        "# problem=TPV102",
        "# author=Tatva validation workflow",
        f"# date={date.today().isoformat()}",
        "# code=Tatva",
        "# code_version=TPV102-validation-1",
        f"# element_size={config.mesh_size:g} m",
        f"# time_step={config.output_dt:.12e}",
        f"# internal_time_step={dt:.12e}",
        f"# num_time_steps={rows.shape[0]}",
        f"# location={abs(z) / 1000:g} km off fault ({side} side), "
        f"{x / 1000:g} km along strike, 0 km depth",
        "# Column #1 = Time (s)",
        "# Column #2 = horizontal displacement, parallel to fault (m)",
        "# Column #3 = horizontal velocity, parallel to fault (m/s)",
        "# Column #4 = vertical displacement, positive downward (m)",
        "# Column #5 = vertical velocity, positive downward (m/s)",
        "# Column #6 = normal displacement, positive toward far side (m)",
        "# Column #7 = normal velocity, positive toward far side (m/s)",
        " ".join(SURFACE_TIME_SERIES_FIELDS),
    ]
    with path.open("w", encoding="ascii") as stream:
        stream.write("\n".join(header) + "\n")
        for row in rows:
            stream.write(
                f"{row[0]:20.12E} "
                + " ".join(f"{value:14.6E}" for value in row[1:])
                + "\n"
            )


def write_scec_dump(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config: TPV101Config = result["config"]
    problem = getattr(config, "problem", "TPV101")
    problem_lower = problem.lower()
    history = result["station_history"]
    station_paths = {}
    for station_index, station in enumerate(STATIONS):
        path = output_dir / f"{station[0]}.txt"
        _write_station_file(
            path,
            station,
            history[:, station_index, :],
            config,
            result["dt"],
        )
        station_paths[station[0]] = str(path)

    surface_history = result.get("surface_history")
    if surface_history is not None and surface_history.shape[1] == len(SURFACE_STATIONS):
        for station_index, station in enumerate(SURFACE_STATIONS):
            path = output_dir / f"{station[0]}.txt"
            _write_surface_station_file(
                path,
                station,
                surface_history[:, station_index, :],
                config,
                result["dt"],
            )
            station_paths[station[0]] = str(path)

    contour_path = output_dir / f"{problem_lower}_rupture_time.txt"
    mask = result["vw_mask"]
    contour_rows = np.column_stack(
        [
            result["fault_x"][mask],
            result["fault_y"][mask],
            result["rupture_arrival"][mask],
        ]
    )
    with contour_path.open("w", encoding="ascii") as stream:
        stream.write(f"# SCEC {problem} rupture-front arrival times\n")
        stream.write(f"# problem={problem}\n")
        stream.write("# author=Tatva validation workflow\n")
        stream.write(f"# date={date.today().isoformat()}\n")
        stream.write("# code=Tatva\n")
        stream.write(f"# code_version={problem}-validation-1\n")
        stream.write(f"# element_size={config.mesh_size:g} m\n")
        stream.write("# Column #1 = distance along strike j (m)\n")
        stream.write("# Column #2 = distance down-dip k (m)\n")
        stream.write("# Column #3 = first time slip rate exceeds 1 mm/s (s)\n")
        stream.write("j k t\n")
        for x, y, arrival in contour_rows:
            stream.write(f"{x:14.6E} {y:14.6E} {arrival:14.6E}\n")

    npz_path = output_dir / f"{problem_lower}_internal_diagnostics.npz"
    np.savez_compressed(
        npz_path,
        fault_x=result["fault_x"],
        fault_y=result["fault_y"],
        direct_effect=result["direct_effect"],
        initial_state=result["initial_state"],
        final_state=result["final_state"],
        rupture_arrival=result["rupture_arrival"],
        station_history=result["station_history"],
        surface_history=result.get("surface_history"),
    )
    summary_path = output_dir / "summary.json"
    summary = {
        "problem": problem,
        "code": "Tatva",
        "config": asdict(config),
        "internal_dt": result["dt"],
        "substeps_per_output": result["substeps_per_output"],
        "problem_size": result["problem_size"],
        "station_files": station_paths,
        "rupture_contour": str(contour_path),
        "internal_diagnostics": str(npz_path),
        "ruptured_vw_nodes": int(
            np.count_nonzero(result["rupture_arrival"][result["vw_mask"]] < 1.0e9)
        ),
        "vw_nodes": int(np.count_nonzero(result["vw_mask"])),
        "boundary_model": (
            "fault-symmetry reduction, y=0 traction-free surface, and outer Lysmer dashpots"
            if getattr(config, "free_surface_y_min", False)
            else "one-half symmetry reduction with axis-aligned Lysmer dashpots"
            if config.symmetry_reduced
            else "finite 3D domain with axis-aligned Lysmer dashpots"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "contour": str(contour_path),
        "diagnostics": str(npz_path),
        **station_paths,
    }


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
    symmetry_group = parser.add_mutually_exclusive_group()
    symmetry_group.add_argument(
        "--symmetry-reduced", dest="symmetry_reduced", action="store_true"
    )
    symmetry_group.add_argument(
        "--full-domain", dest="symmetry_reduced", action="store_false"
    )
    parser.set_defaults(symmetry_reduced=None)
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
        "symmetry_reduced": args.symmetry_reduced,
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
    output_dir = args.output_dir or Path("output") / f"tpv101_{args.preset}"
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
