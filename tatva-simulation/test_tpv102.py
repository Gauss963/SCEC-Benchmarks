from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tpv101 import SURFACE_STATIONS, SURFACE_TIME_SERIES_FIELDS, build_block, run_simulation, write_scec_dump
from tpv102 import TPV102Config, preset_config
from validate_scec_dump import _read_ascii_table, validate_dump


def test_tpv102_100m_preset_is_half_space_with_surface_station_clearance():
    config = preset_config("hpc-100m")

    assert isinstance(config, TPV102Config)
    assert config.problem == "TPV102"
    assert config.free_surface_y_min
    assert config.y_min == 0.0
    assert config.z_extent == 12_000.0
    assert config.symmetry_reduced
    assert config.z_extent > max(abs(station[1]) for station in SURFACE_STATIONS)


def test_tpv102_expanded_domain_keeps_physical_free_surface():
    config = replace(
        preset_config("hpc-100m"),
        x_min=-30_000.0,
        x_max=30_000.0,
        y_max=32_000.0,
    )

    assert config.y_min == 0.0
    assert config.free_surface_y_min
    assert (config.x_max - config.x_min) / config.mesh_size == 600
    assert (config.y_max - config.y_min) / config.mesh_size == 320


def test_tpv102_free_surface_has_no_y_boundary_dashpot():
    config = preset_config("smoke")
    block = build_block(config, plus_side=True)
    damping = np.asarray(block.damping).reshape(len(block.x), len(block.y), len(block.z), 3)

    assert damping[1:-1, 0, 1:-1] == pytest.approx(0.0)
    assert np.any(damping[1:-1, -1, 1:-1] > 0.0)


def test_tpv102_smoke_writes_fault_surface_and_contour_files(tmp_path):
    config = replace(preset_config("smoke"), duration=0.02)
    result = run_simulation(config)
    paths = write_scec_dump(result, tmp_path)
    report = validate_dump(tmp_path)

    assert report["valid"], report["errors"]
    assert report["problem"] == "TPV102"
    assert len(report["surface_stations"]) == len(SURFACE_STATIONS)
    assert "body060st120dp000" in paths
    assert (tmp_path / "tpv102_rupture_time.txt").exists()
    assert not (tmp_path / "tpv101_rupture_time.txt").exists()

    far_fields, far = _read_ascii_table(tmp_path / "body060st120dp000.txt")
    near_fields, near = _read_ascii_table(tmp_path / "body-060st-120dp000.txt")
    assert far_fields == near_fields == list(SURFACE_TIME_SERIES_FIELDS)
    assert far[0, 2] == pytest.approx(0.5e-12)
    assert near[0, 2] == pytest.approx(-0.5e-12)
