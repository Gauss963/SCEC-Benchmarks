from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from analyze_horizontal_shear_tail import _comparison_metrics
from tatva.friction import (
    regularized_rate_state_initial_state,
    regularized_rate_state_strength,
)
from tpv101 import (
    TPV101Config,
    _block_axes,
    _trapezoid_weights,
    direct_effect_profile,
    estimate_problem_size,
    nucleation_perturbation,
    preset_config,
    run_simulation,
    smooth_boxcar,
    write_scec_dump,
)
from validate_scec_dump import _has_uniform_time_step, validate_dump
from compare_scec_reference import _contour_metrics
from plot_tpv101_validation import _reference_submissions, _summarize_station_metrics


def test_smooth_boxcar_and_direct_effect_match_tpv101_regions():
    coordinates = jnp.asarray([0.0, 15_000.0, 16_500.0, 18_000.0, 19_000.0])
    values = np.asarray(smooth_boxcar(coordinates, 15_000.0, 3_000.0))

    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(0.5)
    assert values[3] == pytest.approx(0.0)
    assert values[4] == pytest.approx(0.0)

    config = TPV101Config()
    direct_effect = np.asarray(
        direct_effect_profile(
            jnp.asarray([0.0, 20_000.0]),
            jnp.asarray([7_500.0, 7_500.0]),
            config,
        )
    )
    assert direct_effect == pytest.approx([0.008, 0.016])


def test_nucleation_perturbation_matches_space_time_limits():
    config = TPV101Config()
    x = jnp.asarray([0.0, config.nucleation_radius, 4_000.0])
    y = jnp.full(x.shape, config.hypocenter_y)

    at_start = np.asarray(nucleation_perturbation(x, y, jnp.asarray(0.0), config))
    at_end = np.asarray(nucleation_perturbation(x, y, jnp.asarray(1.0), config))

    assert at_start == pytest.approx([0.0, 0.0, 0.0])
    assert at_end == pytest.approx([25.0e6, 0.0, 0.0])


def test_hpc_200m_preset_matches_validation_run():
    config = preset_config("hpc-200m")

    assert config.mesh_size == 200.0
    assert config.z_extent == 8_000.0
    assert config.output_dt == 0.005
    assert config.operator_batch_size == 32768
    assert config.symmetry_reduced


def test_fine_hpc_presets_use_valid_symmetric_domains():
    config_150m = preset_config("hpc-150m")
    config_160m = preset_config("hpc-160m")
    config_100m = preset_config("hpc-100m")

    assert config_150m.mesh_size == pytest.approx(149.42528735632183)
    assert config_150m.z_extent / config_150m.mesh_size == pytest.approx(53.0)
    assert config_150m.operator_batch_size == 2048
    assert config_150m.symmetry_reduced
    assert config_160m.mesh_size == pytest.approx(158.53658536585365)
    assert config_160m.z_extent / config_160m.mesh_size == pytest.approx(50.0)
    assert config_160m.symmetry_reduced
    assert config_100m.mesh_size == 100.0
    assert config_100m.z_extent == 8_000.0
    assert config_100m.symmetry_reduced


def test_expanded_domain_remains_aligned_to_100m_mesh():
    config = replace(
        preset_config("hpc-100m"),
        x_min=-30_000.0,
        x_max=30_000.0,
        y_min=-16_000.0,
        y_max=30_000.0,
    )

    assert (config.x_max - config.x_min) / config.mesh_size == 600
    assert (config.y_max - config.y_min) / config.mesh_size == 460


def test_graded_mesh_preserves_fine_region_and_smoothly_reaches_boundaries():
    config = replace(
        preset_config("hpc-100m"),
        graded_mesh=True,
        x_min=-50_000.0,
        x_max=50_000.0,
        y_min=-35_000.0,
        y_max=50_000.0,
        z_extent=30_000.0,
        fine_x_min=-18_000.0,
        fine_x_max=18_000.0,
        fine_y_min=-3_000.0,
        fine_y_max=18_000.0,
        fine_z_extent=8_000.0,
        max_mesh_size=750.0,
        mesh_growth_ratio=1.05,
    )

    x, y, z = _block_axes(config, plus_side=True)
    for axis, lower, upper in (
        (x, config.x_min, config.x_max),
        (y, config.y_min, config.y_max),
        (z, 0.0, config.z_extent),
    ):
        widths = np.diff(axis)
        ratios = np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
        assert axis[[0, -1]] == pytest.approx([lower, upper])
        assert np.min(widths) == pytest.approx(config.mesh_size)
        assert np.max(widths) <= config.max_mesh_size * (1.0 + 1.0e-12)
        assert np.max(ratios) <= config.mesh_growth_ratio * (1.0 + 1.0e-12)

    fine_x = x[(x >= config.fine_x_min) & (x <= config.fine_x_max)]
    fine_y = y[(y >= config.fine_y_min) & (y <= config.fine_y_max)]
    fine_z = z[z <= config.fine_z_extent]
    assert np.diff(fine_x) == pytest.approx(config.mesh_size)
    assert np.diff(fine_y) == pytest.approx(config.mesh_size)
    assert np.diff(fine_z) == pytest.approx(config.mesh_size)

    size = estimate_problem_size(config)
    assert size["graded_mesh"]
    assert size["min_cell_size"] == pytest.approx(100.0)
    assert size["max_cell_size"] <= 750.0 * (1.0 + 1.0e-12)


def test_nonuniform_trapezoid_weights_integrate_axis_length():
    axis = np.asarray([0.0, 1.0, 3.0, 6.0])
    weights = _trapezoid_weights(axis)

    assert weights == pytest.approx([0.5, 1.5, 2.5, 1.5])
    assert np.sum(weights) == pytest.approx(axis[-1] - axis[0])


def test_horizontal_shear_tail_metrics_detect_late_uplift():
    time = np.linspace(0.0, 15.0, 301)
    reference = np.column_stack((time, -2.0 - 0.1 * time))
    candidate = np.column_stack((time, reference[:, 1] + np.maximum(time - 10.0, 0.0)))

    metrics = _comparison_metrics(candidate, reference, 10.0, 15.0, 0.5)

    assert metrics["tail_change_mpa"] > 0.0
    assert metrics["late_bias_mpa"] == pytest.approx(4.75, abs=0.03)
    assert metrics["tail_rmse_mpa"] > 0.0


def test_initial_state_is_exact_inverse_at_vw_and_strengthening_points():
    config = TPV101Config()
    direct_effect = jnp.asarray([0.008, 0.016])
    velocity = jnp.full((2,), config.initial_slip_rate)
    normal = jnp.full((2,), config.normal_stress)
    shear = jnp.full((2,), config.initial_shear_stress)

    state = regularized_rate_state_initial_state(
        velocity,
        shear,
        normal,
        reference_friction=config.reference_friction,
        direct_effect=direct_effect,
        state_effect=config.state_effect,
        reference_velocity=config.reference_velocity,
        characteristic_slip=config.characteristic_slip,
    )
    recovered = regularized_rate_state_strength(
        velocity,
        normal,
        state,
        reference_friction=config.reference_friction,
        direct_effect=direct_effect,
        state_effect=config.state_effect,
        reference_velocity=config.reference_velocity,
        characteristic_slip=config.characteristic_slip,
    )

    assert float(state[0]) == pytest.approx(1.606238999213454e9, rel=2.0e-12)
    assert float(state[1]) > float(state[0])
    assert np.asarray(recovered) == pytest.approx([75.0e6, 75.0e6], rel=2.0e-12)


def test_smoke_simulation_writes_a_valid_partial_scec_dump(tmp_path):
    config = replace(preset_config("smoke"), duration=0.02)

    result = run_simulation(config)
    write_scec_dump(result, tmp_path)
    report = validate_dump(tmp_path, require_full_duration=False)

    assert report["valid"], report["errors"]
    assert report["contour"]["ruptured_nodes"] == 0
    assert report["stations"]["faultst000dp075"]["final_time"] == pytest.approx(
        0.02
    )

    full_report = validate_dump(tmp_path)
    assert full_report["valid"], full_report["errors"]
    assert full_report["expected_duration"] == pytest.approx(0.02)


def test_validator_rejects_dynamic_normal_stress_for_tpv101(tmp_path):
    config = replace(preset_config("smoke"), duration=0.02)
    write_scec_dump(run_simulation(config), tmp_path)
    station_path = tmp_path / "faultst000dp075.txt"
    lines = station_path.read_text(encoding="ascii").splitlines()
    fields_index = next(index for index, line in enumerate(lines) if line.startswith("t "))
    values = lines[fields_index + 2].split()
    values[7] = "1.190000E+02"
    lines[fields_index + 2] = " ".join(values)
    station_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    report = validate_dump(tmp_path)

    assert not report["valid"]
    assert any("normal stress varies" in error for error in report["errors"])


def test_symmetry_reduction_matches_full_domain_for_primary_outputs():
    full_config = replace(preset_config("smoke"), duration=0.02)
    symmetry_config = replace(full_config, symmetry_reduced=True)

    full = run_simulation(full_config)
    symmetry = run_simulation(symmetry_config)

    primary_columns = [0, 1, 2, 3, 7, 8]
    assert symmetry["station_history"][:, :, primary_columns] == pytest.approx(
        full["station_history"][:, :, primary_columns], abs=1.0e-12
    )
    assert symmetry["rupture_arrival"] == pytest.approx(full["rupture_arrival"])


def test_checkpoint_resume_reproduces_completed_smoke_run(tmp_path):
    config = replace(
        preset_config("smoke"),
        duration=0.02,
        symmetry_reduced=True,
    )
    checkpoint = tmp_path / "restart.npz"

    original = run_simulation(
        config,
        checkpoint_path=checkpoint,
        checkpoint_interval_s=0.01,
    )
    resumed = run_simulation(
        config,
        checkpoint_path=checkpoint,
        checkpoint_interval_s=0.01,
        resume=True,
    )

    assert resumed["station_history"] == pytest.approx(original["station_history"])
    assert resumed["final_state"] == pytest.approx(original["final_state"])
    assert resumed["rupture_arrival"] == pytest.approx(original["rupture_arrival"])


def test_contour_comparison_interpolates_cell_centered_reference():
    reference = np.asarray(
        [
            [-0.5, -0.5, 1.0],
            [-0.5, 0.5, 2.0],
            [0.5, -0.5, 2.0],
            [0.5, 0.5, 3.0],
        ]
    )
    candidate = np.asarray([[0.0, 0.0, 2.1]])

    metrics = _contour_metrics(candidate, reference, comparison_time=3.0)

    assert metrics["matched_coordinates"] == 1
    assert metrics["both_ruptured_nodes"] == 1
    assert metrics["rupture_time_bias_s"] == pytest.approx(0.1)


def test_reference_filter_selects_cvws_ke_as_uguca(tmp_path):
    (tmp_path / "dunham_100m").mkdir()
    (tmp_path / "ke_100m").mkdir()

    references = _reference_submissions(tmp_path, {"ke"})

    assert len(references) == 1
    assert references[0].path.name == "ke_100m"
    assert references[0].label == "UGUCA (User: ke, Chun-Yu Ke)"


def test_reference_filter_selects_kaneko_as_specfem3d(tmp_path):
    (tmp_path / "kaneko_100m").mkdir()
    (tmp_path / "ke_100m").mkdir()

    references = _reference_submissions(tmp_path, {"kaneko"})

    assert len(references) == 1
    assert references[0].path.name == "kaneko_100m"
    assert references[0].label == "SPECFEM3D (Kaneko et al.)"


def test_uniform_time_step_accepts_scec_ascii_rounding():
    formatted = np.asarray(
        [float(f"{value:.12E}") for value in np.arange(3001) * 0.005]
    )
    assert _has_uniform_time_step(formatted)

    formatted[1500] += 1.0e-5
    assert not _has_uniform_time_step(formatted)


def test_station_metric_summary_groups_reference_statistics():
    rows = [
        {
            "candidate": "Tatva",
            "mesh_size_m": 160.0,
            "reference": "SPECFEM3D",
            "station": station,
            "metric": "arrival_difference_s",
            "value": value,
        }
        for station, value in (("A", 0.01), ("B", 0.03))
    ]

    summary = _summarize_station_metrics(rows)

    assert len(summary) == 1
    assert summary[0]["station_count"] == 2
    assert summary[0]["mean"] == pytest.approx(0.02)
    assert summary[0]["median"] == pytest.approx(0.02)
