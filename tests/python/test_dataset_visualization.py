"""Tests for dataset visualization math helpers."""

from __future__ import annotations

import numpy as np
import pytest

from gnss_imu.dataset_visualization import (
    WGS84_A_M,
    WGS84_E2,
    decimation_indices,
    calibrate_lever_arm_and_time_offset,
    calibrate_time_offset_with_fixed_lever,
    antenna_velocity_ned,
    residual_correlation_diagnostics,
    score_frozen_trajectory_time_offsets,
    frozen_trajectory_nis_matrix,
    bootstrap_frozen_time_offset,
    fit_body_lever_arm_from_residuals,
    geodetic_to_ned,
    increments_to_rates,
    interpolate_columns,
    rpy_deg_to_body_to_ned,
    sampling_summary,
)
from gnss_imu.imu_mechanization import euler_zyx_to_quat, quat_to_dcm


def test_geodetic_to_ned_matches_small_local_offsets() -> None:
    reference = (30.0, 114.0, 20.0)
    lat_rad = np.deg2rad(reference[0])
    prime_vertical = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat_rad) ** 2)
    meridian = (
        WGS84_A_M
        * (1.0 - WGS84_E2)
        / (1.0 - WGS84_E2 * np.sin(lat_rad) ** 2) ** 1.5
    )
    meters_per_deg_lat = (meridian + reference[2]) * np.pi / 180.0
    meters_per_deg_lon = (
        (prime_vertical + reference[2]) * np.cos(lat_rad) * np.pi / 180.0
    )
    latitude = np.array(
        [
            reference[0],
            reference[0] + 10.0 / meters_per_deg_lat,
            reference[0],
            reference[0],
        ]
    )
    longitude = np.array(
        [
            reference[1],
            reference[1],
            reference[1] + 10.0 / meters_per_deg_lon,
            reference[1],
        ]
    )
    height = np.array([20.0, 20.0, 20.0, 15.0])

    ned = geodetic_to_ned(latitude, longitude, height, reference)

    np.testing.assert_allclose(ned[0], [0.0, 0.0, 0.0], atol=1e-8)
    assert ned[1, 0] == pytest.approx(10.0, rel=0.004)
    assert ned[2, 1] == pytest.approx(10.0, rel=0.004)
    assert ned[3, 2] == pytest.approx(5.0, rel=0.001)


def test_increments_to_rates_uses_backward_intervals() -> None:
    time_s = np.array([0.0, 0.1, 0.3])
    increments = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.4, 0.6],
            [0.3, 0.6, 0.9],
        ]
    )

    rate_time_s, rates = increments_to_rates(time_s, increments)

    np.testing.assert_allclose(rate_time_s, [0.1, 0.3])
    np.testing.assert_allclose(rates[0], [2.0, 4.0, 6.0])
    np.testing.assert_allclose(rates[1], [1.5, 3.0, 4.5])


def test_sampling_summary_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sampling_summary(np.array([1.0, 1.0, 2.0]))


def test_decimation_indices_keeps_endpoints() -> None:
    indices = decimation_indices(length=101, max_points=10)

    assert indices[0] == 0
    assert indices[-1] == 100
    assert indices.size <= 10


def test_interpolate_columns_handles_each_axis() -> None:
    source_time_s = np.array([0.0, 1.0, 2.0])
    source_values = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
    query_time_s = np.array([0.5, 1.5])

    interpolated = interpolate_columns(
        source_time_s, source_values, query_time_s
    )

    np.testing.assert_allclose(interpolated, [[1.0, 15.0], [3.0, 30.0]])


def test_rpy_to_body_to_ned_matches_heading_convention() -> None:
    identity = rpy_deg_to_body_to_ned(0.0, 0.0, 0.0)
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)

    east_heading = rpy_deg_to_body_to_ned(0.0, 0.0, 90.0)
    body_forward = np.array([1.0, 0.0, 0.0])
    body_right = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(east_heading @ body_forward, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(east_heading @ body_right, [-1.0, 0.0, 0.0], atol=1e-12)


def test_rpy_to_body_to_ned_matches_quaternion_for_full_attitude() -> None:
    roll_deg = 12.0
    pitch_deg = -7.0
    yaw_deg = 123.0

    matrix_from_rpy = rpy_deg_to_body_to_ned(
        roll_deg, pitch_deg, yaw_deg
    )
    matrix_from_quaternion = quat_to_dcm(
        euler_zyx_to_quat(
            roll_deg,
            pitch_deg,
            yaw_deg,
            degrees=True,
        )
    )

    np.testing.assert_allclose(
        matrix_from_rpy,
        matrix_from_quaternion,
        atol=1e-12,
    )


def test_fit_body_lever_arm_recovers_known_offset() -> None:
    attitudes = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 180.0],
            [1.0, -2.0, 270.0],
        ]
    )
    rotations = rpy_deg_to_body_to_ned(
        attitudes[:, 0], attitudes[:, 1], attitudes[:, 2]
    )
    expected_lever_b_m = np.array([0.2, -0.3, -0.1])
    residuals = np.einsum("nij,j->ni", rotations, expected_lever_b_m)

    lever_b_m, predicted, corrected = fit_body_lever_arm_from_residuals(
        rotations, residuals
    )

    np.testing.assert_allclose(lever_b_m, expected_lever_b_m, atol=1e-12)
    np.testing.assert_allclose(predicted, residuals, atol=1e-12)
    np.testing.assert_allclose(corrected, 0.0, atol=1e-12)


def test_joint_lever_time_calibration_is_robust_to_position_outliers() -> None:
    rng = np.random.default_rng(42)
    sample_count = 120
    attitudes = np.column_stack(
        [
            rng.uniform(-5.0, 5.0, sample_count),
            rng.uniform(-4.0, 4.0, sample_count),
            rng.uniform(0.0, 360.0, sample_count),
        ]
    )
    rotations = rpy_deg_to_body_to_ned(
        attitudes[:, 0],
        attitudes[:, 1],
        attitudes[:, 2],
    )
    velocities = rng.normal(size=(sample_count, 3)) * np.array([4.0, 3.0, 0.3])
    velocities[:, 0] += 8.0
    angular_rates = rng.normal(scale=0.3, size=(sample_count, 3))
    expected_lever = np.array([0.15, -0.30, -0.18])
    expected_offset_s = 0.004
    residuals = np.einsum("nij,j->ni", rotations, expected_lever)
    antenna_velocity = antenna_velocity_ned(
        rotations,
        velocities,
        angular_rates,
        expected_lever,
    )
    residuals += antenna_velocity * expected_offset_s
    residuals += rng.normal(scale=0.002, size=residuals.shape)
    residuals[[7, 33, 81]] += np.array([2.0, -1.0, 1.5])

    result = calibrate_lever_arm_and_time_offset(
        rotations,
        velocities,
        residuals,
        np.full((sample_count, 3), 0.01),
        angular_rate_b_rps=angular_rates,
    )

    np.testing.assert_allclose(result.lever_arm_b_m, expected_lever, atol=0.005)
    assert result.time_offset_s == pytest.approx(expected_offset_s, abs=5e-4)
    assert result.covariance.shape == (4, 4)
    assert result.correlation.shape == (4, 4)
    assert np.all(result.parameter_std > 0.0)
    assert result.downweighted_epoch_count >= 3
    assert np.linalg.norm(result.corrected_rms_ned_m) < np.linalg.norm(result.raw_rms_ned_m)


def test_joint_lever_time_calibration_rejects_unobservable_motion() -> None:
    rotations = np.repeat(np.eye(3)[np.newaxis, :, :], 10, axis=0)
    velocities = np.repeat(np.array([[10.0, 0.0, 0.0]]), 10, axis=0)
    residuals = np.zeros((10, 3))

    with pytest.raises(ValueError, match="unobservable"):
        calibrate_lever_arm_and_time_offset(
            rotations,
            velocities,
            residuals,
        )


def test_lever_prior_breaks_constant_motion_parameter_ambiguity() -> None:
    rotations = np.repeat(np.eye(3)[np.newaxis, :, :], 10, axis=0)
    velocities = np.repeat(np.array([[10.0, 0.0, 0.0]]), 10, axis=0)
    lever = np.array([0.2, -0.3, -0.1])
    time_offset_s = 0.004
    residuals = np.einsum("nij,j->ni", rotations, lever)
    residuals += velocities * time_offset_s

    result = calibrate_lever_arm_and_time_offset(
        rotations,
        velocities,
        residuals,
        np.full((10, 3), 0.01),
        lever_prior_b_m=lever,
        lever_prior_std_m=np.full(3, 0.001),
    )

    np.testing.assert_allclose(result.lever_arm_b_m, lever, atol=1e-10)
    assert result.time_offset_s == pytest.approx(time_offset_s, abs=1e-10)


def test_fixed_lever_time_calibration_recovers_constant_offset() -> None:
    rotations = np.repeat(np.eye(3)[np.newaxis, :, :], 20, axis=0)
    velocities = np.repeat(np.array([[10.0, 2.0, 0.0]]), 20, axis=0)
    lever = np.array([0.2, -0.3, -0.1])
    expected_offset_s = 0.004
    residuals = np.einsum("nij,j->ni", rotations, lever)
    residuals += velocities * expected_offset_s

    result = calibrate_time_offset_with_fixed_lever(
        rotations,
        velocities,
        residuals,
        lever,
        np.full((20, 3), 0.01),
    )

    assert result.time_offset_s == pytest.approx(expected_offset_s, abs=1e-12)
    assert result.time_offset_std_s > 0.0
    np.testing.assert_allclose(result.corrected_residual_ned_m, 0.0, atol=1e-12)


def test_antenna_rotational_velocity_cross_product_direction() -> None:
    rotations = np.eye(3)[np.newaxis, :, :]
    imu_velocity = np.array([[10.0, 0.0, 0.0]])
    angular_rate = np.array([[0.0, 0.0, 1.0]])
    lever = np.array([2.0, 0.0, 0.0])

    velocity = antenna_velocity_ned(
        rotations,
        imu_velocity,
        angular_rate,
        lever,
    )

    np.testing.assert_allclose(velocity, [[10.0, 2.0, 0.0]], atol=1e-12)


def test_rotation_makes_fixed_lever_time_offset_observable() -> None:
    sample_count = 20
    rotations = np.repeat(np.eye(3)[np.newaxis, :, :], sample_count, axis=0)
    imu_velocity = np.zeros((sample_count, 3))
    angular_rate = np.repeat(np.array([[0.0, 0.0, 1.0]]), sample_count, axis=0)
    lever = np.array([2.0, 0.0, 0.0])
    expected_offset_s = 0.01
    antenna_velocity = antenna_velocity_ned(
        rotations,
        imu_velocity,
        angular_rate,
        lever,
    )
    residuals = np.einsum("nij,j->ni", rotations, lever)
    residuals += antenna_velocity * expected_offset_s

    with pytest.raises(ValueError, match="velocity excitation is zero"):
        calibrate_time_offset_with_fixed_lever(
            rotations,
            imu_velocity,
            residuals,
            lever,
            np.full((sample_count, 3), 0.01),
        )

    result = calibrate_time_offset_with_fixed_lever(
        rotations,
        imu_velocity,
        residuals,
        lever,
        np.full((sample_count, 3), 0.01),
        angular_rate_b_rps=angular_rate,
    )

    assert result.time_offset_s == pytest.approx(expected_offset_s, abs=1e-12)


def test_frozen_trajectory_profile_recovers_known_time_offset() -> None:
    trajectory_time = np.arange(0.0, 10.01, 0.1)
    antenna_position = np.column_stack(
        [2.0 * trajectory_time, np.zeros_like(trajectory_time), np.zeros_like(trajectory_time)]
    )
    measurement_time = np.arange(1.0, 9.0, 1.0)
    true_offset_s = 0.2
    measured_position = np.column_stack(
        [
            2.0 * (measurement_time + true_offset_s),
            np.zeros_like(measurement_time),
            np.zeros_like(measurement_time),
        ]
    )

    scores = score_frozen_trajectory_time_offsets(
        trajectory_time,
        antenna_position,
        measurement_time,
        measured_position,
        np.full((measurement_time.size, 3), 0.01),
        np.array([-0.2, 0.0, 0.2]),
    )

    best = min(scores, key=lambda item: item.robust_mean_nis)
    assert best.offset_s == pytest.approx(true_offset_s)
    assert best.robust_mean_nis == pytest.approx(0.0, abs=1e-20)
    nis_matrix = frozen_trajectory_nis_matrix(
        trajectory_time,
        antenna_position,
        measurement_time,
        measured_position,
        np.full((measurement_time.size, 3), 0.01),
        np.array([-0.2, 0.0, 0.2]),
    )
    assert nis_matrix.shape == (3, measurement_time.size)
    np.testing.assert_allclose(nis_matrix[2], 0.0, atol=1e-20)


def test_residual_correlation_reduces_effective_sample_size() -> None:
    rng = np.random.default_rng(7)
    sample_count = 2000
    white = rng.normal(size=(sample_count, 3))
    correlated = np.empty_like(white)
    correlated[0] = white[0]
    for index in range(1, sample_count):
        correlated[index] = 0.85 * correlated[index - 1] + white[index]

    white_diagnostics = residual_correlation_diagnostics(white, max_lag=100)
    correlated_diagnostics = residual_correlation_diagnostics(
        correlated,
        max_lag=100,
    )

    assert np.all(white_diagnostics.effective_sample_size > 0.5 * sample_count)
    assert np.all(correlated_diagnostics.effective_sample_size < 0.2 * sample_count)
    assert np.all(correlated_diagnostics.lag1_autocorrelation > 0.8)


def test_moving_block_bootstrap_is_reproducible_and_covers_truth() -> None:
    rng = np.random.default_rng(11)
    epoch_count = 160
    candidates = np.arange(-0.04, 0.041, 0.005)
    true_offset_s = 0.012
    correlated_error = np.zeros(epoch_count)
    innovations = rng.normal(scale=0.0015, size=epoch_count)
    for index in range(1, epoch_count):
        correlated_error[index] = 0.8 * correlated_error[index - 1] + innovations[index]
    epoch_best = true_offset_s + correlated_error
    epoch_nis = ((candidates[:, None] - epoch_best[None, :]) / 0.005) ** 2

    first = bootstrap_frozen_time_offset(
        candidates,
        epoch_nis,
        block_length_epochs=10,
        replicate_count=500,
        random_seed=5,
    )
    second = bootstrap_frozen_time_offset(
        candidates,
        epoch_nis,
        block_length_epochs=10,
        replicate_count=500,
        random_seed=5,
    )

    assert first.point_estimate_s == pytest.approx(true_offset_s, abs=0.003)
    assert first.lower_offset_s <= true_offset_s <= first.upper_offset_s
    assert first.upper_offset_s - first.lower_offset_s >= 0.005 - 1e-12
    assert first.boundary_hit_fraction < 0.05
    np.testing.assert_allclose(first.bootstrap_offsets_s, second.bootstrap_offsets_s)
