"""Tests for dataset visualization math helpers."""

from __future__ import annotations

import numpy as np
import pytest

from gnss_imu.dataset_visualization import (
    WGS84_A_M,
    WGS84_E2,
    decimation_indices,
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
