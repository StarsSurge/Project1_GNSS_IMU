"""Checks for static detection, leveling, and gyrocompass initialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnss_imu import (
    StaticAlignmentConfig,
    TimedIMUIncrement,
    euler_zyx_to_quat,
    earth_rate_ned,
    initialize_from_static_imu,
    normal_gravity_mps2,
    quat_to_dcm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_static_samples(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    latitude_rad: float,
    height_m: float,
    gyro_bias_rps: np.ndarray | None = None,
    count: int = 1000,
    dt: float = 0.01,
) -> list[TimedIMUIncrement]:
    q_bn = euler_zyx_to_quat(
        roll_deg,
        pitch_deg,
        yaw_deg,
        degrees=True,
    )
    c_nb = quat_to_dcm(q_bn).T
    gravity = normal_gravity_mps2(latitude_rad, height_m)
    specific_force_b = c_nb @ np.array([0.0, 0.0, -gravity])
    gyro_rate_b = c_nb @ earth_rate_ned(latitude_rad)
    if gyro_bias_rps is not None:
        gyro_rate_b = gyro_rate_b + gyro_bias_rps
    return [
        TimedIMUIncrement(
            time_s=(index + 1) * dt,
            dtheta_rad=gyro_rate_b * dt,
            dvel_mps=specific_force_b * dt,
            dt_s=dt,
        )
        for index in range(count)
    ]


def test_external_yaw_leveling_and_gyro_bias_estimation() -> None:
    latitude = np.deg2rad(30.0)
    bias = np.array([2.0e-5, -1.0e-5, 3.0e-5])
    samples = make_static_samples(3.0, -4.0, 120.0, latitude, 50.0, bias)

    result = initialize_from_static_imu(
        samples,
        latitude_rad=latitude,
        longitude_rad=np.deg2rad(114.0),
        height_m=50.0,
        yaw_rad=np.deg2rad(120.0),
    )

    assert result.diagnostics.roll_deg == pytest.approx(3.0, abs=1e-10)
    assert result.diagnostics.pitch_deg == pytest.approx(-4.0, abs=1e-10)
    assert result.diagnostics.yaw_deg == pytest.approx(120.0, abs=1e-10)
    np.testing.assert_allclose(result.state.gyro_bias_rps, bias, atol=1e-12)


def test_static_accelerometer_without_heading_source_is_rejected() -> None:
    latitude = np.deg2rad(30.0)
    samples = make_static_samples(0.0, 0.0, 0.0, latitude, 0.0)

    with pytest.raises(ValueError, match="yaw is unobservable"):
        initialize_from_static_imu(
            samples,
            latitude_rad=latitude,
            longitude_rad=0.0,
            height_m=0.0,
        )


def test_moving_window_is_rejected() -> None:
    latitude = np.deg2rad(30.0)
    samples = make_static_samples(0.0, 0.0, 0.0, latitude, 0.0)
    corrupted = list(samples)
    for index in range(0, len(corrupted), 2):
        sample = corrupted[index]
        corrupted[index] = TimedIMUIncrement(
            sample.time_s,
            sample.dtheta_rad + np.array([0.01, 0.0, 0.0]) * sample.dt_s,
            sample.dvel_mps,
            sample.dt_s,
        )

    with pytest.raises(ValueError, match="gyro variation"):
        initialize_from_static_imu(
            corrupted,
            latitude_rad=latitude,
            longitude_rad=0.0,
            height_m=0.0,
            yaw_rad=0.0,
        )


def test_dataset1_navigation_grade_gyrocompass_is_close_to_truth() -> None:
    table = np.loadtxt(
        PROJECT_ROOT / "data" / "dataset1" / "Leador-A15.txt",
        max_rows=10000,
    )
    times = table[:, 0]
    dt = np.diff(times)
    samples = [
        TimedIMUIncrement(
            times[index],
            table[index, 1:4],
            table[index, 4:7],
            dt[index - 1],
        )
        for index in range(1, table.shape[0])
    ]

    result = initialize_from_static_imu(
        samples,
        latitude_rad=np.deg2rad(30.4447873710),
        longitude_rad=np.deg2rad(114.4718631927),
        height_m=20.904,
        use_gyrocompass=True,
        config=StaticAlignmentConfig(min_samples=5000, min_duration_s=20.0),
    )

    yaw_error = (
        result.diagnostics.yaw_deg - 185.67273 + 180.0
    ) % 360.0 - 180.0
    assert result.diagnostics.roll_deg == pytest.approx(0.85266, abs=0.05)
    assert result.diagnostics.pitch_deg == pytest.approx(-2.03401, abs=0.05)
    assert abs(yaw_error) < 2.0
    assert result.diagnostics.yaw_source == "gyrocompass"
