"""Deterministic checks for the production-oriented loose ESKF baseline."""

from __future__ import annotations

import numpy as np
import pytest

from gnss_imu import euler_zyx_to_quat
from gnss_imu.loosely_coupled_eskf import (
    ESKFConfig,
    ESKFState,
    GNSSPositionMeasurement,
    IMUCalibration,
    IMUNoiseModel,
    LooselyCoupledESKF,
    TimedIMUIncrement,
    apply_ned_position_delta,
    default_initial_covariance,
    earth_rate_ned,
    normal_gravity_mps2,
    quat_to_dcm,
    radii_of_curvature,
)


def make_state(
    *,
    time_s: float = 0.0,
    covariance: np.ndarray | None = None,
) -> ESKFState:
    return ESKFState(
        time_s=time_s,
        latitude_rad=np.deg2rad(30.0),
        longitude_rad=np.deg2rad(114.0),
        height_m=20.0,
        velocity_ned_mps=np.zeros(3),
        q_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        accel_bias_mps2=np.zeros(3),
        gyro_bias_rps=np.zeros(3),
        covariance=(
            default_initial_covariance() if covariance is None else covariance
        ),
    )


def static_imu_pair(state: ESKFState, dt_sub: float = 0.005):
    omega_ie_b = quat_to_dcm(state.q_bn).T @ earth_rate_ned(
        state.latitude_rad
    )
    gravity = normal_gravity_mps2(state.latitude_rad, state.height_m)
    dtheta = omega_ie_b * dt_sub
    dvel = np.array([0.0, 0.0, -gravity * dt_sub])
    return (
        TimedIMUIncrement(state.time_s + dt_sub, dtheta, dvel, dt_sub),
        TimedIMUIncrement(state.time_s + 2.0 * dt_sub, dtheta, dvel, dt_sub),
    )


def test_wgs84_helpers_have_expected_magnitudes() -> None:
    rm, rn = radii_of_curvature(np.deg2rad(30.0))

    assert 6.3e6 < rm < 6.4e6
    assert 6.3e6 < rn < 6.4e6
    assert 9.7 < normal_gravity_mps2(np.deg2rad(30.0), 20.0) < 9.9
    np.testing.assert_allclose(
        np.linalg.norm(earth_rate_ned(np.deg2rad(30.0))),
        7.2921151467e-5,
        rtol=1e-12,
    )


def test_static_earth_fixed_state_remains_nearly_stationary() -> None:
    state = make_state()
    eskf = LooselyCoupledESKF(state)
    imu1, imu2 = static_imu_pair(state)

    eskf.predict_two_sample(imu1, imu2)

    np.testing.assert_allclose(eskf.state.velocity_ned_mps, 0.0, atol=1e-9)
    np.testing.assert_allclose(eskf.state.height_m, 20.0, atol=1e-11)
    np.testing.assert_allclose(
        quat_to_dcm(eskf.state.q_bn),
        np.eye(3),
        atol=1e-10,
    )


def test_covariance_stays_symmetric_positive_semidefinite() -> None:
    eskf = LooselyCoupledESKF(make_state())

    for _ in range(100):
        imu1, imu2 = static_imu_pair(eskf.state)
        eskf.predict_two_sample(imu1, imu2)

    np.testing.assert_allclose(
        eskf.state.covariance,
        eskf.state.covariance.T,
        atol=1e-12,
    )
    assert np.linalg.eigvalsh(eskf.state.covariance)[0] >= -1e-12


def test_gnss_position_update_pulls_state_toward_measurement() -> None:
    state = make_state(covariance=np.eye(15))
    eskf = LooselyCoupledESKF(state)
    lat, lon, height = apply_ned_position_delta(
        state.latitude_rad,
        state.longitude_rad,
        state.height_m,
        np.array([1.0, 0.0, 0.0]),
    )
    measurement = GNSSPositionMeasurement(
        state.time_s,
        lat,
        lon,
        height,
        np.full(3, 0.1),
    )

    result = eskf.update_gnss_position(measurement)

    assert result.accepted
    north_correction = (
        eskf.state.latitude_rad - np.deg2rad(30.0)
    ) * radii_of_curvature(np.deg2rad(30.0))[0]
    assert 0.98 < north_correction < 1.0
    assert eskf.accepted_gnss_updates == 1


def test_gnss_lever_arm_is_part_of_measurement_model() -> None:
    lever_b = np.array([1.0, -0.2, 0.3])
    config = ESKFConfig(gnss_lever_arm_b_m=lever_b)
    state = make_state()
    state.q_bn = euler_zyx_to_quat(0.0, 0.0, 90.0, degrees=True)
    eskf = LooselyCoupledESKF(state, config)
    antenna_delta_n = quat_to_dcm(state.q_bn) @ lever_b
    lat, lon, height = apply_ned_position_delta(
        state.latitude_rad,
        state.longitude_rad,
        state.height_m,
        antenna_delta_n,
    )

    result = eskf.update_gnss_position(
        GNSSPositionMeasurement(
            state.time_s,
            lat,
            lon,
            height,
            np.full(3, 0.02),
        )
    )

    assert result.accepted
    np.testing.assert_allclose(result.residual_ned_m, 0.0, atol=1e-8)


def test_gnss_outlier_is_rejected_by_nis_gate() -> None:
    state = make_state(covariance=np.eye(15) * 0.01)
    eskf = LooselyCoupledESKF(state)
    lat, lon, height = apply_ned_position_delta(
        state.latitude_rad,
        state.longitude_rad,
        state.height_m,
        np.array([100.0, 0.0, 0.0]),
    )

    result = eskf.update_gnss_position(
        GNSSPositionMeasurement(
            state.time_s,
            lat,
            lon,
            height,
            np.full(3, 0.1),
        )
    )

    assert not result.accepted
    assert eskf.rejected_gnss_updates == 1


def test_sensor_profiles_are_distinct_and_positive() -> None:
    mems = IMUNoiseModel.mems_default()
    navigation = IMUNoiseModel.navigation_grade_default()

    assert mems.gyro_noise_density_rps_sqrthz > navigation.gyro_noise_density_rps_sqrthz
    assert mems.accel_noise_density_mps2_sqrthz > navigation.accel_noise_density_mps2_sqrthz


def test_known_imu_calibration_matrix_is_applied_before_mechanization() -> None:
    state = make_state()
    gravity = normal_gravity_mps2(state.latitude_rad, state.height_m)
    calibration = IMUCalibration(
        gyro_increment_matrix=np.eye(3),
        accel_increment_matrix=np.diag([1.0, 1.0, 0.5]),
    )
    eskf = LooselyCoupledESKF(state, ESKFConfig(imu_calibration=calibration))
    omega_ie = earth_rate_ned(state.latitude_rad)
    # Reported z delta-v is twice the physical value; calibration removes it.
    imu1 = TimedIMUIncrement(
        0.005,
        omega_ie * 0.005,
        [0.0, 0.0, -2.0 * gravity * 0.005],
        0.005,
    )
    imu2 = TimedIMUIncrement(
        0.010,
        omega_ie * 0.005,
        [0.0, 0.0, -2.0 * gravity * 0.005],
        0.005,
    )

    eskf.predict_two_sample(imu1, imu2)

    np.testing.assert_allclose(eskf.state.velocity_ned_mps, 0.0, atol=1e-9)


def test_noncontiguous_imu_and_stale_gnss_are_rejected() -> None:
    state = make_state()
    eskf = LooselyCoupledESKF(state)
    gravity = normal_gravity_mps2(state.latitude_rad, state.height_m)
    imu1 = TimedIMUIncrement(
        0.015,
        np.zeros(3),
        [0.0, 0.0, -gravity * 0.005],
        0.005,
    )
    imu2 = TimedIMUIncrement(
        0.020,
        np.zeros(3),
        [0.0, 0.0, -gravity * 0.005],
        0.005,
    )

    with pytest.raises(ValueError, match="not contiguous"):
        eskf.predict_two_sample(imu1, imu2)

    with pytest.raises(ValueError, match="time mismatch"):
        eskf.update_gnss_position(
            GNSSPositionMeasurement(
                time_s=1.0,
                latitude_rad=state.latitude_rad,
                longitude_rad=state.longitude_rad,
                height_m=state.height_m,
                std_ned_m=np.ones(3),
            )
        )
