"""Verification for delayed GNSS replay and constant time-offset calibration."""

from __future__ import annotations

import numpy as np
import pytest

from gnss_imu.delayed_eskf import (
    FixedLagGNSSFusion,
    TimeOffsetCalibrationResult,
    TimeOffsetCandidateScore,
    calibrate_constant_gnss_time_offset,
    clone_eskf_state,
    compare_clock_offset_models,
    estimate_time_offset_profile_interval,
)
from gnss_imu.loosely_coupled_eskf import (
    ESKFConfig,
    ESKFState,
    GNSSPositionMeasurement,
    LooselyCoupledESKF,
    TimedIMUIncrement,
    apply_ned_position_delta,
    earth_rate_ned,
    normal_gravity_mps2,
    quat_to_dcm,
    transport_rate_ned,
)


def make_state(*, velocity_ned_mps: np.ndarray | None = None) -> ESKFState:
    return ESKFState(
        time_s=0.0,
        latitude_rad=np.deg2rad(30.0),
        longitude_rad=np.deg2rad(114.0),
        height_m=20.0,
        velocity_ned_mps=(
            np.zeros(3) if velocity_ned_mps is None else velocity_ned_mps
        ),
        q_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        accel_bias_mps2=np.zeros(3),
        gyro_bias_rps=np.zeros(3),
        covariance=np.eye(15) * 1e-3,
    )


def static_sample(state: ESKFState, end_time_s: float, dt_s: float) -> TimedIMUIncrement:
    c_bn = quat_to_dcm(state.q_bn)
    omega_ie_b = c_bn.T @ earth_rate_ned(state.latitude_rad)
    gravity = normal_gravity_mps2(state.latitude_rad, state.height_m)
    return TimedIMUIncrement(
        end_time_s,
        omega_ie_b * dt_s,
        np.array([0.0, 0.0, -gravity * dt_s]),
        dt_s,
    )


def assert_states_equal(actual: ESKFState, expected: ESKFState) -> None:
    assert actual.time_s == pytest.approx(expected.time_s, abs=1e-12)
    assert actual.latitude_rad == pytest.approx(expected.latitude_rad, abs=1e-14)
    assert actual.longitude_rad == pytest.approx(expected.longitude_rad, abs=1e-14)
    assert actual.height_m == pytest.approx(expected.height_m, abs=1e-10)
    np.testing.assert_allclose(actual.velocity_ned_mps, expected.velocity_ned_mps, atol=1e-11)
    np.testing.assert_allclose(actual.q_bn, expected.q_bn, atol=1e-12)
    np.testing.assert_allclose(actual.accel_bias_mps2, expected.accel_bias_mps2, atol=1e-12)
    np.testing.assert_allclose(actual.gyro_bias_rps, expected.gyro_bias_rps, atol=1e-12)
    np.testing.assert_allclose(actual.covariance, expected.covariance, atol=1e-11)


def test_delayed_update_matches_manual_mid_interval_split() -> None:
    config = ESKFConfig(gnss_nis_threshold=1e6)
    initial = make_state()
    samples = [static_sample(initial, 0.01 * index, 0.01) for index in range(1, 11)]
    latitude, longitude, height = apply_ned_position_delta(
        initial.latitude_rad,
        initial.longitude_rad,
        initial.height_m,
        np.array([1.0, 0.0, 0.0]),
    )
    measurement = GNSSPositionMeasurement(
        0.055,
        latitude,
        longitude,
        height,
        np.full(3, 0.1),
    )

    reference = LooselyCoupledESKF(clone_eskf_state(initial), config)
    for sample in samples[:5]:
        reference.predict_single_sample(sample)
    crossing = samples[5]
    reference.predict_single_sample(
        TimedIMUIncrement(0.055, crossing.dtheta_rad * 0.5, crossing.dvel_mps * 0.5, 0.005)
    )
    reference.update_gnss_position(measurement)
    reference.predict_single_sample(
        TimedIMUIncrement(0.06, crossing.dtheta_rad * 0.5, crossing.dvel_mps * 0.5, 0.005)
    )
    for sample in samples[6:]:
        reference.predict_single_sample(sample)

    delayed_filter = LooselyCoupledESKF(clone_eskf_state(initial), config)
    fusion = FixedLagGNSSFusion(delayed_filter, lag_s=0.2)
    for sample in samples:
        fusion.process_imu(sample)
    result = fusion.process_gnss(measurement, arrival_time_s=0.1)

    assert result.update.accepted
    assert result.effective_time_s == pytest.approx(0.055)
    assert result.rewind_s == pytest.approx(0.045)
    assert result.replayed_imu_samples == 5
    assert_states_equal(fusion.eskf.state, reference.state)


def test_measurement_older_than_fixed_lag_is_rejected() -> None:
    initial = make_state()
    fusion = FixedLagGNSSFusion(LooselyCoupledESKF(initial), lag_s=0.03)
    for index in range(1, 11):
        fusion.process_imu(static_sample(initial, 0.01 * index, 0.01))
    measurement = GNSSPositionMeasurement(
        0.01,
        initial.latitude_rad,
        initial.longitude_rad,
        initial.height_m,
        np.ones(3),
    )

    with pytest.raises(ValueError, match="older than the fixed-lag"):
        fusion.process_gnss(measurement, arrival_time_s=0.1)


def test_future_arrival_time_is_rejected() -> None:
    initial = make_state()
    fusion = FixedLagGNSSFusion(LooselyCoupledESKF(initial), lag_s=0.1)
    fusion.process_imu(static_sample(initial, 0.01, 0.01))
    measurement = GNSSPositionMeasurement(
        0.01,
        initial.latitude_rad,
        initial.longitude_rad,
        initial.height_m,
        np.ones(3),
    )

    with pytest.raises(ValueError, match="cannot exceed the current filter time"):
        fusion.process_gnss(measurement, arrival_time_s=0.02)


def make_accelerating_sequence() -> tuple[
    ESKFState,
    list[TimedIMUIncrement],
    dict[float, ESKFState],
]:
    initial = make_state(velocity_ned_mps=np.array([3.0, 0.0, 0.0]))
    truth = LooselyCoupledESKF(clone_eskf_state(initial))
    samples: list[TimedIMUIncrement] = []
    states = {0.0: clone_eskf_state(truth.state)}
    dt_s = 0.01
    desired_acceleration_n = np.array([1.5, 0.0, 0.0])
    for index in range(1, 301):
        state = truth.state
        c_bn = quat_to_dcm(state.q_bn)
        omega_ie_n = earth_rate_ned(state.latitude_rad)
        omega_en_n = transport_rate_ned(
            state.latitude_rad,
            state.height_m,
            state.velocity_ned_mps,
        )
        gravity_n = np.array(
            [0.0, 0.0, normal_gravity_mps2(state.latitude_rad, state.height_m)]
        )
        coriolis_n = -np.cross(
            2.0 * omega_ie_n + omega_en_n,
            state.velocity_ned_mps,
        )
        specific_force_n = desired_acceleration_n - gravity_n - coriolis_n
        sample = TimedIMUIncrement(
            index * dt_s,
            c_bn.T @ (omega_ie_n + omega_en_n) * dt_s,
            c_bn.T @ specific_force_n * dt_s,
            dt_s,
        )
        truth.predict_single_sample(sample)
        samples.append(sample)
        states[round(index * dt_s, 2)] = clone_eskf_state(truth.state)
    return initial, samples, states


def test_constant_time_offset_calibration_recovers_known_offset() -> None:
    initial, samples, truth_states = make_accelerating_sequence()
    true_offset_s = 0.04
    measurements: list[GNSSPositionMeasurement] = []
    for reported_time_s in np.arange(0.2, 2.61, 0.2):
        state = truth_states[round(float(reported_time_s + true_offset_s), 2)]
        measurements.append(
            GNSSPositionMeasurement(
                float(reported_time_s),
                state.latitude_rad,
                state.longitude_rad,
                state.height_m,
                np.full(3, 0.03),
            )
        )

    result = calibrate_constant_gnss_time_offset(
        initial,
        ESKFConfig(gnss_nis_threshold=1e9),
        samples,
        measurements,
        [-0.04, 0.0, 0.04],
        lag_s=0.2,
        min_peak_speed_mps=2.0,
    )

    assert result.best_offset_s == pytest.approx(true_offset_s)
    score_by_offset = {score.offset_s: score for score in result.scores}
    assert score_by_offset[true_offset_s].robust_mean_nis < score_by_offset[0.0].robust_mean_nis
    assert all(score.measurement_count == len(measurements) for score in result.scores)


def test_time_offset_calibration_rejects_unobservable_static_data() -> None:
    initial = make_state()
    samples = [static_sample(initial, 0.01 * index, 0.01) for index in range(1, 101)]
    measurements = [
        GNSSPositionMeasurement(
            time_s,
            initial.latitude_rad,
            initial.longitude_rad,
            initial.height_m,
            np.ones(3),
        )
        for time_s in (0.2, 0.4, 0.6, 0.8)
    ]

    with pytest.raises(ValueError, match="motion excitation is insufficient"):
        calibrate_constant_gnss_time_offset(
            initial,
            ESKFConfig(),
            samples,
            measurements,
            [-0.02, 0.02],
            lag_s=0.1,
            min_peak_speed_mps=1.0,
        )


def make_profile_result(best_at_boundary: bool = False) -> TimeOffsetCalibrationResult:
    offsets = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    if best_at_boundary:
        total_nis = ((offsets - 0.02) / 0.01) ** 2
        best_offset = 0.02
    else:
        total_nis = (offsets / 0.01) ** 2
        best_offset = 0.0
    measurement_count = 10
    scores = tuple(
        TimeOffsetCandidateScore(
            offset_s=float(offset),
            robust_mean_nis=float(objective / measurement_count),
            median_nis=float(objective / measurement_count),
            measurement_count=measurement_count,
            accepted_count=measurement_count,
            rejected_count=0,
        )
        for offset, objective in zip(offsets, total_nis)
    )
    return TimeOffsetCalibrationResult(best_offset, scores, peak_speed_mps=5.0)


def test_profile_interval_interpolates_threshold_crossings() -> None:
    interval = estimate_time_offset_profile_interval(make_profile_result())

    assert interval.best_offset_s == pytest.approx(0.0)
    assert interval.lower_bounded
    assert interval.upper_bounded
    assert interval.lower_offset_s == pytest.approx(-0.0194715, abs=1e-6)
    assert interval.upper_offset_s == pytest.approx(0.0194715, abs=1e-6)
    assert interval.standard_uncertainty_s is not None


def test_profile_interval_reports_scan_boundary_as_unbounded() -> None:
    interval = estimate_time_offset_profile_interval(
        make_profile_result(best_at_boundary=True)
    )

    assert interval.lower_bounded
    assert interval.upper_offset_s is None
    assert not interval.upper_bounded
    assert interval.standard_uncertainty_s is None


def test_profile_interval_refines_best_offset_below_grid_spacing() -> None:
    offsets = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    true_offset = 0.003
    measurement_count = 10
    objectives = ((offsets - true_offset) / 0.01) ** 2
    scores = tuple(
        TimeOffsetCandidateScore(
            float(offset),
            float(objective / measurement_count),
            float(objective / measurement_count),
            measurement_count,
            measurement_count,
            0,
        )
        for offset, objective in zip(offsets, objectives)
    )
    calibration = TimeOffsetCalibrationResult(0.0, scores, peak_speed_mps=5.0)

    interval = estimate_time_offset_profile_interval(calibration)

    assert interval.best_offset_s == pytest.approx(true_offset, abs=1e-12)


def test_profile_interval_cannot_claim_subgrid_resolution() -> None:
    offsets = np.array([-0.01, 0.0, 0.01])
    scores = tuple(
        TimeOffsetCandidateScore(float(offset), score, score, 20, 20, 0)
        for offset, score in zip(offsets, (100.0, 0.0, 100.0))
    )
    calibration = TimeOffsetCalibrationResult(0.0, scores, peak_speed_mps=5.0)

    interval = estimate_time_offset_profile_interval(calibration)

    assert interval.resolution_limited
    assert interval.lower_offset_s == pytest.approx(-0.005)
    assert interval.upper_offset_s == pytest.approx(0.005)
    assert interval.standard_uncertainty_s > 0.0


def test_clock_model_comparison_selects_constant_offset() -> None:
    times = np.arange(8, dtype=float) * 100.0
    offsets = 0.012 + np.array(
        [-0.0002, 0.0001, 0.0002, -0.0001, 0.0001, -0.0002, 0.0002, -0.0001]
    )

    comparison = compare_clock_offset_models(times, offsets, np.full(8, 0.001))

    assert comparison.preferred_model == "constant"
    assert comparison.constant_offset_s == pytest.approx(0.012, abs=1e-4)
    assert comparison.drift_ci95_s_per_s[0] < 0.0 < comparison.drift_ci95_s_per_s[1]

    unweighted = compare_clock_offset_models(times, offsets)
    assert unweighted.preferred_model == "constant"
    assert np.all(np.isfinite(unweighted.constant_offset_ci95_s))


def test_clock_model_comparison_detects_linear_drift() -> None:
    times = np.arange(8, dtype=float) * 100.0
    reference_time = float(np.mean(times))
    true_drift = 20e-6
    offsets = 0.005 + true_drift * (times - reference_time)
    offsets += np.array([-1.0, 0.5, 0.2, -0.3, 0.4, -0.2, 0.3, -0.1]) * 1e-5

    comparison = compare_clock_offset_models(times, offsets, np.full(8, 0.0002))

    assert comparison.preferred_model == "linear-drift"
    assert comparison.drift_s_per_s == pytest.approx(true_drift, rel=0.01)
    assert comparison.drift_ppm == pytest.approx(20.0, rel=0.01)
    assert comparison.delta_bic_constant_minus_linear > 6.0
    assert comparison.drift_ci95_s_per_s[0] > 0.0
