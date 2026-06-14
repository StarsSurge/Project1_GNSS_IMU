"""Numerical checks for the IMU visual explanation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

VISUAL_DIR = Path(__file__).resolve().parents[2] / "visual_explanations"
sys.path.insert(0, str(VISUAL_DIR))

from imu_visualization_math import (  # noqa: E402
    allan_deviation,
    attitude_error_rotvec,
    coning_correct,
    fit_allan_log_slope,
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
    overlapping_allan_deviation,
    sampling_interval_statistics,
    sculling_rotation_correct,
)


def test_quaternion_rotation_and_error_are_consistent() -> None:
    rotvec = np.array([0.1, -0.2, 0.05])
    q = rotvec_to_quat(rotvec)
    dcm = quat_to_dcm(q)

    np.testing.assert_allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(dcm), 1.0, atol=1e-12)
    np.testing.assert_allclose(attitude_error_rotvec(q, q), 0.0, atol=1e-12)


def test_two_sample_corrections_have_expected_cross_terms() -> None:
    dtheta1 = np.array([0.01, 0.0, 0.0])
    dtheta2 = np.array([0.0, 0.02, 0.0])
    dvel1 = np.array([0.0, 0.0, 0.03])
    dvel2 = np.array([0.0, 0.0, 0.04])

    corrected_angle = coning_correct(dtheta1, dtheta2)
    np.testing.assert_allclose(corrected_angle[2], (2.0 / 3.0) * 0.0002)

    corrected_velocity, sculling, rotation = sculling_rotation_correct(
        dtheta1, dvel1, dtheta2, dvel2
    )
    np.testing.assert_allclose(
        corrected_velocity, dvel1 + dvel2 + sculling + rotation
    )
    assert np.linalg.norm(sculling) > 0.0
    assert np.linalg.norm(rotation) > 0.0


def test_allan_deviation_of_white_noise_has_negative_half_slope() -> None:
    rng = np.random.default_rng(7)
    sample_rate = 100.0
    signal = rng.normal(size=200_000)
    taus, deviation = allan_deviation(
        signal, sample_rate, np.logspace(-1, 1, 20)
    )
    slope = np.polyfit(np.log10(taus), np.log10(deviation), 1)[0]
    assert -0.6 < slope < -0.4


def test_quaternion_composition_matches_dcm_composition() -> None:
    q1 = rotvec_to_quat([0.1, 0.0, 0.0])
    q2 = rotvec_to_quat([0.0, -0.2, 0.0])
    q12 = quat_multiply(q1, q2)
    np.testing.assert_allclose(
        quat_to_dcm(q12), quat_to_dcm(q1) @ quat_to_dcm(q2), atol=1e-12
    )


def test_overlapping_allan_deviation_uses_more_pairs() -> None:
    rng = np.random.default_rng(12)
    sample_rate = 100.0
    signal = rng.normal(size=20_000)
    requested_taus = np.array([0.1, 1.0, 10.0])

    taus_non, deviation_non = allan_deviation(
        signal, sample_rate, requested_taus
    )
    taus_over, deviation_over, pair_counts = overlapping_allan_deviation(
        signal, sample_rate, requested_taus
    )

    np.testing.assert_allclose(taus_over, taus_non)
    np.testing.assert_allclose(
        deviation_over[:2], deviation_non[:2], rtol=0.12
    )
    nonoverlap_pairs = signal.size // (taus_non * sample_rate) - 1
    assert np.all(pair_counts > nonoverlap_pairs)


def test_sampling_interval_statistics_detects_gap() -> None:
    timestamps = np.arange(100, dtype=float) * 0.01
    regular = sampling_interval_statistics(timestamps)
    assert regular["sample_rate_hz"] == pytest.approx(100.0)
    assert regular["max_relative_gap"] == pytest.approx(1.0)

    timestamps[50:] += 0.04
    with_gap = sampling_interval_statistics(timestamps)
    assert with_gap["max_relative_gap"] == pytest.approx(5.0)


def test_allan_log_slope_fit_recovers_power_law() -> None:
    taus = np.logspace(-2, 2, 40)
    deviations = 0.03 * taus**-0.5
    fit = fit_allan_log_slope(taus, deviations, 0.1, 10.0)

    assert fit["slope"] == pytest.approx(-0.5)
    assert fit["value_at_tau_1"] == pytest.approx(0.03)
    assert fit["point_count"] >= 3


def test_rotation_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="nonzero norm"):
        quat_to_dcm([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        rotvec_to_quat([0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="exactly 3"):
        coning_correct([0.1, 0.2], [0.0, 0.0, 0.0])
