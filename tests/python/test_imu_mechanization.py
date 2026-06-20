"""Tests for strapdown IMU mechanization base components."""

from __future__ import annotations

import numpy as np
import pytest

from gnss_imu import (
    IMUIncrement,
    NavigationState,
    bias_correct_increment,
    correct_two_sample_increments,
    euler_zyx_to_quat,
    normalize_quat,
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
    skew,
)


def test_navigation_state_validates_shapes_and_normalizes_quaternion() -> None:
    state = NavigationState(
        p_n=[1.0, 2.0, 3.0],
        v_n=[0.1, 0.2, 0.3],
        q_bn=[2.0, 0.0, 0.0, 0.0],
        b_a=[0.0, 0.0, 0.0],
        b_g=[0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(state.q_bn, [1.0, 0.0, 0.0, 0.0])
    assert state.p_n.shape == (3,)
    assert state.v_n.shape == (3,)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        IMUIncrement(dtheta=np.zeros(3), dvel=np.zeros(3), dt=0.0)

    with pytest.raises(ValueError, match="finite"):
        IMUIncrement(dtheta=[0.0, np.nan, 0.0], dvel=np.zeros(3), dt=0.01)

    with pytest.raises(ValueError, match="exactly 3"):
        NavigationState(
            p_n=[0.0, 0.0],
            v_n=np.zeros(3),
            q_bn=[1.0, 0.0, 0.0, 0.0],
            b_a=np.zeros(3),
            b_g=np.zeros(3),
        )

    with pytest.raises(ValueError, match="nonzero norm"):
        normalize_quat([0.0, 0.0, 0.0, 0.0])


def test_skew_matrix_matches_cross_product() -> None:
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([0.5, 4.0, -1.0])

    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_rotvec_quaternion_and_dcm_are_consistent() -> None:
    q = rotvec_to_quat([0.1, -0.2, 0.3])
    c_bn = quat_to_dcm(q)

    np.testing.assert_allclose(c_bn @ c_bn.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(c_bn), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-12)


def test_yaw_90_degrees_maps_body_x_to_ned_east() -> None:
    q_bn = euler_zyx_to_quat(roll=0.0, pitch=0.0, yaw=90.0, degrees=True)
    c_bn = quat_to_dcm(q_bn)

    body_x = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(c_bn @ body_x, [0.0, 1.0, 0.0], atol=1e-12)


def test_quaternion_composition_matches_dcm_composition() -> None:
    q1 = rotvec_to_quat([0.1, 0.0, 0.0])
    q2 = rotvec_to_quat([0.0, -0.2, 0.0])
    q12 = quat_multiply(q1, q2)

    np.testing.assert_allclose(
        quat_to_dcm(q12),
        quat_to_dcm(q1) @ quat_to_dcm(q2),
        atol=1e-12,
    )


def test_bias_correction_uses_sample_dt() -> None:
    imu = IMUIncrement(
        dtheta=[0.11, 0.22, 0.33],
        dvel=[1.1, 2.2, 3.3],
        dt=0.1,
    )

    dtheta, dvel = bias_correct_increment(
        imu,
        b_g=np.array([0.1, 0.2, 0.3]),
        b_a=np.array([1.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(dtheta, [0.10, 0.20, 0.30])
    np.testing.assert_allclose(dvel, [1.0, 2.0, 3.0])


def test_two_sample_correction_exposes_expected_cross_terms() -> None:
    imu1 = IMUIncrement(
        dtheta=[0.01, 0.0, 0.0],
        dvel=[0.0, 0.0, 0.03],
        dt=0.005,
    )
    imu2 = IMUIncrement(
        dtheta=[0.0, 0.02, 0.0],
        dvel=[0.0, 0.0, 0.04],
        dt=0.005,
    )

    correction = correct_two_sample_increments(imu1, imu2)

    np.testing.assert_allclose(correction.coning, [0.0, 0.0, 2.0e-4 * 2.0 / 3.0])
    assert np.linalg.norm(correction.sculling) > 0.0
    assert np.linalg.norm(correction.rotation) > 0.0
    np.testing.assert_allclose(correction.dt, 0.01)


def test_two_sample_correction_rejects_unequal_intervals() -> None:
    imu1 = IMUIncrement(np.zeros(3), np.zeros(3), dt=0.005)
    imu2 = IMUIncrement(np.zeros(3), np.zeros(3), dt=0.006)

    with pytest.raises(ValueError, match="requires equal sample intervals"):
        correct_two_sample_increments(imu1, imu2)
