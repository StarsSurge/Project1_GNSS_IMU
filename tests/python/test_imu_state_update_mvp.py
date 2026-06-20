"""End-to-end checks for the two-sample IMU state-update prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "python" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from demo_imu_state_update_mvp import (  # noqa: E402
    GRAVITY_NED,
    make_dataset1_measured_case,
    make_static_initial_state,
    make_static_two_sample_imu,
    propagate_two_sample_mvp,
)
from gnss_imu import IMUIncrement, quat_to_dcm  # noqa: E402


def test_static_two_sample_propagation_preserves_state() -> None:
    state0 = make_static_initial_state()
    imu1, imu2 = make_static_two_sample_imu()

    state1 = propagate_two_sample_mvp(state0, imu1, imu2)

    np.testing.assert_allclose(state1.p_n, state0.p_n, atol=1e-15)
    np.testing.assert_allclose(state1.v_n, state0.v_n, atol=1e-15)
    np.testing.assert_allclose(state1.q_bn, state0.q_bn, atol=1e-15)


def test_free_fall_adds_gravity_in_ned() -> None:
    state0 = make_static_initial_state()
    imu1 = IMUIncrement(np.zeros(3), np.zeros(3), dt=0.005)
    imu2 = IMUIncrement(np.zeros(3), np.zeros(3), dt=0.005)

    state1 = propagate_two_sample_mvp(state0, imu1, imu2)
    total_dt = imu1.dt + imu2.dt

    np.testing.assert_allclose(state1.v_n, GRAVITY_NED * total_dt)
    np.testing.assert_allclose(
        state1.p_n,
        0.5 * GRAVITY_NED * total_dt**2,
    )


def test_constant_yaw_updates_attitude_in_the_documented_direction() -> None:
    state0 = make_static_initial_state()
    gravity_cancel = np.array([0.0, 0.0, -GRAVITY_NED[2] * 0.005])
    imu1 = IMUIncrement([0.0, 0.0, np.pi / 4.0], gravity_cancel, 0.005)
    imu2 = IMUIncrement([0.0, 0.0, np.pi / 4.0], gravity_cancel, 0.005)

    state1 = propagate_two_sample_mvp(state0, imu1, imu2)

    np.testing.assert_allclose(
        quat_to_dcm(state1.q_bn) @ np.array([1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(state1.v_n, 0.0, atol=1e-12)


def test_measured_case_uses_post_initialization_increments() -> None:
    state0, imu1, imu2 = make_dataset1_measured_case()

    np.testing.assert_allclose(
        imu1.dtheta,
        [0.0000002888, -0.0000027108, -0.0000018521],
    )
    np.testing.assert_allclose(
        imu2.dtheta,
        [0.0000007072, 0.0000007140, -0.0000007045],
    )

    state1 = propagate_two_sample_mvp(state0, imu1, imu2)
    truth_velocity_at_end = np.array([0.0002, -0.0001, -0.0013])

    assert np.linalg.norm(state1.v_n - truth_velocity_at_end) < 3.0e-4
    np.testing.assert_allclose(np.linalg.norm(state1.q_bn), 1.0, atol=1e-12)
