"""Shared numerical helpers for the IMU visual explanations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from gnss_imu.imu_allan import (  # noqa: E402
    allan_deviation,
    fit_allan_log_slope,
    overlapping_allan_deviation,
    sampling_interval_statistics,
)

Array = np.ndarray


def _finite_vector(value: Array, size: int, name: str) -> Array:
    vector = np.asarray(value, dtype=float)
    if vector.size != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    vector = vector.reshape(size)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _unit_quaternion(value: Array, name: str) -> Array:
    quaternion = _finite_vector(value, 4, name)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-15:
        raise ValueError(f"{name} must have nonzero norm")
    return quaternion / norm


def skew(vector: Array) -> Array:
    """Return the 3x3 cross-product matrix of a 3-vector."""
    x, y, z = _finite_vector(vector, 3, "vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_multiply(q1: Array, q2: Array) -> Array:
    """Hamilton quaternion product for [w, x, y, z]."""
    w1, x1, y1, z1 = _finite_vector(q1, 4, "q1")
    w2, x2, y2, z2 = _finite_vector(q2, 4, "q2")
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: Array) -> Array:
    """Quaternion conjugate."""
    q = _finite_vector(q, 4, "q")
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotvec_to_quat(rotvec: Array) -> Array:
    """Convert a rotation vector in radians to a unit quaternion."""
    rotvec = _finite_vector(rotvec, 3, "rotvec")
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        scale = 0.5 - angle**2 / 48.0
        q = np.concatenate(([1.0 - angle**2 / 8.0], scale * rotvec))
    else:
        q = np.concatenate(
            ([np.cos(angle / 2.0)], rotvec * np.sin(angle / 2.0) / angle)
        )
    return q / np.linalg.norm(q)


def quat_to_rotvec(q: Array) -> Array:
    """Convert a unit quaternion to the principal rotation vector."""
    q = _unit_quaternion(q, "q")
    if q[0] < 0.0:
        q = -q
    vector_norm = np.linalg.norm(q[1:])
    if vector_norm < 1e-12:
        return 2.0 * q[1:]
    angle = 2.0 * np.arctan2(vector_norm, q[0])
    return q[1:] * angle / vector_norm


def quat_to_dcm(q: Array) -> Array:
    """Return C_bn, which maps body vectors into the navigation frame."""
    w, x, y, z = _unit_quaternion(q, "q")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def attitude_error_rotvec(q_est: Array, q_true: Array) -> Array:
    """Return the right-multiplicative body-frame error q_true^-1 * q_est."""
    q_error = quat_multiply(quat_conjugate(q_true), q_est)
    return quat_to_rotvec(q_error)


def coning_correct(dtheta1: Array, dtheta2: Array) -> Array:
    """Two-sample coning correction under linear angular-rate variation."""
    dtheta1 = _finite_vector(dtheta1, 3, "dtheta1")
    dtheta2 = _finite_vector(dtheta2, 3, "dtheta2")
    return dtheta1 + dtheta2 + (2.0 / 3.0) * np.cross(dtheta1, dtheta2)


def sculling_rotation_correct(
    dtheta1: Array,
    dvel1: Array,
    dtheta2: Array,
    dvel2: Array,
) -> tuple[Array, Array, Array]:
    """Return corrected delta-v and its sculling and rotation components."""
    dtheta1 = _finite_vector(dtheta1, 3, "dtheta1")
    dtheta2 = _finite_vector(dtheta2, 3, "dtheta2")
    dvel1 = _finite_vector(dvel1, 3, "dvel1")
    dvel2 = _finite_vector(dvel2, 3, "dvel2")
    dtheta = dtheta1 + dtheta2
    dvel = dvel1 + dvel2
    sculling = (2.0 / 3.0) * (
        np.cross(dtheta1, dvel2) + np.cross(dvel1, dtheta2)
    )
    rotation = 0.5 * np.cross(dtheta, dvel)
    return dvel + sculling + rotation, sculling, rotation
