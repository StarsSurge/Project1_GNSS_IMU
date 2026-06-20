"""Static detection, leveling, and optional gyrocompass initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gnss_imu.imu_mechanization import (
    dcm_to_quat,
    euler_zyx_to_quat,
    finite_vector,
    quat_to_dcm,
)
from gnss_imu.loosely_coupled_eskf import (
    ESKFState,
    TimedIMUIncrement,
    default_initial_covariance,
    earth_rate_ned,
    normal_gravity_mps2,
)

Array = np.ndarray


@dataclass(frozen=True)
class StaticAlignmentConfig:
    min_samples: int = 400
    min_duration_s: float = 5.0
    max_gyro_std_norm_rps: float = 2.0e-3
    max_accel_std_norm_mps2: float = 0.25
    max_gravity_norm_error_mps2: float = 0.5
    max_interval_relative_error: float = 0.02

    def __post_init__(self) -> None:
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        for name in (
            "min_duration_s",
            "max_gyro_std_norm_rps",
            "max_accel_std_norm_mps2",
            "max_gravity_norm_error_mps2",
            "max_interval_relative_error",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class StaticAlignmentDiagnostics:
    sample_count: int
    duration_s: float
    mean_gyro_rps: Array
    gyro_std_rps: Array
    mean_specific_force_mps2: Array
    accel_std_mps2: Array
    gravity_norm_error_mps2: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    yaw_source: str


@dataclass(frozen=True)
class StaticAlignmentResult:
    state: ESKFState
    diagnostics: StaticAlignmentDiagnostics


def _triad_body_to_ned(
    specific_force_b: Array,
    gyro_rate_b: Array,
    latitude_rad: float,
) -> Array:
    """Resolve body-to-NED attitude from gravity and Earth-rate vector pairs."""
    t1_b = specific_force_b / np.linalg.norm(specific_force_b)
    earth_rate_b_horizontal = np.cross(t1_b, gyro_rate_b)
    horizontal_norm = np.linalg.norm(earth_rate_b_horizontal)
    if horizontal_norm < 5e-6:
        raise ValueError(
            "gyrocompass heading is ill-conditioned: horizontal Earth-rate "
            "component is too small relative to bias/noise"
        )
    t2_b = earth_rate_b_horizontal / horizontal_norm
    t3_b = np.cross(t1_b, t2_b)

    specific_force_n = np.array([0.0, 0.0, -1.0])
    omega_ie_n = earth_rate_ned(latitude_rad)
    t1_n = specific_force_n
    t2_n = np.cross(t1_n, omega_ie_n)
    t2_n = t2_n / np.linalg.norm(t2_n)
    t3_n = np.cross(t1_n, t2_n)
    return np.column_stack((t1_n, t2_n, t3_n)) @ np.column_stack(
        (t1_b, t2_b, t3_b)
    ).T


def initialize_from_static_imu(
    samples: Sequence[TimedIMUIncrement],
    *,
    latitude_rad: float,
    longitude_rad: float,
    height_m: float,
    yaw_rad: float | None = None,
    use_gyrocompass: bool = False,
    initial_velocity_ned_mps: Array | None = None,
    accel_bias_prior_mps2: Array | None = None,
    gyro_bias_prior_rps: Array | None = None,
    estimate_gyro_bias: bool = True,
    covariance: Array | None = None,
    config: StaticAlignmentConfig | None = None,
) -> StaticAlignmentResult:
    """Initialize an ESKF state from a verified static IMU window.

    Roll and pitch come from gravity.  Yaw must come from an external heading
    or, for sufficiently stable navigation-grade gyros, static gyrocompassing.
    A stationary accelerometer cannot make yaw observable.
    """
    policy = StaticAlignmentConfig() if config is None else config
    if len(samples) < policy.min_samples:
        raise ValueError(
            f"static alignment requires at least {policy.min_samples} samples"
        )
    times = np.asarray([sample.time_s for sample in samples], dtype=float)
    intervals = np.asarray([sample.dt_s for sample in samples], dtype=float)
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("static alignment timestamps must be strictly increasing")
    median_dt = float(np.median(intervals))
    relative_interval_error = np.max(np.abs(intervals - median_dt)) / median_dt
    if relative_interval_error > policy.max_interval_relative_error:
        raise ValueError("static alignment IMU intervals are excessively irregular")
    duration = float(np.sum(intervals))
    if duration < policy.min_duration_s:
        raise ValueError(
            f"static alignment duration {duration} s is shorter than required"
        )

    gyro_rates = np.vstack(
        [sample.dtheta_rad / sample.dt_s for sample in samples]
    )
    specific_forces = np.vstack(
        [sample.dvel_mps / sample.dt_s for sample in samples]
    )
    gyro_mean = np.mean(gyro_rates, axis=0)
    accel_mean = np.mean(specific_forces, axis=0)
    gyro_std = np.std(gyro_rates, axis=0, ddof=1)
    accel_std = np.std(specific_forces, axis=0, ddof=1)
    gravity = normal_gravity_mps2(latitude_rad, height_m)
    gravity_error = abs(np.linalg.norm(accel_mean) - gravity)
    if np.linalg.norm(gyro_std) > policy.max_gyro_std_norm_rps:
        raise ValueError("IMU window is not static: gyro variation exceeds threshold")
    if np.linalg.norm(accel_std) > policy.max_accel_std_norm_mps2:
        raise ValueError("IMU window is not static: accelerometer variation exceeds threshold")
    if gravity_error > policy.max_gravity_norm_error_mps2:
        raise ValueError("IMU window is not static: mean specific-force norm is not gravity")

    accel_bias_prior = (
        np.zeros(3)
        if accel_bias_prior_mps2 is None
        else finite_vector(accel_bias_prior_mps2, 3, "accel_bias_prior_mps2")
    )
    gyro_bias_prior = (
        np.zeros(3)
        if gyro_bias_prior_rps is None
        else finite_vector(gyro_bias_prior_rps, 3, "gyro_bias_prior_rps")
    )
    force_for_leveling = accel_mean - accel_bias_prior
    force_norm = np.linalg.norm(force_for_leveling)
    roll = np.arctan2(-force_for_leveling[1], -force_for_leveling[2])
    pitch = np.arcsin(np.clip(force_for_leveling[0] / force_norm, -1.0, 1.0))

    if yaw_rad is not None and use_gyrocompass:
        raise ValueError("choose either external yaw or gyrocompass, not both")
    if yaw_rad is not None:
        yaw = float(yaw_rad)
        if not np.isfinite(yaw):
            raise ValueError("yaw_rad must be finite")
        q_bn = euler_zyx_to_quat(roll, pitch, yaw)
        yaw_source = "external"
    elif use_gyrocompass:
        c_bn = _triad_body_to_ned(
            force_for_leveling,
            gyro_mean - gyro_bias_prior,
            latitude_rad,
        )
        q_bn = dcm_to_quat(c_bn)
        yaw = float(np.arctan2(c_bn[1, 0], c_bn[0, 0]))
        roll = float(np.arctan2(c_bn[2, 1], c_bn[2, 2]))
        pitch = float(np.arcsin(np.clip(-c_bn[2, 0], -1.0, 1.0)))
        yaw_source = "gyrocompass"
    else:
        raise ValueError(
            "yaw is unobservable from a static accelerometer; provide yaw_rad "
            "or enable gyrocompass for a suitable navigation-grade gyro"
        )

    c_bn = quat_to_dcm(q_bn)
    expected_gyro_b = c_bn.T @ earth_rate_ned(latitude_rad)
    gyro_bias = (
        gyro_mean - expected_gyro_b if estimate_gyro_bias else gyro_bias_prior
    )
    velocity = (
        np.zeros(3)
        if initial_velocity_ned_mps is None
        else finite_vector(initial_velocity_ned_mps, 3, "initial_velocity_ned_mps")
    )
    state = ESKFState(
        time_s=float(times[-1]),
        latitude_rad=latitude_rad,
        longitude_rad=longitude_rad,
        height_m=height_m,
        velocity_ned_mps=velocity,
        q_bn=q_bn,
        accel_bias_mps2=accel_bias_prior,
        gyro_bias_rps=gyro_bias,
        covariance=(default_initial_covariance() if covariance is None else covariance),
    )
    diagnostics = StaticAlignmentDiagnostics(
        sample_count=len(samples),
        duration_s=duration,
        mean_gyro_rps=gyro_mean,
        gyro_std_rps=gyro_std,
        mean_specific_force_mps2=accel_mean,
        accel_std_mps2=accel_std,
        gravity_norm_error_mps2=float(gravity_error),
        roll_deg=float(np.rad2deg(roll)),
        pitch_deg=float(np.rad2deg(pitch)),
        yaw_deg=float(np.rad2deg(yaw) % 360.0),
        yaw_source=yaw_source,
    )
    return StaticAlignmentResult(state=state, diagnostics=diagnostics)
