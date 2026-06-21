"""Dataset loading and coordinate helpers for GNSS/IMU visualization.

The helpers in this module intentionally avoid heavy geodesy dependencies.
They implement the small WGS-84 subset needed to inspect the KF-GINS style
dataset before building a GNSS/INS estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class RtkData:
    """GNSS RTK position observations in geodetic coordinates."""

    time_s: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    height_m: np.ndarray
    std_ned_m: np.ndarray


@dataclass(frozen=True)
class ImuIncrementData:
    """IMU increment measurements.

    The KF-GINS text format stores one timestamp followed by delta-angle
    increments in rad and delta-velocity increments in m/s.
    """

    time_s: np.ndarray
    delta_angle_rad: np.ndarray
    delta_velocity_mps: np.ndarray


@dataclass(frozen=True)
class TruthNavData:
    """Reference navigation solution."""

    gps_week: np.ndarray
    time_s: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    height_m: np.ndarray
    velocity_ned_mps: np.ndarray
    attitude_rpy_deg: np.ndarray


@dataclass(frozen=True)
class Dataset1:
    """Container for the dataset1 RTK, IMU, and truth files."""

    rtk: RtkData
    imu: ImuIncrementData
    truth: TruthNavData


@dataclass(frozen=True)
class LeverTimeCalibrationResult:
    """Robust joint lever-arm and first-order time-offset calibration."""

    lever_arm_b_m: np.ndarray
    time_offset_s: float
    covariance: np.ndarray
    correlation: np.ndarray
    parameter_std: np.ndarray
    predicted_residual_ned_m: np.ndarray
    corrected_residual_ned_m: np.ndarray
    robust_epoch_weights: np.ndarray
    raw_rms_ned_m: np.ndarray
    corrected_rms_ned_m: np.ndarray
    condition_number: float
    iteration_count: int
    downweighted_epoch_count: int


@dataclass(frozen=True)
class FixedLeverTimeCalibrationResult:
    """Robust time-offset calibration with a fixed independent lever arm."""

    lever_arm_b_m: np.ndarray
    time_offset_s: float
    time_offset_std_s: float
    predicted_residual_ned_m: np.ndarray
    corrected_residual_ned_m: np.ndarray
    robust_epoch_weights: np.ndarray
    raw_rms_ned_m: np.ndarray
    corrected_rms_ned_m: np.ndarray
    iteration_count: int
    downweighted_epoch_count: int
    time_information: float


@dataclass(frozen=True)
class ResidualCorrelationDiagnostics:
    """Per-axis autocorrelation and effective independent sample count."""

    autocorrelation: np.ndarray
    integrated_autocorrelation_time: np.ndarray
    effective_sample_size: np.ndarray
    lag1_autocorrelation: np.ndarray
    max_lag: int


@dataclass(frozen=True)
class FrozenTimeOffsetScore:
    """Time-offset score evaluated against a trajectory that is never updated."""

    offset_s: float
    robust_mean_nis: float
    median_nis: float
    measurement_count: int


@dataclass(frozen=True)
class FrozenTimeOffsetBootstrapResult:
    """Circular moving-block bootstrap interval for a frozen time profile."""

    point_estimate_s: float
    lower_offset_s: float
    upper_offset_s: float
    bootstrap_offsets_s: np.ndarray
    block_length_epochs: int
    replicate_count: int
    confidence_level: float
    boundary_hit_fraction: float
    grid_resolution_limited: bool


def _as_2d_numeric_table(path: Path, expected_columns: int) -> np.ndarray:
    table = np.loadtxt(path, dtype=float)
    if table.ndim != 2 or table.shape[1] != expected_columns:
        raise ValueError(
            f"{path} must have shape (N, {expected_columns}); got "
            f"{table.shape}"
        )
    if table.shape[0] < 2:
        raise ValueError(f"{path} must contain at least two rows")
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{path} contains non-finite values")
    return table


def load_rtk(path: Path) -> RtkData:
    """Load ``GNSS-RTK.txt`` with columns time, lat, lon, height, std N/E/D."""

    table = _as_2d_numeric_table(path, 7)
    return RtkData(
        time_s=table[:, 0],
        latitude_deg=table[:, 1],
        longitude_deg=table[:, 2],
        height_m=table[:, 3],
        std_ned_m=table[:, 4:7],
    )


def load_imu_increments(path: Path) -> ImuIncrementData:
    """Load ``Leador-A15.txt`` with delta-angle and delta-velocity columns."""

    table = _as_2d_numeric_table(path, 7)
    return ImuIncrementData(
        time_s=table[:, 0],
        delta_angle_rad=table[:, 1:4],
        delta_velocity_mps=table[:, 4:7],
    )


def load_truth_nav(path: Path) -> TruthNavData:
    """Load ``truth.nav`` reference states."""

    table = _as_2d_numeric_table(path, 11)
    return TruthNavData(
        gps_week=table[:, 0],
        time_s=table[:, 1],
        latitude_deg=table[:, 2],
        longitude_deg=table[:, 3],
        height_m=table[:, 4],
        velocity_ned_mps=table[:, 5:8],
        attitude_rpy_deg=table[:, 8:11],
    )


def load_dataset1(dataset_dir: Path) -> Dataset1:
    """Load the repository's ``data/dataset1`` files."""

    return Dataset1(
        rtk=load_rtk(dataset_dir / "GNSS-RTK.txt"),
        imu=load_imu_increments(dataset_dir / "Leador-A15.txt"),
        truth=load_truth_nav(dataset_dir / "truth.nav"),
    )


def geodetic_to_ecef(
    latitude_deg: np.ndarray | float,
    longitude_deg: np.ndarray | float,
    height_m: np.ndarray | float,
) -> np.ndarray:
    """Convert WGS-84 geodetic coordinates to ECEF meters.

    Returns an array whose final dimension is ``[x, y, z]``.
    """

    lat = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=float))
    h = np.asarray(height_m, dtype=float)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    prime_vertical_radius = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    x = (prime_vertical_radius + h) * cos_lat * np.cos(lon)
    y = (prime_vertical_radius + h) * cos_lat * np.sin(lon)
    z = (prime_vertical_radius * (1.0 - WGS84_E2) + h) * sin_lat
    return np.stack((x, y, z), axis=-1)


def ecef_delta_to_ned(
    ecef_delta_m: np.ndarray,
    reference_latitude_deg: float,
    reference_longitude_deg: float,
) -> np.ndarray:
    """Rotate ECEF position deltas into local North-East-Down coordinates."""

    delta = np.asarray(ecef_delta_m, dtype=float)
    if delta.shape[-1] != 3:
        raise ValueError("ecef_delta_m must have final dimension 3")

    lat = np.deg2rad(reference_latitude_deg)
    lon = np.deg2rad(reference_longitude_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    rotation = np.array(
        [
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0.0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
        ]
    )
    return delta @ rotation.T


def geodetic_to_ned(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    height_m: np.ndarray,
    reference_llh: tuple[float, float, float],
) -> np.ndarray:
    """Convert geodetic coordinates to local NED meters about a reference LLH."""

    ref_lat, ref_lon, ref_h = reference_llh
    ecef = geodetic_to_ecef(latitude_deg, longitude_deg, height_m)
    ref_ecef = geodetic_to_ecef(ref_lat, ref_lon, ref_h)
    return ecef_delta_to_ned(ecef - ref_ecef, ref_lat, ref_lon)


def relative_time_seconds(time_s: np.ndarray, origin_s: float | None = None) -> np.ndarray:
    """Return timestamps shifted to a local origin for plotting."""

    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time_s must be a one-dimensional array with >=2 rows")
    if not np.all(np.isfinite(t)):
        raise ValueError("time_s contains non-finite values")
    base = float(t[0] if origin_s is None else origin_s)
    return t - base


def sampling_summary(time_s: np.ndarray) -> dict[str, float]:
    """Compute basic timestamp statistics for a sensor stream."""

    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time_s must be a one-dimensional array with >=2 rows")
    dt = np.diff(t)
    if not np.all(dt > 0.0):
        raise ValueError("timestamps must be strictly increasing")
    median_dt = float(np.median(dt))
    return {
        "count": float(t.size),
        "start_s": float(t[0]),
        "end_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "median_dt_s": median_dt,
        "sample_rate_hz": float(1.0 / median_dt),
        "min_dt_s": float(np.min(dt)),
        "max_dt_s": float(np.max(dt)),
    }


def increments_to_rates(
    time_s: np.ndarray,
    increments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-sample increments to rates using backward time intervals."""

    t = np.asarray(time_s, dtype=float)
    x = np.asarray(increments, dtype=float)
    if t.ndim != 1 or x.ndim != 2 or x.shape[0] != t.size:
        raise ValueError("increments must have shape (len(time_s), axis_count)")
    dt = np.diff(t)
    if not np.all(dt > 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return t[1:], x[1:] / dt[:, None]


def decimation_indices(length: int, max_points: int) -> np.ndarray:
    """Return deterministic indices that keep plots readable for large logs."""

    if length <= 0:
        raise ValueError("length must be positive")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def interpolate_columns(
    source_time_s: np.ndarray,
    source_values: np.ndarray,
    query_time_s: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate each column of a time series."""

    source_t = np.asarray(source_time_s, dtype=float)
    query_t = np.asarray(query_time_s, dtype=float)
    values = np.asarray(source_values, dtype=float)
    if values.ndim != 2 or values.shape[0] != source_t.size:
        raise ValueError("source_values must have shape (len(source_time_s), C)")
    columns = [
        np.interp(query_t, source_t, values[:, axis])
        for axis in range(values.shape[1])
    ]
    return np.column_stack(columns)


def rpy_deg_to_body_to_ned(
    roll_deg: np.ndarray,
    pitch_deg: np.ndarray,
    yaw_deg: np.ndarray,
) -> np.ndarray:
    """Convert roll-pitch-yaw angles to body-to-NED direction cosine matrices.

    The convention matches the dataset README: body axes are forward-right-down,
    navigation axes are north-east-down, and yaw/heading is positive clockwise
    from north. The returned shape is ``(..., 3, 3)``.
    """

    roll = np.deg2rad(np.asarray(roll_deg, dtype=float))
    pitch = np.deg2rad(np.asarray(pitch_deg, dtype=float))
    yaw = np.deg2rad(np.asarray(yaw_deg, dtype=float))
    roll, pitch, yaw = np.broadcast_arrays(roll, pitch, yaw)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    matrices = np.empty(roll.shape + (3, 3), dtype=float)
    matrices[..., 0, 0] = cy * cp
    matrices[..., 0, 1] = -sy * cr + cy * sp * sr
    matrices[..., 0, 2] = cy * sp * cr + sy * sr
    matrices[..., 1, 0] = sy * cp
    matrices[..., 1, 1] = cy * cr + sy * sp * sr
    matrices[..., 1, 2] = sy * sp * cr - cy * sr
    matrices[..., 2, 0] = -sp
    matrices[..., 2, 1] = cp * sr
    matrices[..., 2, 2] = cp * cr
    return matrices


def fit_body_lever_arm_from_residuals(
    body_to_ned: np.ndarray,
    residual_ned_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a fixed body-frame lever arm that explains NED residuals.

    This solves ``residual_ned ~= C_n_b * lever_b`` in least squares. It is a
    diagnostic tool for checking whether GNSS antenna positions and truth IMU
    positions refer to different physical points.
    """

    rotations = np.asarray(body_to_ned, dtype=float)
    residuals = np.asarray(residual_ned_m, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("body_to_ned must have shape (N, 3, 3)")
    if residuals.shape != (rotations.shape[0], 3):
        raise ValueError("residual_ned_m must have shape (N, 3)")

    design = rotations.reshape(-1, 3)
    lever_arm_b_m = np.linalg.lstsq(design, residuals.reshape(-1), rcond=None)[0]
    predicted_ned_m = np.einsum("nij,j->ni", rotations, lever_arm_b_m)
    corrected_residual_ned_m = residuals - predicted_ned_m
    return lever_arm_b_m, predicted_ned_m, corrected_residual_ned_m


def antenna_velocity_ned(
    body_to_ned: np.ndarray,
    imu_velocity_ned_mps: np.ndarray,
    angular_rate_b_rps: np.ndarray,
    lever_arm_b_m: np.ndarray,
) -> np.ndarray:
    """Return antenna velocity using ``v_a = v_i + C_bn(omega x lever)``."""
    rotations = np.asarray(body_to_ned, dtype=float)
    velocities = np.asarray(imu_velocity_ned_mps, dtype=float)
    angular_rates = np.asarray(angular_rate_b_rps, dtype=float)
    lever = np.asarray(lever_arm_b_m, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("body_to_ned must have shape (N, 3, 3)")
    sample_count = rotations.shape[0]
    if velocities.shape != (sample_count, 3):
        raise ValueError("imu_velocity_ned_mps must have shape (N, 3)")
    if angular_rates.shape != (sample_count, 3):
        raise ValueError("angular_rate_b_rps must have shape (N, 3)")
    if lever.shape != (3,):
        raise ValueError("lever_arm_b_m must have shape (3,)")
    if not (
        np.all(np.isfinite(rotations))
        and np.all(np.isfinite(velocities))
        and np.all(np.isfinite(angular_rates))
        and np.all(np.isfinite(lever))
    ):
        raise ValueError("antenna velocity inputs must be finite")
    orthogonality_error = np.max(
        np.linalg.norm(
            np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3),
            axis=(1, 2),
        )
    )
    if orthogonality_error > 1e-6 or np.max(
        np.abs(np.linalg.det(rotations) - 1.0)
    ) > 1e-6:
        raise ValueError("body_to_ned must contain proper rotation matrices")
    rotational_velocity_b = np.cross(angular_rates, lever)
    return velocities + np.einsum(
        "nij,nj->ni",
        rotations,
        rotational_velocity_b,
    )


def residual_correlation_diagnostics(
    residual_ned_m: np.ndarray,
    *,
    max_lag: int | None = None,
) -> ResidualCorrelationDiagnostics:
    """Estimate residual autocorrelation and effective sample size per axis.

    The integrated autocorrelation time uses the initial positive sequence:
    summation stops at the first non-positive lag correlation.
    """
    residuals = np.asarray(residual_ned_m, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != 3:
        raise ValueError("residual_ned_m must have shape (N, 3)")
    sample_count = residuals.shape[0]
    if sample_count < 3 or not np.all(np.isfinite(residuals)):
        raise ValueError("residuals require at least three finite epochs")
    selected_max_lag = (
        min(200, sample_count - 1)
        if max_lag is None
        else int(max_lag)
    )
    if selected_max_lag < 1 or selected_max_lag >= sample_count:
        raise ValueError("max_lag must be in [1, sample_count - 1]")

    centered = residuals - np.mean(residuals, axis=0)
    autocorrelation = np.empty((selected_max_lag + 1, 3))
    autocorrelation[0] = 1.0
    integrated_time = np.ones(3)
    for axis in range(3):
        variance_sum = float(centered[:, axis] @ centered[:, axis])
        if variance_sum <= np.finfo(float).eps:
            raise ValueError("residual axis variance is too small for correlation")
        for lag in range(1, selected_max_lag + 1):
            autocorrelation[lag, axis] = float(
                centered[:-lag, axis] @ centered[lag:, axis] / variance_sum
            )
        positive_sum = 0.0
        for lag in range(1, selected_max_lag + 1):
            rho = autocorrelation[lag, axis]
            if rho <= 0.0:
                break
            positive_sum += rho
        integrated_time[axis] = max(1.0, 1.0 + 2.0 * positive_sum)
    effective_sample_size = sample_count / integrated_time
    return ResidualCorrelationDiagnostics(
        autocorrelation=autocorrelation,
        integrated_autocorrelation_time=integrated_time,
        effective_sample_size=effective_sample_size,
        lag1_autocorrelation=autocorrelation[1].copy(),
        max_lag=selected_max_lag,
    )


def score_frozen_trajectory_time_offsets(
    trajectory_time_s: np.ndarray,
    antenna_position_ned_m: np.ndarray,
    measurement_time_s: np.ndarray,
    measured_position_ned_m: np.ndarray,
    std_ned_m: np.ndarray,
    candidate_offsets_s: np.ndarray,
    *,
    nis_clip: float = 100.0,
) -> tuple[FrozenTimeOffsetScore, ...]:
    """Score time offsets without updating or re-estimating the trajectory."""
    trajectory_time = np.asarray(trajectory_time_s, dtype=float)
    antenna_position = np.asarray(antenna_position_ned_m, dtype=float)
    measurement_time = np.asarray(measurement_time_s, dtype=float)
    measured_position = np.asarray(measured_position_ned_m, dtype=float)
    standard_deviation = np.asarray(std_ned_m, dtype=float)
    candidates = np.asarray(candidate_offsets_s, dtype=float)
    if trajectory_time.ndim != 1 or trajectory_time.size < 2:
        raise ValueError("trajectory_time_s must contain at least two epochs")
    if antenna_position.shape != (trajectory_time.size, 3):
        raise ValueError("antenna_position_ned_m must have shape (trajectory N, 3)")
    if measurement_time.ndim != 1 or measurement_time.size < 3:
        raise ValueError("measurement_time_s must contain at least three epochs")
    if measured_position.shape != (measurement_time.size, 3):
        raise ValueError("measured_position_ned_m must have shape (measurement N, 3)")
    if standard_deviation.shape == (3,):
        standard_deviation = np.broadcast_to(
            standard_deviation,
            (measurement_time.size, 3),
        ).copy()
    if standard_deviation.shape != (measurement_time.size, 3):
        raise ValueError("std_ned_m must have shape (3,) or (measurement N, 3)")
    if candidates.ndim != 1 or candidates.size < 2:
        raise ValueError("candidate_offsets_s must contain at least two values")
    if not (
        np.all(np.isfinite(trajectory_time))
        and np.all(np.isfinite(antenna_position))
        and np.all(np.isfinite(measurement_time))
        and np.all(np.isfinite(measured_position))
        and np.all(np.isfinite(standard_deviation))
        and np.all(np.isfinite(candidates))
    ):
        raise ValueError("frozen trajectory inputs must be finite")
    if np.any(np.diff(trajectory_time) <= 0.0) or np.any(
        np.diff(measurement_time) <= 0.0
    ):
        raise ValueError("trajectory and measurement timestamps must increase")
    if np.any(standard_deviation <= 0.0):
        raise ValueError("std_ned_m must be positive")
    if not np.isfinite(nis_clip) or nis_clip <= 0.0:
        raise ValueError("nis_clip must be positive and finite")

    scores = []
    for offset in candidates:
        effective_time = measurement_time + float(offset)
        if (
            effective_time[0] < trajectory_time[0]
            or effective_time[-1] > trajectory_time[-1]
        ):
            raise ValueError("candidate effective time is outside frozen trajectory")
        predicted = interpolate_columns(
            trajectory_time,
            antenna_position,
            effective_time,
        )
        normalized_residual = (measured_position - predicted) / standard_deviation
        nis = np.sum(normalized_residual**2, axis=1)
        scores.append(
            FrozenTimeOffsetScore(
                offset_s=float(offset),
                robust_mean_nis=float(np.mean(np.minimum(nis, nis_clip))),
                median_nis=float(np.median(nis)),
                measurement_count=int(nis.size),
            )
        )
    return tuple(scores)


def frozen_trajectory_nis_matrix(
    trajectory_time_s: np.ndarray,
    antenna_position_ned_m: np.ndarray,
    measurement_time_s: np.ndarray,
    measured_position_ned_m: np.ndarray,
    std_ned_m: np.ndarray,
    candidate_offsets_s: np.ndarray,
) -> np.ndarray:
    """Return per-candidate, per-epoch frozen-trajectory position NIS."""
    # 先复用公开评分器完成形状、时间覆盖、有限值和标准差检查。
    score_frozen_trajectory_time_offsets(
        trajectory_time_s,
        antenna_position_ned_m,
        measurement_time_s,
        measured_position_ned_m,
        std_ned_m,
        candidate_offsets_s,
    )
    trajectory_time = np.asarray(trajectory_time_s, dtype=float)
    antenna_position = np.asarray(antenna_position_ned_m, dtype=float)
    measurement_time = np.asarray(measurement_time_s, dtype=float)
    measured_position = np.asarray(measured_position_ned_m, dtype=float)
    standard_deviation = np.asarray(std_ned_m, dtype=float)
    if standard_deviation.shape == (3,):
        standard_deviation = np.broadcast_to(
            standard_deviation,
            measured_position.shape,
        )
    candidates = np.asarray(candidate_offsets_s, dtype=float)
    nis_matrix = np.empty((candidates.size, measurement_time.size))
    for candidate_index, offset in enumerate(candidates):
        predicted = interpolate_columns(
            trajectory_time,
            antenna_position,
            measurement_time + float(offset),
        )
        normalized_residual = (measured_position - predicted) / standard_deviation
        nis_matrix[candidate_index] = np.sum(normalized_residual**2, axis=1)
    return nis_matrix


def bootstrap_frozen_time_offset(
    candidate_offsets_s: np.ndarray,
    epoch_nis: np.ndarray,
    *,
    block_length_epochs: int,
    replicate_count: int = 2000,
    confidence_level: float = 0.95,
    nis_clip: float = 100.0,
    random_seed: int = 0,
) -> FrozenTimeOffsetBootstrapResult:
    """Bootstrap a frozen-profile minimum with circular moving epoch blocks."""
    candidates = np.asarray(candidate_offsets_s, dtype=float)
    nis_matrix = np.asarray(epoch_nis, dtype=float)
    if candidates.ndim != 1 or candidates.size < 3:
        raise ValueError("candidate_offsets_s must contain at least three values")
    if np.any(np.diff(candidates) <= 0.0):
        raise ValueError("candidate offsets must be strictly increasing")
    if nis_matrix.ndim != 2 or nis_matrix.shape[0] != candidates.size:
        raise ValueError("epoch_nis must have shape (candidate_count, epoch_count)")
    epoch_count = nis_matrix.shape[1]
    if epoch_count < 3 or not np.all(np.isfinite(nis_matrix)):
        raise ValueError("epoch_nis requires at least three finite epochs")
    if np.any(nis_matrix < 0.0):
        raise ValueError("epoch_nis must be nonnegative")
    block_length = int(block_length_epochs)
    if block_length < 1 or block_length > epoch_count:
        raise ValueError("block_length_epochs must be in [1, epoch_count]")
    if replicate_count < 100:
        raise ValueError("replicate_count must be at least 100")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not np.isfinite(nis_clip) or nis_clip <= 0.0:
        raise ValueError("nis_clip must be positive and finite")

    clipped = np.minimum(nis_matrix, nis_clip)

    def refined_minimum(objective: np.ndarray) -> tuple[float, bool]:
        best_index = int(np.argmin(objective))
        refined = float(candidates[best_index])
        boundary = best_index == 0 or best_index == candidates.size - 1
        if not boundary:
            local_x = candidates[best_index - 1 : best_index + 2]
            local_y = objective[best_index - 1 : best_index + 2]
            quadratic = np.polyfit(local_x, local_y, 2)
            if quadratic[0] > 0.0:
                vertex = float(-quadratic[1] / (2.0 * quadratic[0]))
                if local_x[0] <= vertex <= local_x[-1]:
                    refined = vertex
        return refined, boundary

    point_estimate, _ = refined_minimum(np.mean(clipped, axis=1))
    rng = np.random.default_rng(random_seed)
    block_count = int(np.ceil(epoch_count / block_length))
    bootstrap_offsets = np.empty(replicate_count)
    boundary_hits = 0
    within_block = np.arange(block_length)
    for replicate in range(replicate_count):
        starts = rng.integers(0, epoch_count, size=block_count)
        indices = (starts[:, None] + within_block[None, :]) % epoch_count
        indices = indices.reshape(-1)[:epoch_count]
        estimate, boundary = refined_minimum(np.mean(clipped[:, indices], axis=1))
        bootstrap_offsets[replicate] = estimate
        boundary_hits += int(boundary)

    tail_probability = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(
        bootstrap_offsets,
        [tail_probability, 1.0 - tail_probability],
    )
    local_index = int(np.argmin(np.abs(candidates - point_estimate)))
    neighbor_steps = []
    if local_index > 0:
        neighbor_steps.append(candidates[local_index] - candidates[local_index - 1])
    if local_index + 1 < candidates.size:
        neighbor_steps.append(candidates[local_index + 1] - candidates[local_index])
    grid_half_step = 0.5 * min(neighbor_steps)
    conservative_lower = point_estimate - grid_half_step
    conservative_upper = point_estimate + grid_half_step
    resolution_limited = lower > conservative_lower or upper < conservative_upper
    lower = min(float(lower), conservative_lower)
    upper = max(float(upper), conservative_upper)
    return FrozenTimeOffsetBootstrapResult(
        point_estimate_s=point_estimate,
        lower_offset_s=lower,
        upper_offset_s=upper,
        bootstrap_offsets_s=bootstrap_offsets,
        block_length_epochs=block_length,
        replicate_count=int(replicate_count),
        confidence_level=float(confidence_level),
        boundary_hit_fraction=float(boundary_hits / replicate_count),
        grid_resolution_limited=resolution_limited,
    )


def calibrate_lever_arm_and_time_offset(
    body_to_ned: np.ndarray,
    velocity_ned_mps: np.ndarray,
    residual_ned_m: np.ndarray,
    std_ned_m: np.ndarray | None = None,
    *,
    angular_rate_b_rps: np.ndarray | None = None,
    lever_prior_b_m: np.ndarray | None = None,
    lever_prior_std_m: np.ndarray | None = None,
    huber_threshold: float = 2.5,
    max_iterations: int = 20,
    convergence_tolerance: float = 1e-10,
    max_condition_number: float = 1e10,
) -> LeverTimeCalibrationResult:
    """Jointly fit body lever arm and a constant timestamp offset.

    The first-order observation model is
    ``residual_n = C_bn @ lever_b + velocity_n * time_offset``.
    The time-offset sign follows ``effective = reported + offset``.
    """
    rotations = np.asarray(body_to_ned, dtype=float)
    velocities = np.asarray(velocity_ned_mps, dtype=float)
    residuals = np.asarray(residual_ned_m, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("body_to_ned must have shape (N, 3, 3)")
    sample_count = rotations.shape[0]
    if sample_count < 4:
        raise ValueError("joint lever/time calibration requires at least four epochs")
    if velocities.shape != (sample_count, 3):
        raise ValueError("velocity_ned_mps must have shape (N, 3)")
    if residuals.shape != (sample_count, 3):
        raise ValueError("residual_ned_m must have shape (N, 3)")
    if not (
        np.all(np.isfinite(rotations))
        and np.all(np.isfinite(velocities))
        and np.all(np.isfinite(residuals))
    ):
        raise ValueError("joint calibration inputs must be finite")
    angular_rates = (
        np.zeros((sample_count, 3))
        if angular_rate_b_rps is None
        else np.asarray(angular_rate_b_rps, dtype=float)
    )
    if angular_rates.shape != (sample_count, 3) or not np.all(
        np.isfinite(angular_rates)
    ):
        raise ValueError("angular_rate_b_rps must have shape (N, 3) and be finite")
    orthogonality_error = np.max(
        np.linalg.norm(
            np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3),
            axis=(1, 2),
        )
    )
    determinants = np.linalg.det(rotations)
    if orthogonality_error > 1e-6 or np.max(np.abs(determinants - 1.0)) > 1e-6:
        raise ValueError("body_to_ned must contain proper rotation matrices")

    reported_std_is_known = std_ned_m is not None
    if std_ned_m is None:
        standard_deviation = np.ones((sample_count, 3))
    else:
        standard_deviation = np.asarray(std_ned_m, dtype=float)
        if standard_deviation.shape == (3,):
            standard_deviation = np.broadcast_to(
                standard_deviation,
                (sample_count, 3),
            ).copy()
        if standard_deviation.shape != (sample_count, 3):
            raise ValueError("std_ned_m must have shape (3,) or (N, 3)")
        if not np.all(np.isfinite(standard_deviation)) or np.any(
            standard_deviation <= 0.0
        ):
            raise ValueError("std_ned_m must be positive and finite")
    if not np.isfinite(huber_threshold) or huber_threshold <= 0.0:
        raise ValueError("huber_threshold must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive and finite")
    if not np.isfinite(max_condition_number) or max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be finite and greater than one")
    if (lever_prior_b_m is None) != (lever_prior_std_m is None):
        raise ValueError("lever prior mean and standard deviation must be provided together")
    prior_design = None
    if lever_prior_b_m is not None:
        prior_mean = np.asarray(lever_prior_b_m, dtype=float)
        prior_std = np.asarray(lever_prior_std_m, dtype=float)
        if prior_mean.shape != (3,) or not np.all(np.isfinite(prior_mean)):
            raise ValueError("lever_prior_b_m must be a finite 3-vector")
        if (
            prior_std.shape != (3,)
            or not np.all(np.isfinite(prior_std))
            or np.any(prior_std <= 0.0)
        ):
            raise ValueError("lever_prior_std_m must be a positive finite 3-vector")
        # MAP 伪观测：[I_3, 0] * [lever, time] = measured_lever。
        prior_design = np.column_stack([np.eye(3), np.zeros(3)]) / prior_std[:, None]

    observations = residuals.reshape(-1)
    inverse_std = 1.0 / standard_deviation.reshape(-1)
    epoch_weights = np.ones(sample_count)
    parameters = np.zeros(4)
    if lever_prior_b_m is not None:
        parameters[:3] = prior_mean

    omega_skew = np.zeros((sample_count, 3, 3))
    omega_skew[:, 0, 1] = -angular_rates[:, 2]
    omega_skew[:, 0, 2] = angular_rates[:, 1]
    omega_skew[:, 1, 0] = angular_rates[:, 2]
    omega_skew[:, 1, 2] = -angular_rates[:, 0]
    omega_skew[:, 2, 0] = -angular_rates[:, 1]
    omega_skew[:, 2, 1] = angular_rates[:, 0]

    def predict_and_jacobian(parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lever = parameter[:3]
        time_offset = parameter[3]
        antenna_velocity_n = antenna_velocity_ned(
            rotations,
            velocities,
            angular_rates,
            lever,
        )
        prediction = (
            np.einsum("nij,j->ni", rotations, lever)
            + antenna_velocity_n * time_offset
        )
        lever_jacobian = rotations + time_offset * np.einsum(
            "nij,njk->nik",
            rotations,
            omega_skew,
        )
        jacobian = np.concatenate(
            [lever_jacobian, antenna_velocity_n[:, :, np.newaxis]],
            axis=2,
        ).reshape(-1, 4)
        return prediction, jacobian

    iteration_count = 0
    for iteration_count in range(1, max_iterations + 1):
        prediction, design = predict_and_jacobian(parameters)
        linearized_residual = observations - prediction.reshape(-1)
        component_weights = np.repeat(epoch_weights, 3)
        whitening = inverse_std * np.sqrt(component_weights)
        weighted_design = design * whitening[:, np.newaxis]
        weighted_observations = linearized_residual * whitening
        solve_design = weighted_design
        solve_observations = weighted_observations
        if prior_design is not None:
            solve_design = np.vstack([solve_design, prior_design])
            solve_observations = np.concatenate(
                [solve_observations, (prior_mean - parameters[:3]) / prior_std]
            )
        singular_values = np.linalg.svd(solve_design, compute_uv=False)
        if singular_values[-1] <= np.finfo(float).eps * singular_values[0]:
            raise ValueError(
                "lever arm and time offset are unobservable for this motion"
            )
        condition_number = float(singular_values[0] / singular_values[-1])
        if condition_number > max_condition_number:
            raise ValueError(
                "lever/time calibration is ill-conditioned; motion excitation "
                f"condition number is {condition_number:.3e}"
            )
        delta, _, rank, _ = np.linalg.lstsq(
            solve_design,
            solve_observations,
            rcond=None,
        )
        if rank != 4:
            raise ValueError("lever arm and time offset are not jointly observable")
        updated = parameters + delta
        updated_prediction, _ = predict_and_jacobian(updated)
        normalized_epoch_residual = (
            residuals - updated_prediction
        ) / standard_deviation
        mahalanobis_norm = np.linalg.norm(normalized_epoch_residual, axis=1)
        updated_epoch_weights = np.ones(sample_count)
        outliers = mahalanobis_norm > huber_threshold
        updated_epoch_weights[outliers] = (
            huber_threshold / mahalanobis_norm[outliers]
        )
        parameter_change = np.linalg.norm(delta)
        parameter_scale = max(1.0, np.linalg.norm(updated))
        parameters = updated
        epoch_weights = updated_epoch_weights
        if parameter_change <= convergence_tolerance * parameter_scale:
            break

    component_weights = np.repeat(epoch_weights, 3)
    whitening = inverse_std * np.sqrt(component_weights)
    predicted, design = predict_and_jacobian(parameters)
    weighted_design = design * whitening[:, np.newaxis]
    posterior_design = (
        weighted_design
        if prior_design is None
        else np.vstack([weighted_design, prior_design])
    )
    normal_matrix = posterior_design.T @ posterior_design
    singular_values = np.linalg.svd(posterior_design, compute_uv=False)
    condition_number = float(singular_values[0] / singular_values[-1])
    corrected = residuals - predicted
    normalized_components = corrected.reshape(-1) * inverse_std
    weighted_rss = float(
        np.sum(component_weights * normalized_components**2)
    )
    degrees_of_freedom = 3 * sample_count - 4
    reduced_chi_square = weighted_rss / degrees_of_freedom
    covariance_scale = (
        max(1.0, reduced_chi_square)
        if reported_std_is_known
        else max(np.finfo(float).eps, reduced_chi_square)
    )
    covariance = np.linalg.inv(normal_matrix) * covariance_scale
    parameter_std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(parameter_std, parameter_std)
    return LeverTimeCalibrationResult(
        lever_arm_b_m=parameters[:3].copy(),
        time_offset_s=float(parameters[3]),
        covariance=covariance,
        correlation=correlation,
        parameter_std=parameter_std,
        predicted_residual_ned_m=predicted,
        corrected_residual_ned_m=corrected,
        robust_epoch_weights=epoch_weights.copy(),
        raw_rms_ned_m=np.sqrt(np.mean(residuals**2, axis=0)),
        corrected_rms_ned_m=np.sqrt(np.mean(corrected**2, axis=0)),
        condition_number=condition_number,
        iteration_count=iteration_count,
        downweighted_epoch_count=int(np.count_nonzero(epoch_weights < 1.0)),
    )


def calibrate_time_offset_with_fixed_lever(
    body_to_ned: np.ndarray,
    velocity_ned_mps: np.ndarray,
    residual_ned_m: np.ndarray,
    fixed_lever_b_m: np.ndarray,
    std_ned_m: np.ndarray | None = None,
    *,
    angular_rate_b_rps: np.ndarray | None = None,
    huber_threshold: float = 2.5,
    max_iterations: int = 20,
    convergence_tolerance: float = 1e-10,
) -> FixedLeverTimeCalibrationResult:
    """Fit only time offset after applying an independently known lever arm."""
    rotations = np.asarray(body_to_ned, dtype=float)
    velocities = np.asarray(velocity_ned_mps, dtype=float)
    residuals = np.asarray(residual_ned_m, dtype=float)
    lever = np.asarray(fixed_lever_b_m, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("body_to_ned must have shape (N, 3, 3)")
    sample_count = rotations.shape[0]
    if sample_count < 2:
        raise ValueError("fixed-lever time calibration requires at least two epochs")
    if velocities.shape != (sample_count, 3) or residuals.shape != (sample_count, 3):
        raise ValueError("velocity and residual arrays must have shape (N, 3)")
    if lever.shape != (3,) or not np.all(np.isfinite(lever)):
        raise ValueError("fixed_lever_b_m must be a finite 3-vector")
    if not (
        np.all(np.isfinite(rotations))
        and np.all(np.isfinite(velocities))
        and np.all(np.isfinite(residuals))
    ):
        raise ValueError("fixed-lever calibration inputs must be finite")
    angular_rates = (
        np.zeros((sample_count, 3))
        if angular_rate_b_rps is None
        else np.asarray(angular_rate_b_rps, dtype=float)
    )
    if angular_rates.shape != (sample_count, 3) or not np.all(
        np.isfinite(angular_rates)
    ):
        raise ValueError("angular_rate_b_rps must have shape (N, 3) and be finite")
    reported_std_is_known = std_ned_m is not None
    if std_ned_m is None:
        standard_deviation = np.ones((sample_count, 3))
    else:
        standard_deviation = np.asarray(std_ned_m, dtype=float)
        if standard_deviation.shape == (3,):
            standard_deviation = np.broadcast_to(
                standard_deviation,
                (sample_count, 3),
            ).copy()
        if standard_deviation.shape != (sample_count, 3):
            raise ValueError("std_ned_m must have shape (3,) or (N, 3)")
        if not np.all(np.isfinite(standard_deviation)) or np.any(
            standard_deviation <= 0.0
        ):
            raise ValueError("std_ned_m must be positive and finite")
    if not np.isfinite(huber_threshold) or huber_threshold <= 0.0:
        raise ValueError("huber_threshold must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(convergence_tolerance) or convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive and finite")

    lever_prediction = np.einsum("nij,j->ni", rotations, lever)
    residual_after_lever = residuals - lever_prediction
    effective_velocity = antenna_velocity_ned(
        rotations,
        velocities,
        angular_rates,
        lever,
    )
    inverse_std = 1.0 / standard_deviation
    epoch_weights = np.ones(sample_count)
    time_offset = 0.0
    iteration_count = 0
    for iteration_count in range(1, max_iterations + 1):
        weighted_inverse_variance = epoch_weights[:, None] * inverse_std**2
        denominator = float(np.sum(weighted_inverse_variance * effective_velocity**2))
        if denominator <= np.finfo(float).eps:
            raise ValueError("time offset is unobservable: velocity excitation is zero")
        updated_offset = float(
            np.sum(
                weighted_inverse_variance
                * effective_velocity
                * residual_after_lever
            )
            / denominator
        )
        normalized_residual = (
            residual_after_lever - effective_velocity * updated_offset
        ) * inverse_std
        mahalanobis_norm = np.linalg.norm(normalized_residual, axis=1)
        updated_weights = np.ones(sample_count)
        outliers = mahalanobis_norm > huber_threshold
        updated_weights[outliers] = huber_threshold / mahalanobis_norm[outliers]
        if abs(updated_offset - time_offset) <= convergence_tolerance * max(
            1.0,
            abs(updated_offset),
        ):
            time_offset = updated_offset
            epoch_weights = updated_weights
            break
        time_offset = updated_offset
        epoch_weights = updated_weights

    predicted = lever_prediction + effective_velocity * time_offset
    corrected = residuals - predicted
    weighted_inverse_variance = epoch_weights[:, None] * inverse_std**2
    time_information = float(
        np.sum(weighted_inverse_variance * effective_velocity**2)
    )
    normalized_residual = corrected * inverse_std
    weighted_rss = float(np.sum(epoch_weights[:, None] * normalized_residual**2))
    degrees_of_freedom = 3 * sample_count - 1
    reduced_chi_square = weighted_rss / degrees_of_freedom
    covariance_scale = (
        max(1.0, reduced_chi_square)
        if reported_std_is_known
        else max(np.finfo(float).eps, reduced_chi_square)
    )
    time_std = float(np.sqrt(covariance_scale / time_information))
    return FixedLeverTimeCalibrationResult(
        lever_arm_b_m=lever.copy(),
        time_offset_s=time_offset,
        time_offset_std_s=time_std,
        predicted_residual_ned_m=predicted,
        corrected_residual_ned_m=corrected,
        robust_epoch_weights=epoch_weights.copy(),
        raw_rms_ned_m=np.sqrt(np.mean(residuals**2, axis=0)),
        corrected_rms_ned_m=np.sqrt(np.mean(corrected**2, axis=0)),
        iteration_count=iteration_count,
        downweighted_epoch_count=int(np.count_nonzero(epoch_weights < 1.0)),
        time_information=time_information,
    )
