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
    matrices[..., 0, 2] = -sy * sr - cy * sp * cr
    matrices[..., 1, 0] = sy * cp
    matrices[..., 1, 1] = cy * cr + sy * sp * sr
    matrices[..., 1, 2] = cy * sr - sy * sp * cr
    matrices[..., 2, 0] = sp
    matrices[..., 2, 1] = -cp * sr
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
