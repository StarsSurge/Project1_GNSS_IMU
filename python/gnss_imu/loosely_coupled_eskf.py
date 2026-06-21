"""Production-oriented 15-state GNSS/IMU loosely coupled ESKF baseline.

The implementation is designed for auditable offline replay and algorithm
development.  It includes WGS-84 Earth rotation, transport rate, normal
gravity, two-sample IMU mechanization, GNSS antenna lever arm, innovation
gating, Joseph covariance updates, and covariance health checks. Fixed-lag
delayed updates are provided by :mod:`gnss_imu.delayed_eskf`.

It is a verified MVP, not a deployment certification.  Scale-factor,
misalignment, temperature models, clock-drift estimation, and sensor-specific
calibration remain integration responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gnss_imu.imu_mechanization import (
    IMUIncrement,
    bias_correct_increment,
    correct_two_sample_increments,
    finite_vector,
    normalize_quat,
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
    skew,
)

Array = np.ndarray

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_RATE_RPS = 7.2921151467e-5


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def radii_of_curvature(latitude_rad: float) -> tuple[float, float]:
    """Return WGS-84 meridian and prime-vertical radii ``(Rm, Rn)`` [m]."""
    latitude = _finite_scalar(latitude_rad, "latitude_rad")
    sin_lat = np.sin(latitude)
    denominator = 1.0 - WGS84_E2 * sin_lat**2
    rn = WGS84_A_M / np.sqrt(denominator)
    rm = WGS84_A_M * (1.0 - WGS84_E2) / denominator**1.5
    return float(rm), float(rn)


def normal_gravity_mps2(latitude_rad: float, height_m: float) -> float:
    """Somigliana normal gravity with a second-order free-air correction."""
    latitude = _finite_scalar(latitude_rad, "latitude_rad")
    height = _finite_scalar(height_m, "height_m")
    sin2 = np.sin(latitude) ** 2
    gravity_ellipsoid = (
        9.7803253359
        * (1.0 + 0.00193185265241 * sin2)
        / np.sqrt(1.0 - WGS84_E2 * sin2)
    )
    return float(
        gravity_ellipsoid
        * (1.0 - 2.0 * height / WGS84_A_M + 3.0 * height**2 / WGS84_A_M**2)
    )


def earth_rate_ned(latitude_rad: float) -> Array:
    """Earth rotation rate expressed in NED [rad/s]."""
    latitude = _finite_scalar(latitude_rad, "latitude_rad")
    return np.array(
        [
            EARTH_RATE_RPS * np.cos(latitude),
            0.0,
            -EARTH_RATE_RPS * np.sin(latitude),
        ]
    )


def transport_rate_ned(
    latitude_rad: float,
    height_m: float,
    velocity_ned_mps: Array,
) -> Array:
    """NED transport rate caused by motion over the ellipsoid [rad/s]."""
    latitude = _finite_scalar(latitude_rad, "latitude_rad")
    height = _finite_scalar(height_m, "height_m")
    velocity = finite_vector(velocity_ned_mps, 3, "velocity_ned_mps")
    rm, rn = radii_of_curvature(latitude)
    vn, ve, _ = velocity
    return np.array(
        [
            ve / (rn + height),
            -vn / (rm + height),
            -ve * np.tan(latitude) / (rn + height),
        ]
    )


def geodetic_difference_ned(
    latitude_rad: float,
    longitude_rad: float,
    height_m: float,
    reference_latitude_rad: float,
    reference_longitude_rad: float,
    reference_height_m: float,
) -> Array:
    """Linearized geodetic difference relative to a nearby reference [m]."""
    rm, rn = radii_of_curvature(reference_latitude_rad)
    cos_lat = np.cos(reference_latitude_rad)
    if abs(cos_lat) < 1e-8:
        raise ValueError("local NED longitude difference is singular near the poles")
    return np.array(
        [
            (latitude_rad - reference_latitude_rad) * (rm + reference_height_m),
            (longitude_rad - reference_longitude_rad)
            * (rn + reference_height_m)
            * cos_lat,
            -(height_m - reference_height_m),
        ]
    )


def apply_ned_position_delta(
    latitude_rad: float,
    longitude_rad: float,
    height_m: float,
    delta_ned_m: Array,
) -> tuple[float, float, float]:
    """Apply a small NED position correction to geodetic coordinates."""
    delta = finite_vector(delta_ned_m, 3, "delta_ned_m")
    rm, rn = radii_of_curvature(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    if abs(cos_lat) < 1e-8:
        raise ValueError("local NED position update is singular near the poles")
    latitude_new = latitude_rad + delta[0] / (rm + height_m)
    longitude_new = longitude_rad + delta[1] / ((rn + height_m) * cos_lat)
    height_new = height_m - delta[2]
    return float(latitude_new), float(longitude_new), float(height_new)


@dataclass(frozen=True)
class TimedIMUIncrement:
    """An IMU increment whose timestamp denotes the end of its interval."""

    time_s: float
    dtheta_rad: Array
    dvel_mps: Array
    dt_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _finite_scalar(self.time_s, "time_s"))
        increment = IMUIncrement(self.dtheta_rad, self.dvel_mps, self.dt_s)
        object.__setattr__(self, "dtheta_rad", increment.dtheta.copy())
        object.__setattr__(self, "dvel_mps", increment.dvel.copy())
        object.__setattr__(self, "dt_s", increment.dt)

    def as_increment(self) -> IMUIncrement:
        return IMUIncrement(self.dtheta_rad, self.dvel_mps, self.dt_s)


@dataclass(frozen=True)
class GNSSPositionMeasurement:
    """GNSS antenna position measurement and NED one-sigma uncertainty."""

    time_s: float
    latitude_rad: float
    longitude_rad: float
    height_m: float
    std_ned_m: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _finite_scalar(self.time_s, "time_s"))
        object.__setattr__(
            self, "latitude_rad", _finite_scalar(self.latitude_rad, "latitude_rad")
        )
        object.__setattr__(
            self, "longitude_rad", _finite_scalar(self.longitude_rad, "longitude_rad")
        )
        object.__setattr__(self, "height_m", _finite_scalar(self.height_m, "height_m"))
        std = finite_vector(self.std_ned_m, 3, "std_ned_m")
        if np.any(std <= 0.0):
            raise ValueError("std_ned_m must be strictly positive")
        object.__setattr__(self, "std_ned_m", std.copy())


@dataclass(frozen=True)
class IMUNoiseModel:
    """Continuous-time IMU noise parameters in SI units."""

    accel_noise_density_mps2_sqrthz: float
    gyro_noise_density_rps_sqrthz: float
    accel_bias_drive_mps2_sqrts: float
    gyro_bias_drive_rps_sqrts: float
    accel_bias_correlation_s: float = np.inf
    gyro_bias_correlation_s: float = np.inf

    def __post_init__(self) -> None:
        for name in (
            "accel_noise_density_mps2_sqrthz",
            "gyro_noise_density_rps_sqrthz",
            "accel_bias_drive_mps2_sqrts",
            "gyro_bias_drive_rps_sqrts",
        ):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))
        for name in ("accel_bias_correlation_s", "gyro_bias_correlation_s"):
            value = float(getattr(self, name))
            if not (np.isinf(value) or (np.isfinite(value) and value > 0.0)):
                raise ValueError(f"{name} must be positive or infinity")
            object.__setattr__(self, name, value)

    @classmethod
    def mems_default(cls) -> "IMUNoiseModel":
        """Conservative starting profile; replace with calibrated Allan results."""
        return cls(2.0e-3, 8.0e-5, 2.0e-4, 8.0e-6, 3600.0, 3600.0)

    @classmethod
    def navigation_grade_default(cls) -> "IMUNoiseModel":
        """High-grade starting profile; device datasheet/calibration still required."""
        return cls(3.0e-4, 2.0e-6, 2.0e-5, 2.0e-7, 10800.0, 10800.0)


@dataclass(frozen=True)
class IMUCalibration:
    """Known linear corrections from reported sensor increments to body axes.

    The matrices may combine scale-factor, non-orthogonality, cross-axis, and
    sensor-to-body mounting corrections.  Bias remains an estimated ESKF state.
    """

    gyro_increment_matrix: Array = field(default_factory=lambda: np.eye(3))
    accel_increment_matrix: Array = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        for name in ("gyro_increment_matrix", "accel_increment_matrix"):
            matrix = np.asarray(getattr(self, name), dtype=float)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite 3x3 matrix")
            if abs(np.linalg.det(matrix)) < 1e-9:
                raise ValueError(f"{name} must be nonsingular")
            object.__setattr__(self, name, matrix.copy())


@dataclass(frozen=True)
class ESKFConfig:
    """Filter policy, timing limits, lever arm, and numerical safeguards."""

    imu_noise: IMUNoiseModel = field(default_factory=IMUNoiseModel.mems_default)
    imu_calibration: IMUCalibration = field(default_factory=IMUCalibration)
    gnss_lever_arm_b_m: Array = field(default_factory=lambda: np.zeros(3))
    gnss_nis_threshold: float = 16.26623619623813
    max_gnss_time_error_s: float = 0.006
    max_imu_gap_factor: float = 3.0
    covariance_eigenvalue_floor: float = 1e-15

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gnss_lever_arm_b_m",
            finite_vector(self.gnss_lever_arm_b_m, 3, "gnss_lever_arm_b_m").copy(),
        )
        for name in (
            "gnss_nis_threshold",
            "max_gnss_time_error_s",
            "max_imu_gap_factor",
            "covariance_eigenvalue_floor",
        ):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))


@dataclass
class ESKFState:
    """Nominal navigation state plus 15x15 right-error covariance."""

    time_s: float
    latitude_rad: float
    longitude_rad: float
    height_m: float
    velocity_ned_mps: Array
    q_bn: Array
    accel_bias_mps2: Array
    gyro_bias_rps: Array
    covariance: Array

    def __post_init__(self) -> None:
        self.time_s = _finite_scalar(self.time_s, "time_s")
        self.latitude_rad = _finite_scalar(self.latitude_rad, "latitude_rad")
        self.longitude_rad = _finite_scalar(self.longitude_rad, "longitude_rad")
        self.height_m = _finite_scalar(self.height_m, "height_m")
        self.velocity_ned_mps = finite_vector(
            self.velocity_ned_mps, 3, "velocity_ned_mps"
        ).copy()
        self.q_bn = normalize_quat(self.q_bn, "q_bn").copy()
        self.accel_bias_mps2 = finite_vector(
            self.accel_bias_mps2, 3, "accel_bias_mps2"
        ).copy()
        self.gyro_bias_rps = finite_vector(
            self.gyro_bias_rps, 3, "gyro_bias_rps"
        ).copy()
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (15, 15) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be a finite 15x15 matrix")
        self.covariance = 0.5 * (covariance + covariance.T)


@dataclass(frozen=True)
class GNSSUpdateResult:
    accepted: bool
    residual_ned_m: Array
    nis: float
    time_error_s: float


class LooselyCoupledESKF:
    """15-state NED error-state filter with in-place predict/update methods."""

    def __init__(self, state: ESKFState, config: ESKFConfig | None = None) -> None:
        self.state = state
        self.config = ESKFConfig() if config is None else config
        self.last_specific_force_b_mps2 = np.zeros(3)
        self.last_angular_rate_b_rps = np.zeros(3)
        self.accepted_gnss_updates = 0
        self.rejected_gnss_updates = 0
        self._enforce_covariance_health()

    def predict_two_sample(
        self,
        imu1: TimedIMUIncrement,
        imu2: TimedIMUIncrement,
    ) -> ESKFState:
        """Propagate nominal state and covariance over two contiguous samples."""
        continuity_tolerance = max(1e-6, 0.05 * max(imu1.dt_s, imu2.dt_s))
        expected_first_end = self.state.time_s + imu1.dt_s
        if abs(imu1.time_s - expected_first_end) > continuity_tolerance:
            raise ValueError(
                "imu1 is not contiguous with filter state: "
                f"state={self.state.time_s}, imu1={imu1.time_s}"
            )
        if abs((imu2.time_s - imu2.dt_s) - imu1.time_s) > continuity_tolerance:
            raise ValueError("imu1 and imu2 intervals are not contiguous")
        if imu1.dt_s > self.config.max_imu_gap_factor * imu2.dt_s or (
            imu2.dt_s > self.config.max_imu_gap_factor * imu1.dt_s
        ):
            raise ValueError("IMU interval gap exceeds configured factor")

        corrected = correct_two_sample_increments(
            IMUIncrement(
                self.config.imu_calibration.gyro_increment_matrix @ imu1.dtheta_rad,
                self.config.imu_calibration.accel_increment_matrix @ imu1.dvel_mps,
                imu1.dt_s,
            ),
            IMUIncrement(
                self.config.imu_calibration.gyro_increment_matrix @ imu2.dtheta_rad,
                self.config.imu_calibration.accel_increment_matrix @ imu2.dvel_mps,
                imu2.dt_s,
            ),
            b_g=self.state.gyro_bias_rps,
            b_a=self.state.accel_bias_mps2,
        )
        return self._propagate_corrected_increment(
            corrected.dtheta,
            corrected.dvel,
            corrected.dt,
            imu2.time_s,
        )

    def predict_single_sample(self, imu: TimedIMUIncrement) -> ESKFState:
        """Propagate one sample for event-boundary alignment.

        The normal navigation path should use two-sample propagation.  This
        method exists for odd sample counts and asynchronous event boundaries,
        where dropping or duplicating one increment would be worse.
        """
        continuity_tolerance = max(1e-6, 0.05 * imu.dt_s)
        expected_end = self.state.time_s + imu.dt_s
        if abs(imu.time_s - expected_end) > continuity_tolerance:
            raise ValueError(
                "IMU sample is not contiguous with filter state: "
                f"state={self.state.time_s}, imu={imu.time_s}"
            )
        calibrated = IMUIncrement(
            self.config.imu_calibration.gyro_increment_matrix @ imu.dtheta_rad,
            self.config.imu_calibration.accel_increment_matrix @ imu.dvel_mps,
            imu.dt_s,
        )
        dtheta, dvel = bias_correct_increment(
            calibrated,
            self.state.gyro_bias_rps,
            self.state.accel_bias_mps2,
        )
        # Constant-rate single-sample approximation for body rotation during
        # the velocity increment.  Coning/sculling require at least two samples.
        corrected_dvel = dvel + 0.5 * np.cross(dtheta, dvel)
        return self._propagate_corrected_increment(
            dtheta,
            corrected_dvel,
            imu.dt_s,
            imu.time_s,
        )

    def _propagate_corrected_increment(
        self,
        corrected_dtheta: Array,
        corrected_dvel: Array,
        dt: float,
        end_time_s: float,
    ) -> ESKFState:
        old_lat = self.state.latitude_rad
        old_lon = self.state.longitude_rad
        old_h = self.state.height_m
        old_v = self.state.velocity_ned_mps.copy()
        old_q = self.state.q_bn.copy()
        old_c_bn = quat_to_dcm(old_q)

        omega_ie_n = earth_rate_ned(old_lat)
        omega_en_n = transport_rate_ned(old_lat, old_h, old_v)
        omega_in_n = omega_ie_n + omega_en_n

        # Navigation frame rotates on the left; body increment rotates on the right.
        dq_nav = rotvec_to_quat(-omega_in_n * dt)
        dq_body = rotvec_to_quat(corrected_dtheta)
        q_new = normalize_quat(
            quat_multiply(quat_multiply(dq_nav, old_q), dq_body),
            "q_new",
        )

        # Two-sample correction expresses delta-v in the old body frame.  Apply
        # a half navigation-frame rotation before mapping it to NED.
        nav_half_rotation = np.eye(3) - 0.5 * skew(omega_in_n * dt)
        dvel_n = nav_half_rotation @ old_c_bn @ corrected_dvel
        gravity_n = np.array([0.0, 0.0, normal_gravity_mps2(old_lat, old_h)])
        coriolis_n = -np.cross(2.0 * omega_ie_n + omega_en_n, old_v)
        v_new = old_v + dvel_n + (gravity_n + coriolis_n) * dt

        average_v = 0.5 * (old_v + v_new)
        rm, rn = radii_of_curvature(old_lat)
        cos_lat = np.cos(old_lat)
        if abs(cos_lat) < 1e-8:
            raise ValueError("geodetic mechanization is singular near the poles")
        lat_new = old_lat + average_v[0] * dt / (rm + old_h)
        lon_new = old_lon + average_v[1] * dt / ((rn + old_h) * cos_lat)
        h_new = old_h - average_v[2] * dt

        specific_force_b = corrected_dvel / dt
        angular_rate_b = corrected_dtheta / dt
        self._propagate_covariance(old_c_bn, specific_force_b, angular_rate_b, dt)

        self.state.time_s = end_time_s
        self.state.latitude_rad = float(lat_new)
        self.state.longitude_rad = float(lon_new)
        self.state.height_m = float(h_new)
        self.state.velocity_ned_mps = v_new
        self.state.q_bn = q_new
        self.last_specific_force_b_mps2 = specific_force_b
        self.last_angular_rate_b_rps = angular_rate_b
        self._enforce_covariance_health()
        return self.state

    def update_gnss_position(
        self,
        measurement: GNSSPositionMeasurement,
    ) -> GNSSUpdateResult:
        """Update with GNSS antenna position after lever-arm compensation."""
        time_error = self.state.time_s - measurement.time_s
        if abs(time_error) > self.config.max_gnss_time_error_s:
            raise ValueError(
                "GNSS/IMU time mismatch exceeds configured limit: "
                f"{time_error} s"
            )

        measured_delta_ned = geodetic_difference_ned(
            measurement.latitude_rad,
            measurement.longitude_rad,
            measurement.height_m,
            self.state.latitude_rad,
            self.state.longitude_rad,
            self.state.height_m,
        )
        c_bn = quat_to_dcm(self.state.q_bn)
        lever_n = c_bn @ self.config.gnss_lever_arm_b_m
        residual = measured_delta_ned - lever_n

        h_matrix = np.zeros((3, 15))
        h_matrix[:, 0:3] = np.eye(3)
        h_matrix[:, 6:9] = -c_bn @ skew(self.config.gnss_lever_arm_b_m)
        measurement_covariance = np.diag(measurement.std_ned_m**2)
        innovation_covariance = (
            h_matrix @ self.state.covariance @ h_matrix.T
            + measurement_covariance
        )
        solved_residual = np.linalg.solve(innovation_covariance, residual)
        nis = float(residual @ solved_residual)
        if nis > self.config.gnss_nis_threshold:
            self.rejected_gnss_updates += 1
            return GNSSUpdateResult(False, residual.copy(), nis, time_error)

        cross_covariance = self.state.covariance @ h_matrix.T
        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            cross_covariance.T,
        ).T
        delta_x = kalman_gain @ residual

        identity = np.eye(15)
        update_operator = identity - kalman_gain @ h_matrix
        covariance_updated = (
            update_operator @ self.state.covariance @ update_operator.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        self.state.covariance = 0.5 * (
            covariance_updated + covariance_updated.T
        )
        self._inject_error_state(delta_x)
        self.accepted_gnss_updates += 1
        self._enforce_covariance_health()
        return GNSSUpdateResult(True, residual.copy(), nis, time_error)

    def _propagate_covariance(
        self,
        c_bn: Array,
        specific_force_b: Array,
        angular_rate_b: Array,
        dt: float,
    ) -> None:
        f_matrix = np.zeros((15, 15))
        f_matrix[0:3, 3:6] = np.eye(3)
        omega_ie_n = earth_rate_ned(self.state.latitude_rad)
        omega_en_n = transport_rate_ned(
            self.state.latitude_rad,
            self.state.height_m,
            self.state.velocity_ned_mps,
        )
        f_matrix[3:6, 3:6] = -skew(2.0 * omega_ie_n + omega_en_n)
        f_matrix[3:6, 6:9] = -c_bn @ skew(specific_force_b)
        f_matrix[3:6, 9:12] = -c_bn
        f_matrix[6:9, 6:9] = -skew(angular_rate_b)
        f_matrix[6:9, 12:15] = -np.eye(3)

        noise = self.config.imu_noise
        if np.isfinite(noise.accel_bias_correlation_s):
            f_matrix[9:12, 9:12] = (
                -np.eye(3) / noise.accel_bias_correlation_s
            )
        if np.isfinite(noise.gyro_bias_correlation_s):
            f_matrix[12:15, 12:15] = (
                -np.eye(3) / noise.gyro_bias_correlation_s
            )

        noise_mapping = np.zeros((15, 12))
        noise_mapping[3:6, 0:3] = -c_bn
        noise_mapping[6:9, 3:6] = -np.eye(3)
        noise_mapping[9:12, 6:9] = np.eye(3)
        noise_mapping[12:15, 9:12] = np.eye(3)
        continuous_noise = np.diag(
            np.repeat(
                [
                    noise.accel_noise_density_mps2_sqrthz**2,
                    noise.gyro_noise_density_rps_sqrthz**2,
                    noise.accel_bias_drive_mps2_sqrts**2,
                    noise.gyro_bias_drive_rps_sqrts**2,
                ],
                3,
            )
        )

        identity = np.eye(15)
        transition = identity + f_matrix * dt + 0.5 * (f_matrix @ f_matrix) * dt**2
        discrete_noise = noise_mapping @ continuous_noise @ noise_mapping.T * dt
        covariance = (
            transition @ self.state.covariance @ transition.T + discrete_noise
        )
        self.state.covariance = 0.5 * (covariance + covariance.T)

    def _inject_error_state(self, delta_x: Array) -> None:
        delta = finite_vector(delta_x, 15, "delta_x")
        lat, lon, height = apply_ned_position_delta(
            self.state.latitude_rad,
            self.state.longitude_rad,
            self.state.height_m,
            delta[0:3],
        )
        self.state.latitude_rad = lat
        self.state.longitude_rad = lon
        self.state.height_m = height
        self.state.velocity_ned_mps = self.state.velocity_ned_mps + delta[3:6]
        self.state.q_bn = normalize_quat(
            quat_multiply(self.state.q_bn, rotvec_to_quat(delta[6:9])),
            "q_bn",
        )
        self.state.accel_bias_mps2 = self.state.accel_bias_mps2 + delta[9:12]
        self.state.gyro_bias_rps = self.state.gyro_bias_rps + delta[12:15]

        reset_jacobian = np.eye(15)
        reset_jacobian[6:9, 6:9] = np.eye(3) - 0.5 * skew(delta[6:9])
        covariance = (
            reset_jacobian @ self.state.covariance @ reset_jacobian.T
        )
        self.state.covariance = 0.5 * (covariance + covariance.T)

    def _enforce_covariance_health(self) -> None:
        covariance = 0.5 * (self.state.covariance + self.state.covariance.T)
        if not np.all(np.isfinite(covariance)):
            raise FloatingPointError("ESKF covariance contains non-finite values")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = self.config.covariance_eigenvalue_floor
        if eigenvalues[0] < -1e-10:
            raise FloatingPointError(
                f"ESKF covariance lost positive semidefiniteness: {eigenvalues[0]}"
            )
        if eigenvalues[0] < floor:
            eigenvalues = np.maximum(eigenvalues, floor)
            covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        self.state.covariance = 0.5 * (covariance + covariance.T)


def default_initial_covariance(
    position_std_m: float = 2.0,
    velocity_std_mps: float = 0.5,
    attitude_std_deg: float = 5.0,
    accel_bias_std_mps2: float = 0.1,
    gyro_bias_std_deg_s: float = 0.02,
) -> Array:
    """Build a diagonal 15-state initial covariance from one-sigma values."""
    standard_deviations = np.concatenate(
        [
            np.full(3, _positive_scalar(position_std_m, "position_std_m")),
            np.full(3, _positive_scalar(velocity_std_mps, "velocity_std_mps")),
            np.full(3, np.deg2rad(_positive_scalar(attitude_std_deg, "attitude_std_deg"))),
            np.full(3, _positive_scalar(accel_bias_std_mps2, "accel_bias_std_mps2")),
            np.full(
                3,
                np.deg2rad(
                    _positive_scalar(gyro_bias_std_deg_s, "gyro_bias_std_deg_s")
                ),
            ),
        ]
    )
    return np.diag(standard_deviations**2)
