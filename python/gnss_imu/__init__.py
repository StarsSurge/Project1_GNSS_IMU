"""教学用途的 GNSS/IMU 组合导航原型库。

Educational GNSS/IMU integration prototypes.

提供从 1D 卡尔曼滤波到扩展卡尔曼滤波 (EKF) 的递进式实现，
包含工厂函数和示例脚本，适合作为面试作品集和自学材料。
"""

# ── 扩展卡尔曼滤波器 ──────────────────────────────────────────
from gnss_imu.extended_kalman_filter import (
    ExtendedKalmanFilter,          # EKF 类：非线性系统滤波
    create_range_bearing_ekf,      # 工厂函数：2D 距离-方位角 EKF
)

# ── 通用 N 维线性卡尔曼滤波器 ──────────────────────────────────
from gnss_imu.kalman_filter import (
    KalmanFilter,                  # 通用 KF 类：任意状态/观测维度
    create_constant_velocity_filter_1d,   # 工厂函数：1D 匀速模型（与 KalmanFilter1D 数值等价）
    create_constant_velocity_filter_nd,   # 工厂函数：N 维匀速模型（仅位置观测）
)

# ── 教学用 1D 卡尔曼滤波器 ────────────────────────────────────
from gnss_imu.kalman_filter_1d import (
    KalmanFilter1D,                # 1D KF 类：硬编码 2 状态（位置+速度）
    create_constant_velocity_filter,      # 工厂函数：1D 匀速模型（全状态观测）
)
from gnss_imu.imu_allan import (
    allan_deviation,
    extract_allan_parameters,
    fit_allan_log_slope,
    load_imu_rate_csv,
    load_imu_rate_feather,
    load_imu_rate_table,
    overlapping_allan_deviation,
    sampling_interval_statistics,
    validate_uniform_sampling,
)
from gnss_imu.imu_mechanization import (
    IMUIncrement,
    NavigationState,
    TwoSampleCorrection,
    bias_correct_increment,
    correct_two_sample_increments,
    euler_zyx_to_quat,
    finite_vector,
    normalize_quat,
    positive_dt,
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
    skew,
)
from gnss_imu.loosely_coupled_eskf import (
    ESKFConfig,
    ESKFState,
    GNSSPositionMeasurement,
    GNSSUpdateResult,
    IMUCalibration,
    IMUNoiseModel,
    LooselyCoupledESKF,
    TimedIMUIncrement,
    apply_ned_position_delta,
    default_initial_covariance,
    earth_rate_ned,
    geodetic_difference_ned,
    normal_gravity_mps2,
    radii_of_curvature,
    transport_rate_ned,
)
from gnss_imu.dataset_visualization import (
    Dataset1,
    ImuIncrementData,
    RtkData,
    TruthNavData,
    decimation_indices,
    fit_body_lever_arm_from_residuals,
    geodetic_to_ecef,
    geodetic_to_ned,
    increments_to_rates,
    interpolate_columns,
    load_dataset1,
    rpy_deg_to_body_to_ned,
)

__all__ = [
    # 1D 教学版
    "KalmanFilter1D",
    "create_constant_velocity_filter",
    # 通用 N 维版
    "KalmanFilter",
    "create_constant_velocity_filter_1d",
    "create_constant_velocity_filter_nd",
    # 扩展卡尔曼滤波
    "ExtendedKalmanFilter",
    "create_range_bearing_ekf",
    # IMU Allan analysis
    "allan_deviation",
    "overlapping_allan_deviation",
    "sampling_interval_statistics",
    "validate_uniform_sampling",
    "load_imu_rate_csv",
    "load_imu_rate_feather",
    "load_imu_rate_table",
    "fit_allan_log_slope",
    "extract_allan_parameters",
    # IMU mechanization base components
    "NavigationState",
    "IMUIncrement",
    "TwoSampleCorrection",
    "finite_vector",
    "positive_dt",
    "normalize_quat",
    "skew",
    "quat_multiply",
    "rotvec_to_quat",
    "quat_to_dcm",
    "euler_zyx_to_quat",
    "bias_correct_increment",
    "correct_two_sample_increments",
    # GNSS/IMU loosely coupled ESKF
    "TimedIMUIncrement",
    "GNSSPositionMeasurement",
    "IMUNoiseModel",
    "IMUCalibration",
    "ESKFConfig",
    "ESKFState",
    "GNSSUpdateResult",
    "LooselyCoupledESKF",
    "radii_of_curvature",
    "normal_gravity_mps2",
    "earth_rate_ned",
    "transport_rate_ned",
    "geodetic_difference_ned",
    "apply_ned_position_delta",
    "default_initial_covariance",
    # Dataset visualization
    "Dataset1",
    "RtkData",
    "ImuIncrementData",
    "TruthNavData",
    "load_dataset1",
    "geodetic_to_ecef",
    "geodetic_to_ned",
    "increments_to_rates",
    "interpolate_columns",
    "decimation_indices",
    "rpy_deg_to_body_to_ned",
    "fit_body_lever_arm_from_residuals",
]
