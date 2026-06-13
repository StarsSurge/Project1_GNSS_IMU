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
]
