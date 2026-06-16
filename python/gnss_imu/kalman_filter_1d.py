"""教学用 1D 恒定速度卡尔曼滤波器。

Educational 1D constant-velocity Kalman filter.

状态向量为：

    x = [position, velocity]^T   （位置, 速度）

这个模块刻意保持实现小而显式，让每个矩阵运算直接对应
标准卡尔曼滤波的预测和更新公式。适合作为入门学习材料。

进阶推荐：
    - KalmanFilter（通用 N 维线性 KF）→ kalman_filter.py
    - ExtendedKalmanFilter（非线性 EKF）→ extended_kalman_filter.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 类型别名，提高可读性
Array = np.ndarray


@dataclass
class KalmanFilter1D:
    """1D 卡尔曼滤波器，状态为 [position, velocity]^T。

    所有矩阵尺寸硬编码为 2×2 或 2×1，便于初学者对照公式理解。

    字段
    ----
    x : (2, 1)  状态向量 [位置, 速度]^T
    P : (2, 2)  状态估计误差协方差
    F : (2, 2)  状态转移矩阵（运动模型）
    H : (2, 2)  观测矩阵（本例中位置和速度均可直接观测）
    Q : (2, 2)  过程噪声协方差（模型不确定性）
    R : (2, 2)  观测噪声协方差（传感器不确定性）
    """

    x: Array
    P: Array
    F: Array
    H: Array
    Q: Array
    R: Array

    def __post_init__(self) -> None:
        """构造后立即转换为 float 数组并校验形状。"""
        self.x = np.asarray(self.x, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        self.F = np.asarray(self.F, dtype=float)
        self.H = np.asarray(self.H, dtype=float)
        self.Q = np.asarray(self.Q, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self._check_shapes()

    def predict(self) -> tuple[Array, Array]:
        """用运动模型传播状态和协方差（预测步）。

        运算：
            x = F @ x          — 状态前向预测
            P = F @ P @ F^T + Q — 协方差传播 + 过程噪声注入

        返回
        ----
        x_pred : (2, 1)  预测后的状态
        P_pred : (2, 2)  预测后的协方差
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """用一次观测修正状态和协方差（更新步）。

        运算：
            residual = z - H @ x                    — 新息（观测残差）
            S = H @ P @ H^T + R                     — 新息协方差
            K = P @ H^T @ S^{-1}                    — 卡尔曼增益
            x = x + K @ residual                    — 状态修正
            A = I - K @ H
            P = A @ P @ A^T + K @ R @ K^T           — Joseph 形式

        参数
        ----
        z : (2, 1)  观测向量 [观测位置, 观测速度]^T

        返回
        ----
        x_upd : (2, 1)     更新后的状态
        P_upd : (2, 2)     更新后的协方差
        K : (2, 2)         卡尔曼增益（反映信任模型 vs 观测的程度）
        residual : (2, 1)  观测残差（新息）
        """
        z = np.asarray(z, dtype=float)
        if z.shape != (2, 1):
            raise ValueError(f"Expected z shape (2, 1), got {z.shape}")

        # 计算新息（innovation）
        residual = z - self.H @ self.x

        # 新息协方差
        S = self.H @ self.P @ self.H.T + self.R

        # 使用 solve 而非显式求逆，数值更稳定
        B = self.P @ self.H.T
        K = np.linalg.solve(S.T, B.T).T

        # 状态修正
        self.x = self.x + K @ residual

        # Joseph 形式在浮点运算中更好地保持对称性和半正定性。
        I = np.eye(self.P.shape[0])
        A = I - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x, self.P, K, residual

    def step(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """运行一次完整的预测-更新周期。

        等价于依次调用 predict() 然后 update(z)。
        """
        self.predict()
        return self.update(z)

    def _check_shapes(self) -> None:
        """校验所有矩阵形状是否符合 1D KF 的固定维度。"""
        expected_shapes = {
            "x": (2, 1),
            "P": (2, 2),
            "F": (2, 2),
            "H": (2, 2),
            "Q": (2, 2),
            "R": (2, 2),
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got {value.shape}"
                )


# ── 工厂函数 ──────────────────────────────────────────────────


def create_constant_velocity_filter(
    dt: float = 1.0,
    initial_position: float = 0.0,
    initial_velocity: float = 1.0,
    initial_covariance: float = 1.0,
    process_noise_position: float = 0.1,
    process_noise_velocity: float = 0.1,
    measurement_noise_position: float = 4.0,
    measurement_noise_velocity: float = 1.0,
) -> KalmanFilter1D:
    """创建入门示例中使用的标准 1D 恒定速度滤波器。

    参数
    ----
    dt : float
        时间步长（预测间隔）。
    initial_position : float
        初始位置估计 [m]。
    initial_velocity : float
        初始速度估计 [m/s]。
    initial_covariance : float
        初始协方差矩阵的对角值（越大表示越不确定）。
    process_noise_position : float
        位置的过程噪声方差（模型不匹配程度）。
    process_noise_velocity : float
        速度的过程噪声方差。
    measurement_noise_position : float
        位置观测噪声方差（传感器噪声）。
    measurement_noise_velocity : float
        速度观测噪声方差。

    返回
    ----
    KalmanFilter1D  配置好的滤波器实例
    """
    x = np.array([[initial_position], [initial_velocity]])
    P = np.eye(2) * initial_covariance
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = np.diag([process_noise_position, process_noise_velocity])
    R = np.diag([measurement_noise_position, measurement_noise_velocity])

    return KalmanFilter1D(x=x, P=P, F=F, H=H, Q=Q, R=R)
