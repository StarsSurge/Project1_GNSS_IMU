"""扩展卡尔曼滤波器（EKF）—— 用于非线性系统。

Extended Kalman Filter for non-linear systems.

当运动模型 ``f(x)`` 和/或观测模型 ``h(x)`` 是非线性函数时，
EKF 在每一步围绕当前状态估计进行一阶泰勒展开（线性化），
然后套用标准卡尔曼滤波的预测-更新框架。

与线性 KF 的关键区别：
    - KF: 存储常量 F, H 矩阵
    - EKF: 每步调用 F_jac(x), H_jac(x) 重新计算雅可比

用法：:

    ekf = ExtendedKalmanFilter(x=x0, P=P0, Q=Q, R=R,
                                f=cv_predict, F_jac=cv_jac,
                                h=range_bearing, H_jac=range_bearing_jac)
    ekf.predict()
    x_upd, P_upd, K, residual = ekf.update(z)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Array = np.ndarray

#: 非线性函数签名：``fn(x) → y``，接收 (n,1) 返回 (n,1) 或 (m,1)
NonlinearFn = Callable[[Array], Array]
#: 雅可比函数签名：``jac(x) → J``，接收 (n,1) 返回 (n,n) 或 (m,n)
JacobianFn = Callable[[Array], Array]


@dataclass
class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器，使用用户提供的非线性模型和雅可比。

    字段
    ----
    x : (n, 1)
        状态向量。
    P : (n, n)
        状态估计误差协方差。
    Q : (n, n)
        过程噪声协方差。
    R : (m, m)
        观测噪声协方差。
    f : NonlinearFn
        非线性状态转移函数：``x_pred = f(x)``。
    F_jac : JacobianFn
        ``f`` 在当前状态处的雅可比矩阵，形状 (n, n)。
        ``F = F_jac(x)``，用于协方差传播。
    h : NonlinearFn
        非线性观测函数：``z_pred = h(x)``。
    H_jac : JacobianFn
        ``h`` 在当前状态处的雅可比矩阵，形状 (m, n)。
        ``H = H_jac(x)``，用于卡尔曼增益计算。
    """

    x: Array
    P: Array
    Q: Array
    R: Array
    # 可调用对象不参与 repr 和比较（dataclass 默认行为会导致异常）
    f: NonlinearFn = field(repr=False, compare=False)
    F_jac: JacobianFn = field(repr=False, compare=False)
    h: NonlinearFn = field(repr=False, compare=False)
    H_jac: JacobianFn = field(repr=False, compare=False)

    #: 缓存的状态维度 n 和观测维度 m（在 __post_init__ 中推断）
    _n: int = field(default=0, init=False, repr=False)
    _m: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """构造后转换为 float 数组并校验形状，同时 smoke-test 所有 callable。"""
        self.x = np.asarray(self.x, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        self.Q = np.asarray(self.Q, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self._check_shapes()

    @property
    def state_dim(self) -> int:
        """状态向量的维度 *n*（只读）。"""
        return self._n

    @property
    def measurement_dim(self) -> int:
        """观测向量的维度 *m*（只读）。"""
        return self._m

    # ------------------------------------------------------------------
    # 核心算法
    # ------------------------------------------------------------------

    def predict(self) -> tuple[Array, Array]:
        """用非线性运动模型传播状态，用线性化雅可比传播协方差（预测步）。

        运算：
            F = F_jac(x)              — 在当前状态处线性化运动模型
            x = f(x)                  — 用完整的非线性模型传播状态
            P = F @ P @ F^T + Q       — 用线性化雅可比传播协方差

        返回
        ----
        x_pred : (n, 1)  预测后的状态（经非线性 f 传播）
        P_pred : (n, n)  预测后的协方差（经线性化 F 传播）
        """
        # 先在当前状态处计算雅可比（必须在 f(x) 之前）
        F = self.F_jac(self.x)
        # 用完整的非线性函数传播状态
        self.x = self.f(self.x)
        # 用线性化的 F 传播协方差
        self.P = F @ self.P @ F.T + self.Q
        return self.x, self.P

    def update(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """用观测修正状态（更新步）。

        运算：
            H = H_jac(x)                    — 在当前状态处线性化观测模型
            z_pred = h(x)                   — 用完整非线性模型预测观测
            residual = z - z_pred           — 新息（观测 - 预测观测）
            S = H @ P @ H^T + R             — 新息协方差
            K = P @ H^T @ S^{-1}            — 卡尔曼增益
            x = x + K @ residual            — 状态修正
            P = (I - K @ H) @ P             — 约瑟夫形式协方差更新

        参数
        ----
        z : (m, 1)  观测向量

        返回
        ----
        x_upd : (n, 1)     更新后的状态
        P_upd : (n, n)     更新后的协方差（约瑟夫形式）
        K : (n, m)         卡尔曼增益
        residual : (m, 1)  观测残差（新息）
        """
        z = np.asarray(z, dtype=float)
        if z.shape != (self._m, 1):
            raise ValueError(
                f"Expected z shape ({self._m}, 1), got {z.shape}"
            )

        # 在当前状态处线性化观测模型
        H = self.H_jac(self.x)
        # 用完整非线性模型预测观测
        z_pred = self.h(self.x)
        # 计算新息
        residual = z - z_pred

        # 新息协方差
        S = H @ self.P @ H.T + self.R
        # 数值稳定的卡尔曼增益计算（solve 替代 inv）
        B = self.P @ H.T
        K = np.linalg.solve(S.T, B.T).T

        # 状态修正
        self.x = self.x + K @ residual
        # 约瑟夫形式协方差更新
        I = np.eye(self._n)
        self.P = (I - K @ H) @ self.P
        return self.x, self.P, K, residual

    def step(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """运行一次完整的预测-更新周期。

        等价于 ``self.predict()`` 然后 ``self.update(z)``。
        """
        self.predict()
        return self.update(z)

    # ------------------------------------------------------------------
    # 形状校验
    # ------------------------------------------------------------------

    def _check_shapes(self) -> None:
        """校验矩阵形状，推断 n 和 m，并对所有 callable 做 smoke-test。

        校验规则：
            - x 必须是 (n, 1) 列向量
            - P, Q 必须是 (n, n)
            - R 必须是方阵 (m, m)
            - f(x) 输出形状必须为 (n, 1)
            - F_jac(x) 输出形状必须为 (n, n)
            - h(x) 输出形状必须为 (m, 1)
            - H_jac(x) 输出形状必须为 (m, n)
        """
        # 确保 x 是二维列向量
        if self.x.ndim != 2 or self.x.shape[1] != 1:
            raise ValueError(
                f"Expected x shape (n, 1), got {self.x.shape}"
            )
        n = self.x.shape[0]

        # P 和 Q 必须是 (n, n)
        for name in ("P", "Q"):
            value = getattr(self, name)
            if value.shape != (n, n):
                raise ValueError(
                    f"Expected {name} shape ({n}, {n}), got {value.shape}"
                )

        # R 必须是方阵
        m = self.R.shape[0]
        if self.R.shape[1] != m:
            raise ValueError(
                f"Expected square R, got {self.R.shape}"
            )

        # Smoke-test: 在初始状态上调用所有 callable，
        # 确保不抛出异常且输出形状正确
        for fn, name, out_cols in [
            (self.F_jac, "F_jac", n),
            (self.h, "h", m),
            (self.H_jac, "H_jac", n),
        ]:
            try:
                out = fn(self.x)
            except Exception as exc:
                raise ValueError(
                    f"{name}(x) raised {type(exc).__name__}: {exc}"
                ) from exc

            # 确定期望的形状
            expected = (m, n) if name == "H_jac" else (
                (n, n) if name == "F_jac" else (m, 1)
            )
            if out.shape != expected:
                raise ValueError(
                    f"Expected {name}(x) shape {expected}, got {out.shape}"
                )

        # Smoke-test f(x) 输出形状
        try:
            out_f = self.f(self.x)
        except Exception as exc:
            raise ValueError(
                f"f(x) raised {type(exc).__name__}: {exc}"
            ) from exc
        if out_f.shape != (n, 1):
            raise ValueError(
                f"Expected f(x) shape ({n}, 1), got {out_f.shape}"
            )

        self._n = n
        self._m = m


# ══════════════════════════════════════════════════════════════════
# 距离-方位角传感器模型（Range-Bearing Sensor Model）
# ══════════════════════════════════════════════════════════════════


def cv_predict(x: Array, dt: float) -> Array:
    """恒定速度 (Constant Velocity) 状态转移函数。

    状态布局：``[p₁ … p_d, v₁ … v_d]ᵀ``（共 2*d 个元素）。
    转移规则：位置 += 速度 × dt；速度保持不变。

    参数
    ----
    x : (2d, 1)  当前状态向量
    dt : float   时间步长 [s]

    返回
    ----
    (2d, 1)  传播后的状态
    """
    n = x.shape[0]
    dim = n // 2                     # 空间维度数
    F = np.eye(n)
    F[:dim, dim:] = np.eye(dim) * dt  # 位置 += 速度 × dt
    return F @ x


def cv_F_jac(_x: Array, dt: float) -> Array:
    """恒定速度模型的雅可比矩阵（实际上与状态无关，为常量矩阵）。

    对于恒定速度模型，f(x) = F @ x 本身就是线性的，
    所以雅可比就是 F 本身。这里保留函数接口以保持 API 一致性。

    参数
    ----
    _x : (2d, 1)  状态向量（仅用于确定维度，值被忽略）
    dt : float    时间步长 [s]

    返回
    ----
    (2d, 2d)  雅可比矩阵 F
    """
    n = _x.shape[0]
    dim = n // 2
    F = np.eye(n)
    F[:dim, dim:] = np.eye(dim) * dt
    return F


def range_bearing_h(
    x: Array, sensor_pos: tuple[float, float] = (0.0, 0.0)
) -> Array:
    """距离-方位角观测函数。

    将目标笛卡尔坐标映射为极坐标观测。

    状态布局：``[px, py, vx, vy]ᵀ``。

    观测：
        r  = √((px - sx)² + (py - sy)²)   — 距离
        θ  = atan2(py - sy, px - sx)       — 方位角 [rad]

    参数
    ----
    x : (4, 1)  目标状态 [px, py, vx, vy]^T
    sensor_pos : (float, float)  传感器位置 (sx, sy)

    返回
    ----
    z_pred : (2, 1)  预测观测 [r, θ]^T
    """
    px, py = x[0, 0], x[1, 0]
    sx, sy = sensor_pos
    dx, dy = px - sx, py - sy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return np.array([[r], [theta]])


def range_bearing_H_jac(
    x: Array, sensor_pos: tuple[float, float] = (0.0, 0.0)
) -> Array:
    """距离-方位角观测函数的解析雅可比矩阵。

    Jacobian of :func:`range_bearing_h` with respect to state x.

    数学形式：:

        H = [[ ∂r/∂px,  ∂r/∂py,  0, 0 ],
             [ ∂θ/∂px,  ∂θ/∂py,  0, 0 ]]

          = [[ dx/r,   dy/r,   0, 0 ],
             [-dy/r²,  dx/r²,  0, 0 ]]

    其中 dx = px - sx, dy = py - sy, r = √(dx² + dy²)。

    当目标恰好位于传感器处 (r ≈ 0) 时，返回零矩阵避免除零。

    参数
    ----
    x : (4, 1)  目标状态 [px, py, vx, vy]^T
    sensor_pos : (float, float)  传感器位置 (sx, sy)

    返回
    ----
    H : (2, 4)  观测雅可比矩阵
    """
    px, py = x[0, 0], x[1, 0]
    sx, sy = sensor_pos
    dx, dy = px - sx, py - sy
    r = np.sqrt(dx**2 + dy**2)
    r2 = r**2

    # 防止除零：目标恰好在传感器位置时返回零矩阵
    if r < 1e-12:
        return np.zeros((2, 4))

    H = np.zeros((2, 4))
    H[0, 0] = dx / r       # ∂r/∂px
    H[0, 1] = dy / r       # ∂r/∂py
    H[1, 0] = -dy / r2     # ∂θ/∂px
    H[1, 1] = dx / r2      # ∂θ/∂py
    return H


# ── 工厂函数 ──────────────────────────────────────────────────


def create_range_bearing_ekf(
    dt: float = 0.1,
    initial_position: tuple[float, float] = (1.0, 0.0),
    initial_velocity: tuple[float, float] = (0.0, 1.0),
    initial_covariance: float = 1.0,
    process_noise_position: float = 0.01,
    process_noise_velocity: float = 0.01,
    measurement_noise_range: float = 0.1,
    measurement_noise_bearing: float = 0.05,
    sensor_pos: tuple[float, float] = (0.0, 0.0),
) -> ExtendedKalmanFilter:
    """创建用于 2D 距离-方位角目标跟踪的 EKF。

    状态：``[px, py, vx, vy]ᵀ``（4 维）。
    观测：``[距离, 方位角]ᵀ``（2 维），传感器位于 ``sensor_pos``。

    这是扩展卡尔曼滤波的经典教学示例，与 GNSS 伪距测量的
    非线性特性直接相关。

    参数
    ----
    dt : float
        预测步的时间间隔 [s]。
    initial_position : (float, float)
        初始位置 (px, py) [m]。
    initial_velocity : (float, float)
        初始速度 (vx, vy) [m/s]。
    initial_covariance : float
        初始协方差矩阵的对角值。
    process_noise_position : float
        位置的 diagonal 过程噪声方差。
    process_noise_velocity : float
        速度的 diagonal 过程噪声方差。
    measurement_noise_range : float
        距离观测噪声的标准差 [m]。
    measurement_noise_bearing : float
        方位角观测噪声的标准差 [rad]。
    sensor_pos : (float, float)
        传感器的笛卡尔坐标 (sx, sy) [m]。

    返回
    ----
    ExtendedKalmanFilter  状态维度 n=4，观测维度 m=2
    """
    x = np.array(
        [
            [initial_position[0]],
            [initial_position[1]],
            [initial_velocity[0]],
            [initial_velocity[1]],
        ],
        dtype=float,
    )
    P = np.eye(4) * initial_covariance
    Q = np.diag(
        [process_noise_position, process_noise_position,
         process_noise_velocity, process_noise_velocity]
    )
    # R 对角线使用方差（标准差平方）
    R = np.diag([measurement_noise_range**2, measurement_noise_bearing**2])

    # 用闭包将 dt 和 sensor_pos 绑定到 callable 中
    def _f(state: Array) -> Array:
        return cv_predict(state, dt)

    def _F_jac(state: Array) -> Array:
        return cv_F_jac(state, dt)

    def _h(state: Array) -> Array:
        return range_bearing_h(state, sensor_pos)

    def _H_jac(state: Array) -> Array:
        return range_bearing_H_jac(state, sensor_pos)

    return ExtendedKalmanFilter(
        x=x, P=P, Q=Q, R=R, f=_f, F_jac=_F_jac, h=_h, H_jac=_H_jac,
    )
