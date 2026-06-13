"""通用 N 维线性卡尔曼滤波器。

General N-dimensional linear Kalman filter.

本模块将 ``KalmanFilter1D`` 推广到任意状态维度和观测维度，
可用于 2D / 3D 定位问题，是迈向非线性扩展卡尔曼滤波 (EKF) 的中间一步。

状态与观测：:

    x: (n, 1) — 状态向量（n 维）
    z: (m, 1) — 观测向量（m 维，通常 m ≤ n）

用法：:

    kf = KalmanFilter(x=x0, P=P0, F=F, H=H, Q=Q, R=R)
    kf.predict()                          # 预测步
    x_upd, P_upd, K, residual = kf.update(z)  # 更新步
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 类型别名
Array = np.ndarray


@dataclass
class KalmanFilter:
    """线性卡尔曼滤波器，支持自定义状态和观测维度。

    与 ``KalmanFilter1D`` 的区别：
        - 状态维度 *n* 和观测维度 *m* 从输入矩阵形状自动推断
        - ``H`` 可以是 (m, n) 而非必须 (n, n)
        - 适用于 2D/3D 定位等更复杂的场景

    字段
    ----
    x : (n, 1)
        状态向量。
    P : (n, n)
        状态估计误差协方差。
    F : (n, n)
        状态转移（运动模型）矩阵。
    H : (m, n)
        观测矩阵，将状态映射到观测空间。
    Q : (n, n)
        过程噪声协方差（模型不确定性）。
    R : (m, m)
        观测噪声协方差（传感器不确定性）。
    """

    x: Array
    P: Array
    F: Array
    H: Array
    Q: Array
    R: Array

    #: 缓存的维度（在 __post_init__ 中设置）
    _n: int = 0
    _m: int = 0

    def __post_init__(self) -> None:
        """构造后转换为 float 并校验形状，同时推断 n 和 m。"""
        self.x = np.asarray(self.x, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        self.F = np.asarray(self.F, dtype=float)
        self.H = np.asarray(self.H, dtype=float)
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
        """用线性运动模型传播状态和协方差（预测步）。

        运算：
            x = F @ x          — 状态前向传播
            P = F @ P @ F^T + Q — 协方差传播 + 过程噪声注入

        返回
        ----
        x_pred : (n, 1)  预测后的状态
        P_pred : (n, n)  预测后的协方差
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """用观测修正状态（更新步）。

        运算：
            residual = z - H @ x             — 新息（观测残差）
            S = H @ P @ H^T + R              — 新息协方差
            K = P @ H^T @ S^{-1}             — 卡尔曼增益
            x = x + K @ residual             — 状态修正
            P = (I - K @ H) @ P              — 约瑟夫形式协方差更新

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

        # 计算新息
        residual = z - self.H @ self.x

        # 新息协方差
        S = self.H @ self.P @ self.H.T + self.R

        # 用 solve 而非 inv 计算卡尔曼增益（数值更稳定）
        B = self.P @ self.H.T
        K = np.linalg.solve(S.T, B.T).T

        # 状态修正
        self.x = self.x + K @ residual

        # 约瑟夫形式：保证协方差对称正定
        I = np.eye(self._n)
        self.P = (I - K @ self.H) @ self.P
        return self.x, self.P, K, residual

    def step(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """运行一次完整的预测-更新周期。

        便捷方法，等价于 ``self.predict()`` 然后 ``self.update(z)``。
        """
        self.predict()
        return self.update(z)

    # ------------------------------------------------------------------
    # 形状校验
    # ------------------------------------------------------------------

    def _check_shapes(self) -> None:
        """校验所有矩阵形状，并推断状态维度 n 和观测维度 m。

        规则：
            - x 必须是 (n, 1) 列向量
            - P, F, Q 必须是 (n, n)
            - H 必须是 (m, n)，列数匹配状态维度
            - R 必须是 (m, m)，行数匹配 H 的行数
        """
        # 先确保 x 是二维列向量
        if self.x.ndim != 2 or self.x.shape[1] != 1:
            raise ValueError(
                f"Expected x shape (n, 1), got {self.x.shape}"
            )
        n = self.x.shape[0]
        m, n_h = self.H.shape

        # P, F, Q 必须都是 (n, n)
        for name, expected in [
            ("P", (n, n)),
            ("F", (n, n)),
            ("Q", (n, n)),
        ]:
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(
                    f"Expected {name} shape {expected}, got {value.shape}"
                )

        # H 的列数必须等于状态维度 n
        if n_h != n:
            raise ValueError(
                f"H must have {n} columns to match state dim, got {n_h}"
            )

        # R 的行列数必须等于观测维度 m
        if self.R.shape != (m, m):
            raise ValueError(
                f"Expected R shape {(m, m)} to match H rows, got {self.R.shape}"
            )

        self._n = n
        self._m = m


# ── 工厂函数 ──────────────────────────────────────────────────


def create_constant_velocity_filter_1d(
    dt: float = 1.0,
    initial_position: float = 0.0,
    initial_velocity: float = 1.0,
    initial_covariance: float = 1.0,
    process_noise_position: float = 0.1,
    process_noise_velocity: float = 0.1,
    measurement_noise_position: float = 4.0,
    measurement_noise_velocity: float = 1.0,
) -> KalmanFilter:
    """创建 1D 恒定速度卡尔曼滤波器（用于交叉验证）。

    使用与 ``KalmanFilter1D.create_constant_velocity_filter``
    相同的参数，产生数值完全相同的结果。位置和速度均可直接观测。

    返回
    ----
    KalmanFilter  配置好的通用 KF 实例（n=2, m=2）
    """
    x = np.array([[initial_position], [initial_velocity]])
    P = np.eye(2) * initial_covariance
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.eye(2)
    Q = np.diag([process_noise_position, process_noise_velocity])
    R = np.diag([measurement_noise_position, measurement_noise_velocity])
    return KalmanFilter(x=x, P=P, F=F, H=H, Q=Q, R=R)


def create_constant_velocity_filter_nd(
    dim: int = 2,
    dt: float = 1.0,
    initial_position: Array | None = None,
    initial_velocity: Array | None = None,
    initial_covariance: float = 1.0,
    process_noise_position: float = 0.1,
    process_noise_velocity: float = 0.1,
    measurement_noise: float = 1.0,
) -> KalmanFilter:
    """创建 *d* 维恒定速度卡尔曼滤波器。

    状态排列为 ``[p₁ … p_d, v₁ … v_d]ᵀ`` — 先位置后速度。
    仅观测位置（``H`` 提取前 *d* 个元素），速度由滤波器内部估计。
    这是典型定位场景：外部传感器提供位置，滤波器维护速度估计。

    参数
    ----
    dim : int
        空间维度数（默认 2）。
    dt : float
        预测步的时间间隔 [s]。
    initial_position : (dim,) array, 可选
        各维度初始位置，默认为零向量。
    initial_velocity : (dim,) array, 可选
        各维度初始速度，默认为零向量。
    initial_covariance : float
        初始 ``P`` 矩阵的对角值。
    process_noise_position : float
        位置状态的过程噪声方差。
    process_noise_velocity : float
        速度状态的过程噪声方差。
    measurement_noise : float
        每个位置观测维度的噪声方差。

    返回
    ----
    KalmanFilter  状态维度 n=2*dim，观测维度 m=dim
    """
    n = 2 * dim   # 状态维度：位置 + 速度
    m = dim       # 观测维度：仅位置

    if initial_position is None:
        initial_position = np.zeros(dim)
    if initial_velocity is None:
        initial_velocity = np.zeros(dim)

    # 组装状态向量 [p₁…p_d | v₁…v_d]^T
    x = np.concatenate(
        [
            np.asarray(initial_position, dtype=float).reshape(dim, 1),
            np.asarray(initial_velocity, dtype=float).reshape(dim, 1),
        ],
        axis=0,
    )

    P = np.eye(n) * initial_covariance

    # F = [[I_d,  dt·I_d],
    #      [0_d,     I_d]]
    # 含义：位置 += 速度 × dt，速度不变（恒定速度假设）
    F = np.eye(n)
    F[:dim, dim:] = np.eye(dim) * dt

    # H = [I_d, 0_d]  — 仅观测位置分量
    H = np.zeros((m, n))
    H[:, :dim] = np.eye(dim)

    # Q = diag(σ²_pos·I_d, σ²_vel·I_d)
    Q = np.diag(
        np.concatenate(
            [
                np.full(dim, process_noise_position),
                np.full(dim, process_noise_velocity),
            ]
        )
    )

    R = np.eye(m) * measurement_noise

    return KalmanFilter(x=x, P=P, F=F, H=H, Q=Q, R=R)
