"""Basic components for educational strapdown IMU mechanization.

This module intentionally stops at reusable building blocks.  The full nominal
state propagation step is left as the next MVP layer so its math can be written
and reviewed explicitly.

Conventions
-----------
- Navigation frame: local NED.
- ``q_bn`` maps body-frame vectors into the navigation frame.
- Quaternion convention: Hamilton ``[w, x, y, z]``.
- IMU inputs are increments: delta angle [rad] and delta velocity [m/s].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def finite_vector(value: Array, size: int, name: str) -> Array:
    """Return ``value`` as a finite 1-D vector with exactly ``size`` entries.

    中文说明:
        把输入整理成指定长度的一维浮点向量，并拒绝 NaN/Inf。
        IMU 机械编排里很多量必须是严格的 3 维或 4 维向量；
        如果维度错了还继续算，错误会很隐蔽地传播到姿态、速度和位置。
    """
    # 转成 float 数组，允许调用者传 list、tuple 或 numpy array。
    vector = np.asarray(value, dtype=float)
    # 用元素总数检查维度，既能接受 (3,) 也能接受 (3, 1) 这类输入。
    if vector.size != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    # 统一压成一维向量，方便后续矩阵和叉乘运算。
    vector = vector.reshape(size)
    # 拒绝 NaN、+Inf、-Inf，避免坏数据污染整个导航状态。
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def positive_dt(value: float, name: str = "dt") -> float:
    """Validate a positive, finite time interval in seconds.

    中文说明:
        IMU 时间间隔 dt 的单位是秒 [s]，必须大于 0。
        零或负 dt 通常表示时间戳重复、乱序或数据解析错误。
    """
    dt = float(value)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return dt


def normalize_quat(q: Array, name: str = "q") -> Array:
    """Return a unit Hamilton quaternion ``[w, x, y, z]``.

    中文说明:
        四元数必须保持单位范数，才能表示纯旋转。
        归一化可以抑制浮点误差累积；零四元数不能表示姿态，必须拒绝。
    """
    quat = finite_vector(q, 4, name)
    norm = np.linalg.norm(quat)
    if norm < 1e-15:
        raise ValueError(f"{name} must have nonzero norm")
    return quat / norm


@dataclass(frozen=True)
class NavigationState:
    """Nominal strapdown navigation state.

    Attributes
    ----------
    p_n : (3,)
        NED position [m].
    v_n : (3,)
        NED velocity [m/s].
    q_bn : (4,)
        Unit Hamilton quaternion mapping body vectors to NED.
    b_a : (3,)
        Accelerometer bias [m/s^2].
    b_g : (3,)
        Gyro bias [rad/s].
    """

    p_n: Array
    v_n: Array
    q_bn: Array
    b_a: Array
    b_g: Array

    def __post_init__(self) -> None:
        # dataclass 创建后立刻做形状、有限值和四元数归一化检查。
        # 这样后续算法可以假定状态量已经满足基本物理和数学约定。
        object.__setattr__(self, "p_n", finite_vector(self.p_n, 3, "p_n"))
        object.__setattr__(self, "v_n", finite_vector(self.v_n, 3, "v_n"))
        object.__setattr__(self, "q_bn", normalize_quat(self.q_bn, "q_bn"))
        object.__setattr__(self, "b_a", finite_vector(self.b_a, 3, "b_a"))
        object.__setattr__(self, "b_g", finite_vector(self.b_g, 3, "b_g"))


@dataclass(frozen=True)
class IMUIncrement:
    """One IMU increment sample.

    ``dtheta`` and ``dvel`` are already integrated over ``dt``.  Rate samples
    must be converted before constructing this object.
    """

    dtheta: Array
    dvel: Array
    dt: float

    def __post_init__(self) -> None:
        # IMUIncrement 表示“增量样本”而不是 rate 样本。
        # dtheta 单位 rad，dvel 单位 m/s，dt 单位 s。
        object.__setattr__(
            self, "dtheta", finite_vector(self.dtheta, 3, "dtheta")
        )
        object.__setattr__(self, "dvel", finite_vector(self.dvel, 3, "dvel"))
        object.__setattr__(self, "dt", positive_dt(self.dt))


@dataclass(frozen=True)
class TwoSampleCorrection:
    """Bias-compensated two-sample increments and correction terms."""

    dtheta: Array
    dvel: Array
    coning: Array
    sculling: Array
    rotation: Array
    dt: float


def skew(vector: Array) -> Array:
    """Return the cross-product matrix ``[vector x]``.

    中文说明:
        skew(a) @ b 等价于 np.cross(a, b)。
        在惯导误差方程里，叉乘矩阵常用来写线性化形式。
    """
    x, y, z = finite_vector(vector, 3, "vector")
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def quat_multiply(q1: Array, q2: Array) -> Array:
    """Hamilton quaternion product ``q1 ⊗ q2`` for ``[w, x, y, z]``.

    中文说明:
        四元数乘法不满足交换律，q1 ⊗ q2 与 q2 ⊗ q1 含义不同。
        本仓库姿态增量采用右乘：q_new = q_old ⊗ dq_body。
    """
    w1, x1, y1, z1 = finite_vector(q1, 4, "q1")
    w2, x2, y2, z2 = finite_vector(q2, 4, "q2")
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def rotvec_to_quat(rotvec: Array) -> Array:
    """Convert a body-frame rotation vector [rad] to a unit quaternion.

    中文说明:
        rotvec 的方向是旋转轴，模长是旋转角，单位是 rad。
        小角度时使用近似展开，避免 sin(angle/2)/angle 的数值不稳定。
    """
    rotvec = finite_vector(rotvec, 3, "rotvec")
    angle = np.linalg.norm(rotvec)
    # 小角度近似:
    #   q_w   ≈ 1 - angle^2 / 8
    #   q_xyz ≈ 0.5 * rotvec
    if angle < 1e-12:
        scale = 0.5 - angle**2 / 48.0
        quat = np.concatenate(([1.0 - angle**2 / 8.0], scale * rotvec))
    else:
        quat = np.concatenate(
            ([np.cos(0.5 * angle)], rotvec * np.sin(0.5 * angle) / angle)
        )
    return normalize_quat(quat)


def quat_to_dcm(q_bn: Array) -> Array:
    """Return ``C_bn``, mapping body-frame vectors into NED coordinates.

    中文说明:
        返回方向余弦矩阵 C_bn，满足:
            v_n = C_bn @ v_b
        其中 v_b 是机体系向量，v_n 是 NED 导航系向量。
    """
    w, x, y, z = normalize_quat(q_bn, "q_bn")
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def dcm_to_quat(c_bn: Array) -> Array:
    """Convert a proper body-to-NED direction cosine matrix to a quaternion."""
    matrix = np.asarray(c_bn, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("c_bn must be a finite 3x3 matrix")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-8
    ):
        raise ValueError("c_bn must be a proper orthonormal rotation matrix")

    trace = np.trace(matrix)
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return normalize_quat(quat, "q_bn")


def euler_zyx_to_quat(
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = False,
) -> Array:
    """Convert ZYX roll-pitch-yaw angles to ``q_bn``.

    The composition is ``q = q_z(yaw) ⊗ q_y(pitch) ⊗ q_x(roll)``.

    中文说明:
        欧拉角采用 ZYX 顺序：先 yaw，再 pitch，再 roll。
        输入可选 degree 或 rad，输出仍是 body -> NED 的 Hamilton 四元数。
    """
    angles = np.array([roll, pitch, yaw], dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("roll, pitch, and yaw must be finite")
    if degrees:
        angles = np.deg2rad(angles)

    r, p, y = angles
    qx = np.array([np.cos(r / 2.0), np.sin(r / 2.0), 0.0, 0.0])
    qy = np.array([np.cos(p / 2.0), 0.0, np.sin(p / 2.0), 0.0])
    qz = np.array([np.cos(y / 2.0), 0.0, 0.0, np.sin(y / 2.0)])
    return normalize_quat(quat_multiply(quat_multiply(qz, qy), qx))


def bias_correct_increment(
    imu: IMUIncrement,
    b_g: Array,
    b_a: Array,
) -> tuple[Array, Array]:
    """Remove additive gyro and accelerometer biases from one increment.

    中文说明:
        当前 IMU 输入是增量:
            dtheta [rad], dvel [m/s]
        但零偏是 rate 形式:
            b_g [rad/s], b_a [m/s^2]
        因此补偿增量时必须乘以 dt:
            dtheta_corrected = dtheta_measured - b_g * dt
            dvel_corrected   = dvel_measured   - b_a * dt
    """
    gyro_bias = finite_vector(b_g, 3, "b_g")
    accel_bias = finite_vector(b_a, 3, "b_a")
    return imu.dtheta - gyro_bias * imu.dt, imu.dvel - accel_bias * imu.dt


def correct_two_sample_increments(
    imu1: IMUIncrement,
    imu2: IMUIncrement,
    b_g: Array | None = None,
    b_a: Array | None = None,
) -> TwoSampleCorrection:
    """Apply two-sample coning, sculling, and rotation corrections.

    This helper prepares the corrected delta angle and delta velocity.  It does
    not update position, velocity, or attitude.
    """
    if b_g is None:
        b_g = np.zeros(3)
    if b_a is None:
        b_a = np.zeros(3)

    # 经典 2/3 圆锥/划桨系数建立在两个子样本等时长的假设上。
    # 允许很小的时间戳量化抖动，但拒绝真正的不等间隔输入，避免静默误用。
    if not np.isclose(imu1.dt, imu2.dt, rtol=5e-3, atol=1e-9):
        raise ValueError(
            "two-sample correction requires equal sample intervals; "
            f"got {imu1.dt} s and {imu2.dt} s"
        )

    dtheta1, dvel1 = bias_correct_increment(imu1, b_g, b_a)
    dtheta2, dvel2 = bias_correct_increment(imu2, b_g, b_a)

    # 先形成普通的一阶总增量。若忽略三维旋转不可交换性，
    # 这两个量就是最朴素的 delta-angle 和 delta-velocity 积分结果。
    # 文字公式:
    #   raw_dtheta = dtheta1 + dtheta2
    #   raw_dvel   = dvel1   + dvel2
    # 单位:
    #   raw_dtheta: rad, 是整个两子样周期内的角度增量，不是 rad/s。
    #   raw_dvel  : m/s, 是整个两子样周期内的速度增量，不是 m/s^2。
    # 注意:
    #   这里的 dtheta1/dtheta2 和 dvel1/dvel2 已经完成 bias 补偿。
    #   raw_* 只是普通相加结果，尚未包含 coning/sculling/rotation 交叉项。
    #   后续若要构造连续时间 F 矩阵，才可用 raw_dtheta / dt 近似角速度、
    #   raw_dvel / dt 近似比力；不要在机械编排这里把它们误当成 rate。
    raw_dtheta = dtheta1 + dtheta2
    raw_dvel = dvel1 + dvel2

    # 圆锥效应修正 coning correction:
    # 当两个连续小转角绕不同轴发生时，最终姿态不等于简单相加。
    # 例如先绕 x 轴转、再绕 y 轴转，会产生一个二阶的 z 轴等效转角。
    # 双子样线性角速度假设下，该二阶交叉项系数为 2/3。
    # 文字公式:
    #   coning = (2/3) * (dtheta1 x dtheta2)
    #   corrected_dtheta = raw_dtheta + coning
    coning = (2.0 / 3.0) * np.cross(dtheta1, dtheta2)

    # 划桨效应修正 sculling correction:
    # 速度增量 dvel 是在旋转中的 body frame 里积累的。若 IMU 一边转动
    # 一边受到比力，dtheta 与 dvel 的交叉项会改变最终等效速度增量。
    # 这里的两项分别对应“第一段转角影响第二段速度增量”和
    # “第一段速度增量与第二段转角耦合”的二阶效应。
    # 文字公式:
    #   sculling = (2/3) * [
    #       dtheta1 x dvel2 + dvel1 x dtheta2
    #   ]
    sculling = (2.0 / 3.0) * (
        np.cross(dtheta1, dvel2) + np.cross(dvel1, dtheta2)
    )

    # 旋转效应修正 rotation correction:
    # 即使不考虑双子样线性变化，整个积分周期内 body frame 本身也在旋转。
    # 速度增量方向需要用平均意义上的旋转做一次补偿，常用近似为
    # 0.5 * (总角增量 x 总速度增量)。
    # 文字公式:
    #   rotation = 0.5 * (raw_dtheta x raw_dvel)
    #   corrected_dvel = raw_dvel + sculling + rotation
    rotation = 0.5 * np.cross(raw_dtheta, raw_dvel)

    return TwoSampleCorrection(
        dtheta=raw_dtheta + coning,
        dvel=raw_dvel + sculling + rotation,
        coning=coning,
        sculling=sculling,
        rotation=rotation,
        dt=imu1.dt + imu2.dt,
    )
