"""扩展卡尔曼滤波器 (EKF) 单元测试。

Tests for the Extended Kalman Filter.
验证：
    - 解析雅可比与有限差分数值雅可比一致
    - 极点处的雅可比安全性（避免除零）
    - EKF 在简单轨迹上收敛
    - 返回值和形状校验正确
"""

import numpy as np

from gnss_imu import ExtendedKalmanFilter, create_range_bearing_ekf
from gnss_imu.extended_kalman_filter import (
    cv_F_jac,              # 恒定速度雅可比
    cv_predict,            # 恒定速度状态转移
    range_bearing_h,       # 距离-方位角观测函数
    range_bearing_H_jac,   # 距离-方位角解析雅可比
    range_bearing_residual,
)


# ══════════════════════════════════════════════════════════════════
# 雅可比验证（有限差分 vs 解析）
# ══════════════════════════════════════════════════════════════════


def _finite_diff_H(
    x: np.ndarray,
    h_fn: callable,
    eps: float = 1e-6,
) -> np.ndarray:
    """用中心差分法数值计算 *h_fn* 在 *x* 处的雅可比矩阵。

    参数
    ----
    x : (n, 1)  展开点
    h_fn : callable  观测函数 h(x) → (m, 1)
    eps : float  差分步长

    返回
    ----
    J : (m, n)  数值雅可比
    """
    n = x.shape[0]
    h0 = h_fn(x)
    m = h0.shape[0]
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i, 0] = eps
        h_plus = h_fn(x + dx)
        h_minus = h_fn(x - dx)
        J[:, i] = ((h_plus - h_minus) / (2 * eps)).ravel()
    return J


def test_range_bearing_H_jac_matches_finite_diff() -> None:
    """解析 H_jac 必须与中心差分数值雅可比一致。

    这是 EKF 实现正确性的关键测试 —
    雅可比错了，整个滤波器就错了。
    """
    x = np.array([[3.0], [4.0], [1.0], [0.5]])  # 典型状态
    analytic = range_bearing_H_jac(x, sensor_pos=(0.0, 0.0))
    numeric = _finite_diff_H(x, lambda s: range_bearing_h(s, (0.0, 0.0)))
    assert np.allclose(analytic, numeric, atol=1e-5), (
        f"解析雅可比:\n{analytic}\n数值雅可比:\n{numeric}"
    )


def test_range_bearing_H_jac_near_sensor() -> None:
    """目标在传感器位置时，雅可比应返回零矩阵（无除零错误）。

    这是边界条件测试 — r ≈ 0 时公式退化为 0/0。
    """
    x = np.array([[0.0], [0.0], [1.0], [0.5]])
    H = range_bearing_H_jac(x, sensor_pos=(0.0, 0.0))
    assert np.allclose(H, 0.0)


def test_range_bearing_residual_wraps_across_pi() -> None:
    """跨越 ±pi 分支时，方位角新息应选择最短角距离。"""
    z = np.array([[10.0], [np.pi - 0.001]])
    z_pred = np.array([[10.0], [-np.pi + 0.001]])

    residual = range_bearing_residual(z, z_pred)

    np.testing.assert_allclose(residual, [[0.0], [-0.002]], atol=1e-12)


def test_ekf_update_is_stable_across_bearing_wrap() -> None:
    """接近负 x 轴时，等价方位角不能触发巨大状态修正。"""
    ekf = create_range_bearing_ekf(
        initial_position=(-10.0, -0.01),
        initial_velocity=(0.0, 0.0),
        initial_covariance=1.0,
        measurement_noise_range=0.1,
        measurement_noise_bearing=0.01,
    )
    x_before = ekf.x.copy()
    z = np.array([[np.hypot(10.0, 0.01)], [np.pi - 0.001]])

    x_upd, _, _, residual = ekf.update(z)

    assert abs(residual[1, 0]) < 0.01
    assert np.linalg.norm(x_upd[:2] - x_before[:2]) < 0.1


def test_cv_F_jac_is_constant() -> None:
    """恒定速度模型的雅可比应与状态无关（因为 f(x) = F·x 是线性的）。

    任意两个状态应产生相同的 F_jac。
    """
    x1 = np.array([[0.0], [0.0], [1.0], [0.5]])
    x2 = np.array([[100.0], [-50.0], [3.0], [-2.0]])
    F1 = cv_F_jac(x1, dt=0.1)
    F2 = cv_F_jac(x2, dt=0.1)
    assert np.allclose(F1, F2)


# ══════════════════════════════════════════════════════════════════
# 收敛性测试
# ══════════════════════════════════════════════════════════════════


def test_ekf_converges_on_straight_line() -> None:
    """EKF 必须能跟踪简单的恒定速度目标。

    从精确初始状态启动，稳态位置误差应很小。
    """
    ekf = create_range_bearing_ekf(
        dt=0.1,
        initial_position=(5.0, 0.0),
        initial_velocity=(0.5, 0.0),
        initial_covariance=1.0,
        measurement_noise_range=0.1,
        measurement_noise_bearing=0.02,
        sensor_pos=(0.0, 0.0),
    )

    true = np.array([[5.0], [0.0], [0.5], [0.0]], dtype=float)
    rng = np.random.RandomState(42)

    errors = []
    for step in range(50):
        # 真实运动：恒定速度前进
        true = cv_predict(true, dt=0.1)

        # 带噪声的距离-方位角观测
        z_true = range_bearing_h(true, sensor_pos=(0.0, 0.0))
        noise = np.array(
            [[rng.randn() * 0.1], [rng.randn() * 0.02]]
        )
        z = z_true + noise

        ekf.step(z)
        pos_err = np.linalg.norm(ekf.x[:2] - true[:2])
        errors.append(pos_err)

    # 收敛后稳态误差应小于 0.5 m
    steady_errors = errors[-20:]
    mean_err = np.mean(steady_errors)
    assert mean_err < 0.5, (
        f"稳态位置误差 {mean_err:.3f} 超过 0.5 m"
    )


def test_ekf_initial_state_matters() -> None:
    """从远离真实值的初始状态启动，滤波器也应最终收敛。

    这是滤波器鲁棒性的基本检验 —
    大初始误差不应导致发散。
    """
    ekf = create_range_bearing_ekf(
        dt=0.1,
        initial_position=(8.0, 3.0),   # 远离真实 (5, 0)
        initial_velocity=(0.0, 0.0),
        initial_covariance=5.0,         # 大初始不确定性
        measurement_noise_range=0.1,
        measurement_noise_bearing=0.02,
        sensor_pos=(0.0, 0.0),
    )

    true = np.array([[5.0], [0.0], [0.5], [0.0]], dtype=float)
    rng = np.random.RandomState(42)

    errors = []
    for _ in range(80):
        true = cv_predict(true, dt=0.1)
        z_true = range_bearing_h(true, sensor_pos=(0.0, 0.0))
        noise = np.array(
            [[rng.randn() * 0.1], [rng.randn() * 0.02]]
        )
        z = z_true + noise
        ekf.step(z)
        pos_err = np.linalg.norm(ekf.x[:2] - true[:2])
        errors.append(pos_err)

    # 最终误差应小于初始误差（呈下降趋势）
    assert errors[-1] < errors[0], (
        "最终误差应小于初始误差"
    )


# ══════════════════════════════════════════════════════════════════
# 形状和校验
# ══════════════════════════════════════════════════════════════════


def test_ekf_predict_update_return_shapes() -> None:
    """predict() 和 update() 必须返回正确形状的数组。

    状态: (4,1), 协方差: (4,4), 增益: (4,2), 残差: (2,1)
    """
    ekf = create_range_bearing_ekf()
    z = np.array([[2.0], [0.5]])

    x_pred, P_pred = ekf.predict()
    assert x_pred.shape == (4, 1)
    assert P_pred.shape == (4, 4)

    x_upd, P_upd, K, residual = ekf.update(z)
    assert x_upd.shape == (4, 1)
    assert P_upd.shape == (4, 4)
    assert K.shape == (4, 2)
    assert residual.shape == (2, 1)


def test_ekf_rejects_bad_z_shape() -> None:
    """传入 (m,) 而非 (m,1) 的观测应抛出 ValueError。"""
    ekf = create_range_bearing_ekf()
    ekf.predict()
    with np.testing.assert_raises(ValueError):
        ekf.update(np.array([2.0, 0.5]))  # 错误: (2,) 而非 (2,1)


def test_ekf_rejects_bad_state_shape() -> None:
    """初始状态为 (n,) 而非 (n,1) 应抛出 ValueError。

    与 KalmanFilter 保持相同的列向量约定。
    """
    import numpy as np
    from gnss_imu.extended_kalman_filter import (
        cv_F_jac, cv_predict,
        range_bearing_h, range_bearing_H_jac,
    )

    with np.testing.assert_raises(ValueError):
        ExtendedKalmanFilter(
            x=np.array([1.0, 0.0, 0.0, 1.0]),  # 错误: (4,) 而非 (4,1)
            P=np.eye(4),
            Q=np.eye(4),
            R=np.eye(2),
            f=lambda s: cv_predict(s, 0.1),
            F_jac=lambda s: cv_F_jac(s, 0.1),
            h=lambda s: range_bearing_h(s),
            H_jac=lambda s: range_bearing_H_jac(s),
        )
