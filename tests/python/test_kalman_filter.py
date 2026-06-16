"""通用 KalmanFilter 单元测试。

Tests for the general N-dimensional Kalman filter.
验证：
    - 与 KalmanFilter1D 的交叉验证（数值一致性）
    - 2D 工厂函数产生的矩阵形状和结构
    - 错误输入被正确拒绝
    - 2D 轨迹上的协方差一致性和误差边界
"""

import numpy as np

from gnss_imu import KalmanFilter, create_constant_velocity_filter_1d
from gnss_imu.kalman_filter import create_constant_velocity_filter_nd


# ══════════════════════════════════════════════════════════════════
# 与 KalmanFilter1D 的交叉验证
# ══════════════════════════════════════════════════════════════════


def test_1d_cross_validation_matches_kalman_filter_1d() -> None:
    """用通用 KF 配置为 1D 时，必须复现 1D KF 的参考结果。

    这是关键的回归测试，确保通用 KF 在退化情况下的数值与
    教学版 KalmanFilter1D 完全一致。
    """
    kf = create_constant_velocity_filter_1d(dt=1.0)
    z = np.array([[1.2], [0.9]])

    kf.predict()
    x_upd, P_upd, K, residual = kf.update(z)

    expected_x = np.array([[1.02387807], [0.96858594]])
    assert np.allclose(x_upd, expected_x)
    assert P_upd.shape == (2, 2)
    assert K.shape == (2, 2)
    assert residual.shape == (2, 1)


def test_1d_step_matches_reference() -> None:
    """step() 便捷方法的结果必须与分步调用一致。"""
    kf = create_constant_velocity_filter_1d(dt=1.0)
    z = np.array([[1.2], [0.9]])

    x_upd, _, _, _ = kf.step(z)
    assert np.allclose(x_upd, [[1.02387807], [0.96858594]])


def test_update_uses_joseph_covariance_form() -> None:
    """协方差更新应与 Joseph 形式的直接计算一致。"""
    kf = create_constant_velocity_filter_1d(dt=1.0)
    kf.predict()
    P_pred = kf.P.copy()
    H = kf.H.copy()
    R = kf.R.copy()

    _, P_upd, K, _ = kf.update(np.array([[1.2], [0.9]]))

    A = np.eye(kf.state_dim) - K @ H
    expected = A @ P_pred @ A.T + K @ R @ K.T
    np.testing.assert_allclose(P_upd, expected)
    np.testing.assert_allclose(P_upd, P_upd.T)
    assert np.linalg.eigvalsh(P_upd).min() >= -1e-12


# ══════════════════════════════════════════════════════════════════
# 2D 工厂函数验证
# ══════════════════════════════════════════════════════════════════


def test_2d_factory_shapes() -> None:
    """create_constant_velocity_filter_nd(dim=2) 产生正确形状的矩阵。"""
    kf = create_constant_velocity_filter_nd(dim=2, dt=1.0)
    assert kf.x.shape == (4, 1)   # 2D 位置 + 2D 速度
    assert kf.P.shape == (4, 4)
    assert kf.F.shape == (4, 4)
    assert kf.H.shape == (2, 4)   # 仅观测位置
    assert kf.Q.shape == (4, 4)
    assert kf.R.shape == (2, 2)
    assert kf.state_dim == 4
    assert kf.measurement_dim == 2


def test_2d_factory_initial_state() -> None:
    """工厂函数必须将初始位置和速度正确写入 x。"""
    kf = create_constant_velocity_filter_nd(
        dim=2,
        initial_position=np.array([5.0, -3.0]),
        initial_velocity=np.array([0.5, 0.1]),
    )
    np.testing.assert_array_almost_equal(
        kf.x, np.array([[5.0], [-3.0], [0.5], [0.1]])
    )


def test_2d_factory_F_structure() -> None:
    """F 必须对每个维度编码 p += v * dt。"""
    kf = create_constant_velocity_filter_nd(dim=2, dt=0.5)
    # 设置状态：零位置，vx=1, vy=2
    kf.x = np.array([[0.0], [0.0], [1.0], [2.0]])
    kf.predict()
    # 预期：位置 = [0+1*0.5, 0+2*0.5] = [0.5, 1.0]
    np.testing.assert_array_almost_equal(
        kf.x, np.array([[0.5], [1.0], [1.0], [2.0]])
    )


# ══════════════════════════════════════════════════════════════════
# 形状校验
# ══════════════════════════════════════════════════════════════════


def test_rejects_wrong_measurement_shape() -> None:
    """传入形状错误的观测应抛出 ValueError。"""
    kf = create_constant_velocity_filter_1d()
    kf.predict()
    with np.testing.assert_raises(ValueError):
        kf.update(np.array([1.2, 0.9]))  # (2,) 不是 (2,1)


def test_rejects_wrong_state_shape() -> None:
    """初始状态为 (n,) 而非 (n,1) 应抛出 ValueError。"""
    with np.testing.assert_raises(ValueError):
        KalmanFilter(
            x=np.array([0.0, 1.0]),  # 错误: (2,) 而非 (2,1)
            P=np.eye(2),
            F=np.eye(2),
            H=np.eye(2),
            Q=np.eye(2),
            R=np.eye(2),
        )


def test_rejects_H_column_mismatch() -> None:
    """H 的列数必须等于状态维度 n。"""
    with np.testing.assert_raises(ValueError):
        KalmanFilter(
            x=np.array([[0.0], [0.0]]),
            P=np.eye(2),
            F=np.eye(2),
            H=np.eye(3),  # 错误: 3 列用于 2 维状态
            Q=np.eye(2),
            R=np.eye(3),
        )


def test_rejects_R_size_mismatch() -> None:
    """R 必须是 m×m，其中 m = H 的行数。"""
    with np.testing.assert_raises(ValueError):
        KalmanFilter(
            x=np.array([[0.0], [0.0]]),
            P=np.eye(2),
            F=np.eye(2),
            H=np.eye(2),
            Q=np.eye(2),
            R=np.eye(3),  # 错误: 3×3 用于 2 行 H
        )


# ══════════════════════════════════════════════════════════════════
# 2D 轨迹一致性
# ══════════════════════════════════════════════════════════════════


def test_2d_covariance_shrinks_with_measurements() -> None:
    """在噪声数据上运行滤波器时，不确定性必须随时间降低。

    协方差的迹 (trace) 在多次观测后应小于初始值。
    """
    kf = create_constant_velocity_filter_nd(
        dim=2, dt=1.0, measurement_noise=1.0
    )

    initial_trace = np.trace(kf.P)

    # 模拟恒定速度目标 + 噪声位置观测
    true_pos = np.array([[1.0], [0.5]])
    true_vel = np.array([[0.1], [0.2]])
    kf.x = np.vstack([true_pos, true_vel])

    rng = np.random.RandomState(42)
    for _ in range(20):
        true_pos += true_vel * 1.0
        z = true_pos + rng.randn(2, 1) * 0.5
        kf.step(z)

    final_trace = np.trace(kf.P)
    assert final_trace < initial_trace, (
        f"协方差迹应该在观测后收缩 "
        f"({initial_trace:.2f} → {final_trace:.2f})"
    )


def test_2d_errors_stay_within_three_sigma() -> None:
    """滤波估计应在真实状态的 3σ 范围内。

    这是滤波器一致性的基本检验：不应过度自信。
    """
    kf = create_constant_velocity_filter_nd(
        dim=2, dt=1.0, measurement_noise=1.0
    )
    true_pos = np.array([[1.0], [0.5]])
    true_vel = np.array([[0.1], [0.2]])
    kf.x = np.vstack([true_pos, true_vel])

    rng = np.random.RandomState(42)
    for _ in range(30):
        true_pos += true_vel * 1.0
        z = true_pos + rng.randn(2, 1) * 0.5
        kf.step(z)

    # 检查最终位置误差是否在 3σ 内
    pos_error = np.linalg.norm(kf.x[:2] - true_pos)
    pos_std = np.sqrt(kf.P[0, 0] + kf.P[1, 1])
    assert pos_error < 3 * max(pos_std, 0.2), (
        f"位置误差 {pos_error:.3f} 超过 3σ 边界 {3 * pos_std:.3f}"
    )
