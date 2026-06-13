"""KalmanFilter1D 单元测试。

Tests for the 1D constant-velocity Kalman filter (educational version).
验证：
    - 预测-更新后状态与参考值一致
    - step() 便捷方法与分步调用等价
    - 错误形状的输入被正确拒绝
"""

import numpy as np

from gnss_imu import KalmanFilter1D, create_constant_velocity_filter


def test_one_step_matches_reference_result() -> None:
    """一次预测-更新后的状态必须与文档中的参考值一致。

    参考值: x_upd ≈ [[1.02387807], [0.96858594]]
    来自 docs/02_1d_kalman_filter.md 中的数值计算。
    """
    kf = create_constant_velocity_filter(dt=1.0)
    z = np.array([[1.2], [0.9]])  # 观测: 位置=1.2m, 速度=0.9m/s

    kf.predict()
    x_upd, P_upd, K, residual = kf.update(z)

    expected_x = np.array([[1.02387807], [0.96858594]])

    # 验证数值精度
    assert np.allclose(x_upd, expected_x)
    # 验证返回矩阵形状
    assert P_upd.shape == (2, 2)
    assert K.shape == (2, 2)
    assert residual.shape == (2, 1)


def test_step_runs_predict_and_update() -> None:
    """step() 一步法结果必须与分步调用 predict+update 一致。"""
    kf = create_constant_velocity_filter(dt=1.0)
    z = np.array([[1.2], [0.9]])

    x_upd, _, _, _ = kf.step(z)

    assert np.allclose(x_upd, [[1.02387807], [0.96858594]])


def test_rejects_wrong_measurement_shape() -> None:
    """传入 (2,) 而非 (2,1) 的观测应抛出 ValueError。"""
    kf = create_constant_velocity_filter(dt=1.0)
    kf.predict()

    with np.testing.assert_raises(ValueError):
        kf.update(np.array([1.2, 0.9]))  # 形状错误: (2,) 不是 (2,1)


def test_rejects_wrong_state_shape() -> None:
    """初始状态为 (2,) 而非 (2,1) 应抛出 ValueError。

    列向量约定是该项目的基本设计决策，在所有 KF/EKF 实现中强制。
    """
    with np.testing.assert_raises(ValueError):
        KalmanFilter1D(
            x=np.array([0.0, 1.0]),  # 错误: (2,) 而非 (2,1)
            P=np.eye(2),
            F=np.eye(2),
            H=np.eye(2),
            Q=np.eye(2),
            R=np.eye(2),
        )
