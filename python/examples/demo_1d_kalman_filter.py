r"""1D 卡尔曼滤波演示脚本。

Demo: 1D constant-velocity Kalman filter (educational).

使用一次观测 ``z = [1.2, 0.9]^T`` 执行预测-更新周期，
打印每个矩阵的值供对照学习笔记中的数值结果。

运行方式（从项目根目录）::

    $env:PYTHONPATH = "$PWD\python"
    python python\examples\demo_1d_kalman_filter.py
"""

import numpy as np

from gnss_imu import create_constant_velocity_filter


def print_matrix(name: str, value: np.ndarray) -> None:
    """打印矩阵名称、形状和内容。"""
    print(f"{name}: shape={value.shape}")
    print(value)
    print()


def main() -> None:
    """运行 1D KF 的预测-更新演示。"""
    # 创建默认参数的 1D 恒定速度滤波器
    kf = create_constant_velocity_filter(dt=1.0)

    # 观测值：[观测位置 = 1.2 m, 观测速度 = 0.9 m/s]
    #
    # 观测方程（线性）：
    #     z = H @ x + v
    #     ┌     ┐   ┌       ┐ ┌   ┐   ┌     ┐
    #     │ z_p │ = │ 1   0 │ │ p │ + │ v_p │   位置和速度均可直接观测
    #     │ z_v │   │ 0   1 │ │ v │   │ v_v │   H = I₂ (单位阵)
    #     └     ┘   └       ┘ └   ┘   └     ┘
    #     v ~ N(0, R),  R = diag(σ²_pos, σ²_vel)
    z = np.array([[1.2], [0.9]])

    # ── 初始状态 ──
    print_matrix("x initial (初始状态)", kf.x)
    print_matrix("P initial (初始协方差)", kf.P)

    # ── 预测步 ──
    x_pred, P_pred = kf.predict()
    print_matrix("x predicted (预测状态)", x_pred)
    print_matrix("P predicted (预测协方差)", P_pred)

    # ── 更新步 ──
    x_upd, P_upd, K, residual = kf.update(z)
    print_matrix("z (观测值)", z)
    print_matrix("residual (新息/残差)", residual)
    print_matrix("K (卡尔曼增益)", K)
    print_matrix("x updated (更新后状态)", x_upd)
    print_matrix("P updated (更新后协方差)", P_upd)

    # 预期结果：x_upd ≈ [[1.02387807], [0.96858594]]
    # 含义：滤波器在模型预测和观测之间做了加权平均


if __name__ == "__main__":
    main()
