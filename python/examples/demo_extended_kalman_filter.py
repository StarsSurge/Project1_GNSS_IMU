r"""2D 距离-方位角扩展卡尔曼滤波演示。

Demo: 2D range-bearing Extended Kalman Filter.

目标以恒定速度直线运动，静止在原点处的传感器测量距离和方位角
（均带噪声）。EKF 从这些非线性观测中估计目标状态 ``[px, py, vx, vy]ᵀ``。

这是 EKF 的经典教学示例，与 GNSS 伪距测量的非线性特性直接相关。

运行方式（从项目根目录）::

    $env:PYTHONPATH = "$PWD\python"
    python python\examples\demo_extended_kalman_filter.py

输出包括控制台摘要和保存到 ``results/`` 的图表。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnss_imu import create_range_bearing_ekf  # noqa: E402
from gnss_imu.extended_kalman_filter import (  # noqa: E402
    cv_predict,       # 恒定速度状态转移
    range_bearing_h,  # 距离-方位角观测函数
)


# ── 合成数据生成 ──────────────────────────────────────────────


def generate_straight_line(
    n_steps: int = 150,
    dt: float = 0.1,
    start_pos: tuple[float, float] = (5.0, 2.0),
    velocity: tuple[float, float] = (0.3, 0.15),
) -> tuple[np.ndarray, np.ndarray]:
    """生成恒定速度直线运动的真实轨迹。

    用恒定速度模型逐帧前向传播。

    参数
    ----
    n_steps : int     时间步数
    dt : float        时间步长 [s]
    start_pos : (float, float)  起始位置 (px, py) [m]
    velocity : (float, float)   恒定速度 (vx, vy) [m/s]

    返回
    ----
    true_positions : (n_steps, 2)  真实位置
    true_velocities : (n_steps, 2) 真实速度
    """
    true = np.array(
        [[start_pos[0]], [start_pos[1]], [velocity[0]], [velocity[1]]],
        dtype=float,
    )
    positions = np.zeros((n_steps, 2))
    velocities = np.zeros((n_steps, 2))
    for i in range(n_steps):
        true = cv_predict(true, dt)          # 恒定速度传播
        positions[i] = true[:2].ravel()
        velocities[i] = true[2:].ravel()
    return positions, velocities


# ── 主程序 ───────────────────────────────────────────────────


def main() -> None:
    """运行 2D 距离-方位角 EKF 演示。"""

    # ---- 参数设置 ----
    dt = 0.1                      # 时间步长 [s]
    n_steps = 150                 # 总步数（共 15 秒）
    sensor_pos = (0.0, 0.0)       # 传感器位置（原点）
    range_noise_std = 0.15        # 距离噪声标准差 [m]
    bearing_noise_std = 0.03      # 方位角噪声标准差 [rad]（≈1.7°）
    start_pos = (5.0, 2.0)        # 目标起始位置 [m]
    velocity = (0.3, 0.15)        # 目标速度 [m/s]

    # ---- 生成真实轨迹 ----
    true_pos, true_vel = generate_straight_line(
        n_steps=n_steps, dt=dt,
        start_pos=start_pos, velocity=velocity,
    )

    # ---- 生成带噪声的距离-方位角观测 ----
    rng = np.random.RandomState(42)
    true_state = np.array(
        [[true_pos[0, 0]], [true_pos[0, 1]],
         [true_vel[0, 0]], [true_vel[0, 1]]],
        dtype=float,
    )

    measurements = np.zeros((n_steps, 2))
    for i in range(n_steps):
        true_state = cv_predict(true_state, dt)
        z_true = range_bearing_h(true_state, sensor_pos=sensor_pos)
        # 给距离和方位角分别加噪声
        noise = np.array(
            [
                [rng.randn() * range_noise_std],
                [rng.randn() * bearing_noise_std],
            ]
        )
        z = z_true + noise
        measurements[i] = z.ravel()

    # ---- 创建 EKF ----
    ekf = create_range_bearing_ekf(
        dt=dt,
        initial_position=start_pos,
        initial_velocity=velocity,
        initial_covariance=1.0,
        process_noise_position=0.001,
        process_noise_velocity=0.001,
        measurement_noise_range=range_noise_std,
        measurement_noise_bearing=bearing_noise_std,
        sensor_pos=sensor_pos,
    )

    # ---- 滤波循环 ----
    estimated_positions = np.zeros((n_steps, 2))
    estimated_velocities = np.zeros((n_steps, 2))

    for i in range(n_steps):
        z = measurements[i].reshape(2, 1)
        ekf.step(z)  # 预测 + 更新
        estimated_positions[i] = ekf.x[:2].ravel()
        estimated_velocities[i] = ekf.x[2:].ravel()

    # ---- 控制台摘要 ----
    pos_rmse = np.sqrt(
        np.mean(np.sum((estimated_positions - true_pos) ** 2, axis=1))
    )
    mean_range = np.mean(np.sqrt(np.sum(true_pos**2, axis=1)))

    print("=" * 64)
    print(" 2D 距离-方位角 EKF 演示 (Range-Bearing EKF Demo)")
    print("=" * 64)
    print(f"  轨迹        : 从 {start_pos} 出发, v={velocity}")
    print(f"  传感器位置  : {sensor_pos}")
    print(f"  步数        : {n_steps} @ dt={dt} s  (总计 {n_steps*dt:.0f} s)")
    print(f"  距离噪声 σ  : {range_noise_std} m")
    print(f"  方位角噪声 σ: {bearing_noise_std:.3f} rad "
          f"({np.rad2deg(bearing_noise_std):.1f}°)")
    print("-" * 64)
    print(f"  目标平均距离        : {mean_range:.2f} m")
    print(f"  滤波位置 RMSE       : {pos_rmse:.4f} m")
    print(f"  最终 P 的迹         : {np.trace(ekf.P):.4f}")
    print("=" * 64)

    # ---- 绘图 ----
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib 不可用 — 跳过绘图)")
        return

    os.makedirs("../results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：2D 轨迹 + 传感器 + 采样射线
    ax = axes[0]
    ax.plot(true_pos[:, 0], true_pos[:, 1], "k-", linewidth=1.2,
            label="真实轨迹 (True)")
    ax.plot(
        estimated_positions[:, 0], estimated_positions[:, 1],
        "r--", linewidth=1.2, label="EKF 估计 (EKF estimate)",
    )
    ax.scatter(
        [sensor_pos[0]], [sensor_pos[1]],
        marker="s", s=80, color="blue", zorder=5, label="传感器 (Sensor)",
    )
    # 画几条采样的距离-方位角射线
    for i in range(0, n_steps, 20):
        ax.plot(
            [sensor_pos[0],
             measurements[i, 0] * np.cos(measurements[i, 1])],
            [sensor_pos[1],
             measurements[i, 0] * np.sin(measurements[i, 1])],
            color="gray", alpha=0.2, linewidth=0.5,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("2D 轨迹（距离-方位角 EKF）")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # 右图：位置误差随时间变化
    ax = axes[1]
    est_error = np.sqrt(
        np.sum((estimated_positions - true_pos) ** 2, axis=1)
    )
    t = np.arange(n_steps) * dt
    ax.plot(t, est_error, "r-", linewidth=1.2,
            label="EKF 位置误差")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("位置误差 [m]")
    ax.set_title("EKF 位置误差随时间变化")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "results",
        "demo_ekf_range_bearing.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"\n图表已保存到 {os.path.abspath(out_path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
