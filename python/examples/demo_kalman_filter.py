r"""2D 恒定速度卡尔曼滤波演示。

Demo: 2D constant-velocity Kalman filter.

生成一条合成圆形轨迹，加上带噪的位置观测，
然后用通用 ``KalmanFilter`` 恢复底层状态。

运行方式（从项目根目录）::

    $env:PYTHONPATH = "$PWD\python"
    python python\examples\demo_kalman_filter.py

输出包括控制台摘要和保存到 ``results/`` 的图表。
"""

from __future__ import annotations

import os
import sys

import numpy as np

# 确保以脚本方式运行时能导入 gnss_imu 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnss_imu import create_constant_velocity_filter_nd  # noqa: E402


# ── 合成数据生成 ──────────────────────────────────────────────


def generate_circular_trajectory(
    n_steps: int = 200,
    dt: float = 0.1,
    radius: float = 5.0,
    angular_velocity: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """生成匀速圆周运动的真实轨迹（位置 + 速度）。

    运动学：
        px(t) = r·cos(ωt)      vx(t) = -rω·sin(ωt)
        py(t) = r·sin(ωt)      vy(t) =  rω·cos(ωt)

    参数
    ----
    n_steps : int     时间步数
    dt : float        时间步长 [s]
    radius : float    圆的半径 [m]
    angular_velocity : float  角速度 [rad/s]

    返回
    ----
    true_positions : (n_steps, 2)  每个时间步的真实 (px, py)
    true_velocities : (n_steps, 2) 每个时间步的真实 (vx, vy)
    """
    t = np.arange(n_steps) * dt
    theta = angular_velocity * t
    px = radius * np.cos(theta)
    py = radius * np.sin(theta)
    vx = -radius * angular_velocity * np.sin(theta)
    vy = radius * angular_velocity * np.cos(theta)
    return (
        np.column_stack([px, py]),
        np.column_stack([vx, vy]),
    )


# ── 主程序 ───────────────────────────────────────────────────


def main() -> None:
    """运行 2D KF 演示。"""

    # ---- 参数设置 ----
    dt = 0.1                     # 时间步长 [s]
    n_steps = 200                # 总步数（共 20 秒）
    measurement_noise_std = 0.5  # 位置观测噪声标准差 [m]

    # ---- 生成真实轨迹 ----
    true_pos, true_vel = generate_circular_trajectory(
        n_steps=n_steps, dt=dt, radius=5.0, angular_velocity=0.3
    )

    # ---- 生成带噪声的观测（仅位置） ----
    # 观测方程（线性）：
    #     z = H @ x + v,   v ~ N(0, σ²·I)
    #
    #     ┌     ┐   ┌                ┐ ┌     ┐
    #     │ z_x │ = │ 1  0  0  0 │   │ px  │   ┌     ┐
    #     │ z_y │   │ 0  1  0  0 │   │ py  │ + │ v_x │   仅观测位置
    #     └     ┘   └                ┘ │ vx  │   │ v_y │   速度由滤波器推断
    #                                  │ vy  │   └     ┘
    #                                  └     ┘
    #     H = [I₂, 0₂]  提取前 2 个状态分量（位置）
    rng = np.random.RandomState(42)  # 固定种子，保证可复现
    noisy_pos = true_pos + rng.randn(n_steps, 2) * measurement_noise_std

    # ---- 创建 2D 恒定速度卡尔曼滤波器 ----
    # 仅观测位置，速度由滤波器内部估计
    kf = create_constant_velocity_filter_nd(
        dim=2,
        dt=dt,
        initial_position=true_pos[0],
        initial_velocity=true_vel[0],
        initial_covariance=1.0,
        process_noise_position=0.01,
        process_noise_velocity=0.01,
        measurement_noise=measurement_noise_std**2,  # 传入方差
    )

    # ---- 滤波循环 ----
    estimated_positions = np.zeros((n_steps, 2))
    estimated_velocities = np.zeros((n_steps, 2))

    for i in range(n_steps):
        z = noisy_pos[i].reshape(2, 1)  # 观测转为列向量
        kf.step(z)                       # 预测 + 更新
        estimated_positions[i] = kf.x[:2].ravel()
        estimated_velocities[i] = kf.x[2:].ravel()

    # ---- 控制台摘要 ----
    # RMSE = root mean squared error（均方根误差）
    pos_rmse = np.sqrt(
        np.mean(np.sum((estimated_positions - true_pos) ** 2, axis=1))
    )
    raw_rmse = np.sqrt(
        np.mean(np.sum((noisy_pos - true_pos) ** 2, axis=1))
    )

    print("=" * 64)
    print(" 2D 卡尔曼滤波演示 (2D Kalman Filter Demo)")
    print("=" * 64)
    print(f"  轨迹        : 圆形, r=5 m, ω=0.3 rad/s")
    print(f"  步数        : {n_steps} @ dt={dt} s  (总计 {n_steps*dt:.0f} s)")
    print(f"  观测噪声 σ  : {measurement_noise_std} m")
    print("-" * 64)
    print(f"  原始观测 RMSE  : {raw_rmse:.4f} m")
    print(f"  滤波估计 RMSE  : {pos_rmse:.4f} m")
    print(f"  改善           : {raw_rmse - pos_rmse:.4f} m  "
          f"({(1 - pos_rmse/raw_rmse) * 100:.1f}%)")
    print(f"  最终 P 的迹    : {np.trace(kf.P):.4f}")
    print("=" * 64)

    # ---- 绘图 ----
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        print("\n(matplotlib 不可用 — 跳过绘图)")
        return

    # ── 配置中文字体 ──
    _CHINESE_CANDIDATES = [
        "Microsoft YaHei", "SimHei", "KaiTi",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei",
        "PingFang SC", "Heiti SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((n for n in _CHINESE_CANDIDATES if n in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        print(f"[font] 使用中文字体: {chosen}")
    else:
        print("[font] 未找到中文字体，图表中文可能乱码")
    plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示

    os.makedirs("../results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：2D 轨迹对比
    ax = axes[0]
    ax.plot(true_pos[:, 0], true_pos[:, 1], "k-", linewidth=1,
            label="真实轨迹 (True)")
    ax.scatter(
        noisy_pos[::5, 0], noisy_pos[::5, 1],
        s=8, alpha=0.4, color="gray", label="带噪观测 (Noisy)",
    )
    ax.plot(
        estimated_positions[:, 0], estimated_positions[:, 1],
        "r--", linewidth=1.2, label="KF 估计 (Filtered)",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("2D 轨迹 (Trajectory)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # 右图：位置误差随时间变化
    ax = axes[1]
    est_error = np.sqrt(
        np.sum((estimated_positions - true_pos) ** 2, axis=1)
    )
    raw_error = np.sqrt(np.sum((noisy_pos - true_pos) ** 2, axis=1))
    t = np.arange(n_steps) * dt
    ax.plot(t, raw_error, color="gray", alpha=0.5, linewidth=0.8,
            label="原始误差 (Raw error)")
    ax.plot(t, est_error, "r-", linewidth=1.2,
            label="KF 误差 (KF error)")
    ax.set_xlabel("时间 [s]")
    ax.set_ylabel("位置误差 [m]")
    ax.set_title("误差随时间变化 (Error over time)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "demo_kf_2d.png"
    )
    fig.savefig(out_path, dpi=150)
    print(f"\n图表已保存到 {os.path.abspath(out_path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
