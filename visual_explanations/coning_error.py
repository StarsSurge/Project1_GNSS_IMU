r"""Coning error visualized with a fine-integration attitude truth model."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

from imu_visualization_math import (
    attitude_error_rotvec,
    coning_correct,
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
)

matplotlib.use("Agg")
_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
_AVAILABLE = {font.name for font in fm.fontManager.ttflist}
_FONT = next((name for name in _CANDIDATES if name in _AVAILABLE), None)
if _FONT:
    matplotlib.rcParams["font.sans-serif"] = [_FONT, "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(os.path.join(OUT, filename), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


# A circular angular-rate excitation. Frequency is explicitly in Hz.
duration = 4.0
fine_dt = 0.0002
nav_dt = 0.01
frequency_hz = 3.0
angular_frequency = 2.0 * np.pi * frequency_hz
cone_angle = np.deg2rad(2.0)
t = np.arange(0.0, duration, fine_dt)
omega_x = cone_angle * angular_frequency * np.cos(angular_frequency * t)
omega_y = cone_angle * angular_frequency * np.sin(angular_frequency * t)
omega_body = np.column_stack([omega_x, omega_y, np.zeros_like(t)])

# Fine quaternion integration is the numerical truth.
q_truth = np.array([1.0, 0.0, 0.0, 0.0])
truth_history = np.empty((t.size, 4))
for index, omega in enumerate(omega_body):
    q_truth = quat_multiply(q_truth, rotvec_to_quat(omega * fine_dt))
    q_truth /= np.linalg.norm(q_truth)
    truth_history[index] = q_truth

samples_per_nav = int(round(nav_dt / fine_dt))
half_samples = samples_per_nav // 2
nav_steps = t.size // samples_per_nav
nav_time = (np.arange(nav_steps) + 1) * nav_dt

q_sum = np.array([1.0, 0.0, 0.0, 0.0])
q_two_sample = q_sum.copy()
error_sum = np.zeros((nav_steps, 3))
error_two_sample = np.zeros((nav_steps, 3))
cross_terms = np.zeros((nav_steps, 3))

for epoch in range(nav_steps):
    start = epoch * samples_per_nav
    middle = start + half_samples
    end = start + samples_per_nav
    dtheta1 = omega_body[start:middle].sum(axis=0) * fine_dt
    dtheta2 = omega_body[middle:end].sum(axis=0) * fine_dt

    q_sum = quat_multiply(q_sum, rotvec_to_quat(dtheta1 + dtheta2))
    q_sum /= np.linalg.norm(q_sum)

    corrected = coning_correct(dtheta1, dtheta2)
    q_two_sample = quat_multiply(q_two_sample, rotvec_to_quat(corrected))
    q_two_sample /= np.linalg.norm(q_two_sample)

    truth_at_end = truth_history[end - 1]
    error_sum[epoch] = attitude_error_rotvec(q_sum, truth_at_end)
    error_two_sample[epoch] = attitude_error_rotvec(
        q_two_sample, truth_at_end
    )
    cross_terms[epoch] = (2.0 / 3.0) * np.cross(dtheta1, dtheta2)

arcsec_per_rad = np.rad2deg(1.0) * 3600.0
z_error_sum_arcsec = error_sum[:, 2] * arcsec_per_rad
z_error_two_arcsec = error_two_sample[:, 2] * arcsec_per_rad
cross_z_arcsec = cross_terms[:, 2] * arcsec_per_rad
equivalent_rate_sum = abs(z_error_sum_arcsec[-1]) / duration
equivalent_rate_two = abs(z_error_two_arcsec[-1]) / duration
improvement = equivalent_rate_sum / max(equivalent_rate_two, 1e-12)


# ---------------------------------------------------------------------------
# Figure 1: mechanism
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

window = t <= 1.0
ax = axes[0, 0]
ax.plot(t[window], np.rad2deg(omega_x[window]), label="ωx")
ax.plot(t[window], np.rad2deg(omega_y[window]), label="ωy")
ax.plot(t[window], np.zeros(window.sum()), "k--", lw=0.8, label="ωz = 0")
ax.set_title("① 两个正交角速度分量相差 90°")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("角速度 [deg/s]")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[0, 1]
ax.plot(
    omega_x[window] / np.max(abs(omega_x)),
    omega_y[window] / np.max(abs(omega_y)),
    color="tab:blue",
)
ax.set_aspect("equal", adjustable="box")
ax.set_title("② ωx-ωy 相位轨迹是圆：旋转轴绕锥面运动")
ax.set_xlabel("归一化 ωx")
ax.set_ylabel("归一化 ωy")
ax.grid(alpha=0.25)

ax = axes[1, 0]
d1 = np.array([1.0, 0.15])
d2 = np.array([-0.1, 0.9])
ax.quiver(0, 0, *d1, angles="xy", scale_units="xy", scale=1, color="tab:red")
ax.quiver(
    d1[0], d1[1], *d2, angles="xy", scale_units="xy", scale=1, color="tab:blue"
)
ax.quiver(
    0, 0, *(d1 + d2), angles="xy", scale_units="xy", scale=1, color="0.45"
)
ax.text(0.48, 0.02, "Δθ_1", color="tab:red")
ax.text(0.82, 0.62, "Δθ_2", color="tab:blue")
ax.text(0.30, 0.72, "简单向量和", color="0.35")
ax.text(
    0.02,
    0.96,
    "有限旋转不满足交换律：\n"
    "Exp(Δθ_1) Exp(Δθ_2) ≠ Exp(Δθ_1+Δθ_2)\n\n"
    "两子样线性角速度近似：\n"
    "Δθ_corr = Δθ_1 + Δθ_2 + 2/3(Δθ_1×Δθ_2)\n\n"
    "叉积沿第三轴，补偿被向量相加遗漏的二阶旋转。",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
)
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-0.2, 1.3)
ax.set_aspect("equal", adjustable="box")
ax.set_title("③ 向量相加遗漏旋转次序")
ax.grid(alpha=0.2)

ax = axes[1, 1]
epochs = np.arange(min(80, nav_steps))
ax.bar(epochs, cross_z_arcsec[: epochs.size], color="tab:purple", width=0.8)
ax.axhline(0.0, color="black", lw=0.7)
ax.set_title("④ 每个导航历元的圆锥补偿量")
ax.set_xlabel("导航历元")
ax.set_ylabel("Z 分量 [arcsec]")
ax.grid(axis="y", alpha=0.25)

fig.suptitle(
    "圆锥误差：角增量的不可交换性与两子样交叉项",
    fontsize=15,
    fontweight="bold",
)
save(fig, "coning_1_mechanism.png")


# ---------------------------------------------------------------------------
# Figure 2: spatial interpretation
# ---------------------------------------------------------------------------

stride = max(1, truth_history.shape[0] // 1800)
body_z_nav = np.array(
    [quat_to_dcm(q)[:, 2] for q in truth_history[::stride]]
)

fig = plt.figure(figsize=(15, 8), constrained_layout=True)
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(
    body_z_nav[:, 0],
    body_z_nav[:, 1],
    body_z_nav[:, 2],
    color="tab:blue",
    lw=1.0,
)
ax.scatter(
    body_z_nav[0, 0],
    body_z_nav[0, 1],
    body_z_nav[0, 2],
    color="tab:green",
    label="起点",
)
ax.scatter(
    body_z_nav[-1, 0],
    body_z_nav[-1, 1],
    body_z_nav[-1, 2],
    color="tab:red",
    label="终点",
)
ax.set_xlabel("N")
ax.set_ylabel("E")
ax.set_zlabel("D")
ax.set_title("Body Z 轴在导航系单位球上的轨迹")
ax.legend()
ax.view_init(elev=22, azim=-55)

ax = fig.add_subplot(1, 2, 2)
ax.axis("off")
ax.text(
    0.03,
    0.96,
    "空间直觉\n\n"
    "旋转轴在 X-Y 平面连续绕行，body 轴尖端形成闭合锥形轨迹。\n"
    "即使 ωz=0，一周有限旋转的乘积仍可能包含 Z 方向净旋转。\n\n"
    "这不是传感器 bias，而是离散算法遗漏的二阶项：\n"
    "• 提高 IMU 采样率会减小它\n"
    "• 缩短导航更新周期会减小它\n"
    "• 两子样/多子样算法显式补偿它\n\n"
    "适用前提\n"
    "2/3 系数来自两个等时间子样以及角速度在周期内线性变化假设。\n"
    "它不是任意采样结构下都通用的魔法常数。",
    transform=ax.transAxes,
    va="top",
    fontsize=12,
    linespacing=1.5,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
)
ax.text(
    0.03,
    0.25,
    f"本实验：锥角幅值 {np.rad2deg(cone_angle):.1f} deg，"
    f"频率 {frequency_hz:.1f} Hz\n"
    f"导航周期 {nav_dt*1000:.0f} ms，真值积分步长 {fine_dt*1e6:.0f} us",
    transform=ax.transAxes,
    fontsize=11,
)

fig.suptitle(
    "圆锥运动的空间解释：闭合轴轨迹不代表姿态乘积完全闭合",
    fontsize=15,
    fontweight="bold",
)
save(fig, "coning_2_spatial.png")


# ---------------------------------------------------------------------------
# Figure 3: numerical impact
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)

ax = axes[0]
ax.plot(nav_time, z_error_sum_arcsec, color="0.45", label="简单相加")
ax.plot(nav_time, z_error_two_arcsec, color="tab:red", label="两子样补偿")
ax.set_title("相对细积分真值的 Z 姿态误差")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("姿态误差 [arcsec]")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1]
cumulative_correction = -np.cumsum(cross_z_arcsec)
error_reduction = z_error_sum_arcsec - z_error_two_arcsec
ax.plot(nav_time, cumulative_correction, color="tab:purple", label="预测的未补偿误差")
ax.plot(nav_time, error_reduction, "k--", label="实际误差改善量")
ax.set_title("交叉项同时解释误差大小与符号")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("角度 [arcsec]")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[2]
ax.bar(
    ["简单相加", "两子样"],
    [equivalent_rate_sum, equivalent_rate_two],
    color=["0.5", "tab:red"],
)
ax.set_yscale("log")
ax.set_ylabel("末端等效漂移率 [deg/h]")
ax.set_title("末端角误差 / 实验时长")
ax.grid(axis="y", alpha=0.25)
ax.text(
    0.04,
    0.96,
    f"改善倍数：{improvement:.1f}×\n\n"
    "注意：曲线纵轴是角度 arcsec，\n"
    "只有末端角误差除以时长后，\n"
    "才可报告为等效 deg/h。\n\n"
    "两子样只消除主导二阶项，\n"
    "并非在任意高动态下误差为零。",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

fig.suptitle(
    "圆锥补偿的定量验证：细积分真值、角度误差与等效漂移率",
    fontsize=15,
    fontweight="bold",
)
save(fig, "coning_3_impact.png")

print(
    "Coning equivalent drift [deg/h]: "
    f"simple={equivalent_rate_sum:.6f}, corrected={equivalent_rate_two:.6f}; "
    f"improvement={improvement:.1f}x"
)
