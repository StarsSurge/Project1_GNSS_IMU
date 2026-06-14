r"""圆锥效应 (Coning) 图解。

当两个正交轴 (X,Y) 上存在同频率角振动时,
第三轴 (Z) 会产生一个净旋转 —— 即使 Z 轴没有净角速度。

模拟: 绕 X 轴振摆动 + 绕 Y 轴振摆动
      → 计算真实姿态演化 (用连续四元数积分, 细微步长)
      → 对比单子样 (简单相加) 和双子样 (叉积补偿) 的效果

运行: python visual_explanations/coning_error.py
输出: results/coning_error.png
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
               "PingFang SC", "Heiti SC"]
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = next((n for n in _CANDIDATES if n in _available), None)
if _chosen:
    matplotlib.rcParams["font.sans-serif"] = [_chosen, "DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
# 1. Simulate coning motion
# ══════════════════════════════════════════════════════════════════

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def axis_angle_to_quat(axis, angle):
    axis = np.asarray(axis)
    n = np.linalg.norm(axis)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    u = axis / n
    half = angle * 0.5
    return np.array([np.cos(half), u[0]*np.sin(half),
                     u[1]*np.sin(half), u[2]*np.sin(half)])

def quat_to_euler_zxy(q):
    """Quaternion → Roll(X), Pitch(Y), Yaw(Z) [rad], ZXY order for clarity."""
    w, x, y, z = q
    # Roll (X)
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    # Pitch (Y)
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # Yaw (Z)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw

# Parameters
T_total = 2.0           # 2 seconds
dt_fine = 0.0001        # 0.1ms for "truth" (fine integration)
nav_T = 0.01            # 10ms navigation epoch (100 Hz)
omega_freq = 5.0        # 5 Hz vibration frequency
amplitude = 0.02        # 2e-2 rad amplitude on each axis (~1.15°)

t_fine = np.arange(0, T_total, dt_fine)
n_fine = len(t_fine)

# Generate time-varying angular velocity (coning motion):
#   ω_x(t) = A·ω·cos(ωt)           →  angle on X oscillates at ω
#   ω_y(t) = A·ω·cos(ωt + φ)       →  angle on Y oscillates at ω, with phase φ
#   ω_z(t) = 0                       →  no explicit Z rotation
#
# But: cos(ωt) × cos(ωt+φ) ≠ 0 → net rotation about Z!

phi = np.pi / 2   # 90° phase → maximum coning effect

# Compute angular velocity at each fine step
wx = amplitude * omega_freq * np.cos(omega_freq * t_fine)
wy = amplitude * omega_freq * np.cos(omega_freq * t_fine + phi)
wz = np.zeros_like(t_fine)

# Truth: integrate quaternion at fine step (0.1ms) — treat as "true" rotation
q_true = np.array([1.0, 0.0, 0.0, 0.0])
q_true_hist = np.zeros((n_fine, 4))
for i in range(n_fine):
    dtheta = np.array([wx[i], wy[i], wz[i]]) * dt_fine
    angle = np.linalg.norm(dtheta)
    if angle > 1e-15:
        dq = axis_angle_to_quat(dtheta / angle, angle)
    else:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    q_true = quat_multiply(q_true, dq)
    q_true /= np.linalg.norm(q_true)
    q_true_hist[i] = q_true

# Now simulate navigation at 100 Hz using single-sample and two-sample
nav_steps = int(T_total / nav_T)
fine_per_nav = int(nav_T / dt_fine)  # 100 fine steps per nav epoch
half_fine = fine_per_nav // 2

# Single-sample: just use the total angular increment over the epoch
# Two-sample: split into two halves, apply coning compensation

q_single = np.array([1.0, 0.0, 0.0, 0.0])
q_double = np.array([1.0, 0.0, 0.0, 0.0])

single_yaw = np.zeros(nav_steps)
double_yaw = np.zeros(nav_steps)
true_yaw_at_nav = np.zeros(nav_steps)

for k in range(nav_steps):
    i0 = k * fine_per_nav

    # Get the actual angular increments over this nav epoch
    dtheta_total = np.array([
        np.sum(wx[i0:i0+fine_per_nav]) * dt_fine,
        np.sum(wy[i0:i0+fine_per_nav]) * dt_fine,
        np.sum(wz[i0:i0+fine_per_nav]) * dt_fine,
    ])

    # --- Single-sample ---
    angle = np.linalg.norm(dtheta_total)
    if angle > 1e-15:
        dq_s = axis_angle_to_quat(dtheta_total / angle, angle)
    else:
        dq_s = np.array([1.0, 0.0, 0.0, 0.0])
    q_single = quat_multiply(q_single, dq_s)
    q_single /= np.linalg.norm(q_single)
    _, _, yaw_s = quat_to_euler_zxy(q_single)
    single_yaw[k] = yaw_s

    # --- Two-sample (subsample 1 & 2) ---
    dtheta1 = np.array([
        np.sum(wx[i0:i0+half_fine]) * dt_fine,
        np.sum(wy[i0:i0+half_fine]) * dt_fine,
        np.sum(wz[i0:i0+half_fine]) * dt_fine,
    ])
    dtheta2 = np.array([
        np.sum(wx[i0+half_fine:i0+fine_per_nav]) * dt_fine,
        np.sum(wy[i0+half_fine:i0+fine_per_nav]) * dt_fine,
        np.sum(wz[i0+half_fine:i0+fine_per_nav]) * dt_fine,
    ])

    # Coning compensation
    dtheta_2s = dtheta1 + dtheta2 + (2.0/3.0) * np.cross(dtheta1, dtheta2)

    angle = np.linalg.norm(dtheta_2s)
    if angle > 1e-15:
        dq_d = axis_angle_to_quat(dtheta_2s / angle, angle)
    else:
        dq_d = np.array([1.0, 0.0, 0.0, 0.0])
    q_double = quat_multiply(q_double, dq_d)
    q_double /= np.linalg.norm(q_double)
    _, _, yaw_d = quat_to_euler_zxy(q_double)
    double_yaw[k] = yaw_d

    # True yaw at this epoch
    _, _, yaw_t = quat_to_euler_zxy(q_true_hist[i0])
    true_yaw_at_nav[k] = yaw_t

# ══════════════════════════════════════════════════════════════════
# 2. Plot
# ══════════════════════════════════════════════════════════════════

t_nav = np.arange(nav_steps) * nav_T
yaw_rad_to_deg = np.rad2deg(1.0)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

# ── Top-left: phase diagram (X vs Y angle) ──
ax = fig.add_subplot(gs[0, 0])
ax.plot(np.sin(omega_freq * t_fine[:5000]), np.cos(omega_freq * t_fine[:5000] + phi),
        "b-", linewidth=0.6)
ax.set_xlabel("X 轴角度 (归一化)")
ax.set_ylabel("Y 轴角度 (归一化)")
ax.set_title("圆锥运动\nX-Y 角度轨迹: 圆形 → Z 有净旋转")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# ── Top-middle: 3D visualization ──
ax = fig.add_subplot(gs[0, 1], projection="3d")
# Plot the tip of a "cone"
n_cone = 2000
r_cone = 0.15
theta_cone = 2 * np.pi * omega_freq * t_fine[:n_cone]
z_cone = np.cos(theta_cone) * r_cone
x_cone = np.sin(theta_cone) * r_cone * 0.5   # elliptical
y_cone = np.cos(theta_cone + phi) * r_cone * 0.5
ax.plot(x_cone, y_cone, z_cone, "b", linewidth=0.6)
ax.quiver(0, 0, 0, 0, 0, 0.25, color="red", linewidth=2, arrow_length_ratio=0.1)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("圆锥运动 3D 示意图\n红线 = Z 轴: 绕 X,Y 振荡 → Z 轴净旋转")
ax.view_init(elev=25, azim=-45)

# ── Top-right: ω_x vs ω_y ──
ax = fig.add_subplot(gs[0, 2])
# Sample for scatter
skip = 50
ax.scatter(wx[::skip], wy[::skip], c=t_fine[::skip], s=1, alpha=0.5, cmap="viridis")
ax.set_xlabel("ω_x [rad/s]"); ax.set_ylabel("ω_y [rad/s]")
ax.set_title("角速度相图\nω_x 和 ω_y 为同频率正弦波\n叉积 ≠ 0 → Z 轴有净旋转")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# ── Middle-left: yaw drift comparison ──
ax = fig.add_subplot(gs[1, :])
yaw_error_single = (single_yaw - true_yaw_at_nav) * yaw_rad_to_deg
yaw_error_double = (double_yaw - true_yaw_at_nav) * yaw_rad_to_deg
ax.plot(t_nav, yaw_error_single * 3600, "gray", linewidth=1.2,
        label="单子样 (不补偿) — 航向漂移 ~ °/h 级")
ax.plot(t_nav, yaw_error_double * 3600, "r", linewidth=1.2,
        label="双子样 (圆锥补偿后) — 几乎无漂移")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("航向角误差 [°/h]")
ax.set_title(
    "Z 轴航向漂移 (没有 Z 轴角速度输入!)\n"
    f"仅靠 X-Y 角振动 {amplitude*57.3:.1f}° @ {omega_freq}Hz → 产生 °/h 级的虚假航向旋转"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Bottom row: three snapshots showing the coning cross-product ──
for idx, (t0, label) in enumerate([
    (0.1, "早期 — dth1 x dth2 小"),
    (0.3, "中期 — dth1 x dth2 积累"),
    (0.5, "后期 — 漂移明显"),
]):
    i0 = int(t0 / dt_fine)
    i1 = i0 + half_fine
    ax = fig.add_subplot(gs[2, idx])
    # Show ω trajectory in 3D
    d1 = np.array([np.sum(wx[i0:i1])*dt_fine,
                   np.sum(wy[i0:i1])*dt_fine,
                   np.sum(wz[i0:i1])*dt_fine])
    d2 = np.array([np.sum(wx[i1:i1+half_fine])*dt_fine,
                   np.sum(wy[i1:i1+half_fine])*dt_fine,
                   np.sum(wz[i1:i1+half_fine])*dt_fine])
    cross = np.cross(d1, d2)
    ax.arrow(0, 0, d1[0]*100, d1[1]*100,
             color="blue", width=0.0002, head_width=0.0008, label="dth1")
    ax.arrow(0, 0, d2[0]*100, d2[1]*100,
             color="green", width=0.0002, head_width=0.0008, label="dth2")
    ax.arrow(0, 0, cross[0]*10000, cross[1]*10000,
             color="red", width=0.0002, head_width=0.0008, label="dth1 x dth2 x100")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"{label}\n补偿项 Z = {cross[2]*1e6:.1f} µrad")
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle("圆锥效应 (Coning): X-Y 角振动 → Z 轴净旋转",
             fontsize=14, fontweight="bold", y=0.98)

out = os.path.join(os.path.dirname(__file__), "..", "results", "coning_error.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(out)}")
plt.close(fig)
