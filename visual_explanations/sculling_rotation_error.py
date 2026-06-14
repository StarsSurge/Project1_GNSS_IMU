r"""划桨效应 (Sculling) 与旋转效应 (Rotation) 图解。

划桨: 角振动 × 线振动 = 虚假的常值速度增量
      类比 — 划桨: 手摇 (角) + 桨划水 (线) → 船向前 (常值速度)

旋转: 在一个旋转周期内, 速度增量的测量方向在变化
      简单地把起点和终点的测量平均 → 忽略了中间的旋转

运行: python visual_explanations/sculling_rotation_error.py
输出: visual_explanations/outputs/sculling_rotation_error.png
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
# 1. Sculling Simulation
# ══════════════════════════════════════════════════════════════════

T_total = 1.0
dt_fine = 0.0001
nav_T = 0.01
omega_freq = 5.0
amplitude_angle = 0.02   # angular oscillation amp  [rad], ~1.15°
amplitude_vel = 0.05     # velocity oscillation amp [m/s]

t_fine = np.arange(0, T_total, dt_fine)
n_fine = len(t_fine)

# Generate angular oscillation + linear oscillation
wx = amplitude_angle * omega_freq * np.cos(omega_freq * t_fine)
wy = np.zeros_like(t_fine)
wz = np.zeros_like(t_fine)

ax_body = amplitude_vel * omega_freq * np.cos(omega_freq * t_fine + np.pi/2)
ay = np.zeros_like(t_fine)
az = np.zeros_like(t_fine)

# True velocity (integrate at fine step, accounting for rotation during dt)
v_true = np.array([0.0, 0.0, 0.0])      # [vx, vy, vz], start at 0
v_true_hist = np.zeros((n_fine, 3))

# Body-frame attitude (simple integration for truth)
theta_x = np.cumsum(wx) * dt_fine       # accumulate angle about X
for i in range(n_fine):
    # Velocity increment in body frame at this instant
    dv_body = np.array([ax_body[i], ay[i], az[i]]) * dt_fine
    # Rotate to inertial frame (simple small-angle transformation)
    sin_tx = np.sin(theta_x[i])
    cos_tx = np.cos(theta_x[i])
    # The body frame rotated about X by theta_x[i]
    # y and z components get mixed
    dv_nav = np.array([
        dv_body[0],
        cos_tx * dv_body[1] - sin_tx * dv_body[2],
        sin_tx * dv_body[1] + cos_tx * dv_body[2],
    ])
    v_true += dv_nav
    v_true_hist[i] = v_true

# Now simulate navigation at 100 Hz
nav_steps = int(T_total / nav_T)
fine_per_nav = int(nav_T / dt_fine)
half_fine = fine_per_nav // 2

v_single = np.array([0.0, 0.0, 0.0])
v_double = np.array([0.0, 0.0, 0.0])

# Use simple frame (no rotation about X axis), focus on sculling in Y
single_hist = np.zeros((nav_steps, 3))
double_hist = np.zeros((nav_steps, 3))
true_at_nav = np.zeros((nav_steps, 3))

for k in range(nav_steps):
    i0 = k * fine_per_nav

    # Total angular + velocity increments over epoch
    dtheta_total = np.array([
        np.sum(wx[i0:i0+fine_per_nav]) * dt_fine,
        np.sum(wy[i0:i0+fine_per_nav]) * dt_fine,
        np.sum(wz[i0:i0+fine_per_nav]) * dt_fine,
    ])
    dvel_total = np.array([
        np.sum(ax_body[i0:i0+fine_per_nav]) * dt_fine,
        np.sum(ay[i0:i0+fine_per_nav]) * dt_fine,
        np.sum(az[i0:i0+fine_per_nav]) * dt_fine,
    ])

    # --- Single-sample ---
    v_single += dvel_total

    # --- Two-sample ---
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
    dvel1 = np.array([
        np.sum(ax_body[i0:i0+half_fine]) * dt_fine,
        np.sum(ay[i0:i0+half_fine]) * dt_fine,
        np.sum(az[i0:i0+half_fine]) * dt_fine,
    ])
    dvel2 = np.array([
        np.sum(ax_body[i0+half_fine:i0+fine_per_nav]) * dt_fine,
        np.sum(ay[i0+half_fine:i0+fine_per_nav]) * dt_fine,
        np.sum(az[i0+half_fine:i0+fine_per_nav]) * dt_fine,
    ])

    # Sculling compensation
    dvel_scul = (2.0/3.0) * (np.cross(dtheta1, dvel2) + np.cross(dvel1, dtheta2))
    # Rotation effect compensation
    dvel_rot = 0.5 * np.cross(dtheta_total, dvel_total)
    dvel_2s = dvel_total + dvel_scul + dvel_rot

    v_double += dvel_2s

    single_hist[k] = v_single
    double_hist[k] = v_double
    true_at_nav[k] = v_true_hist[i0]

# ══════════════════════════════════════════════════════════════════
# 2. Plot
# ══════════════════════════════════════════════════════════════════

t_nav = np.arange(nav_steps) * nav_T
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

# ── (0,0): Sculling physical analogy ──
ax = fig.add_subplot(gs[0, 0])
# Draw a simple boat + oar cartoon
theta = np.linspace(-0.3, 0.3, 20)
boat_x = np.array([-0.2, 0.2])
ax.plot(boat_x, [0, 0], "k-", linewidth=3, label="船体")
ax.arrow(-0.15, 0, 0, 0.25, color="green", width=0.01, head_width=0.03, label="手摇 (角)")
ax.arrow(-0.15, 0.25, 0.25, 0, color="blue", width=0.005, head_width=0.02, label="桨划 (线)")
ax.arrow(0.1, 0.05, 0.4, 0, color="red", width=0.005, head_width=0.02, label="船前进 (常值速度)")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-0.5, 0.8); ax.set_ylim(-0.1, 0.4)
ax.set_title("划桨效应 物理类比\n手摇 (角振动) + 桨划水 (线振动)\n→ 船获得常值前进速度")
ax.legend(fontsize=7, loc="lower right")

# ── (0,1): ω_x and a_y vs time ──
ax = fig.add_subplot(gs[0, 1])
ax2 = ax.twinx()
ax.plot(t_fine[:3000], wx[:3000] * 57.3, "red", linewidth=0.8, label="ω_x (角速度)")
ax2.plot(t_fine[:3000], ax_body[:3000], "blue", linewidth=0.8, label="a_y (线加速度)")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("角速度 [°/s]", color="red")
ax2.set_ylabel("加速度 [m/s²]", color="blue")
ax.set_title("角振动 (X 轴) + 线振动 (Y 轴)\n同频率, 90° 相位差 → 最大划桨效应")

# ── (0,2): Cross-product surface ──
t_show = np.arange(0, 0.5, nav_T)
cross_vals = []
for t0 in t_show:
    i0 = int(t0 / dt_fine)
    i1 = i0 + half_fine
    d1 = np.array([np.sum(wx[i0:i1])*dt_fine, 0.0, 0.0])
    d2 = np.array([np.sum(wx[i1:i1+half_fine])*dt_fine, 0.0, 0.0])
    dv1 = np.array([0.0, np.sum(ay[i0:i1])*dt_fine, 0.0])
    dv2 = np.array([0.0, np.sum(ay[i1:i1+half_fine])*dt_fine, 0.0])
    scul = (2/3)*(np.cross(d1, dv2) + np.cross(dv1, d2))
    cross_vals.append(scul[2])  # Z component of sculling

ax = fig.add_subplot(gs[0, 2])
ax.bar(range(len(t_show)), np.array(cross_vals)*1e6, color="purple", alpha=0.7)
ax.set_xlabel("导航历元"); ax.set_ylabel("划桨补偿项 Z [µm/s]")
ax.set_title("每个导航历元的划桨补偿 (dv_scul)_z\n"
             "叉积 dv1 x dth2 + dth1 x dv2 贡献了 Z 方向的虚假速度")
ax.grid(axis="y", alpha=0.3)

# ── (1,:): Velocity error comparison ──
ax = fig.add_subplot(gs[1, :])
# Only Y-axis has sculling effect (X oscillation → Y velocity)
vel_error_single_y = (single_hist[:, 1] - true_at_nav[:, 1]) * 100  # cm/s
vel_error_double_y = (double_hist[:, 1] - true_at_nav[:, 1]) * 100
ax.plot(t_nav, vel_error_single_y, "gray", linewidth=1.2,
        label="单子样 (不补偿) — 速度累积偏差")
ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
ax.plot(t_nav, vel_error_double_y, "r", linewidth=1.2,
        label="双子样 (划桨+旋转补偿后) — 几乎无偏差")
ax.fill_between(t_nav, 0, vel_error_single_y, alpha=0.1, color="gray")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("Y 轴速度误差 [cm/s]")
ax.set_title(
    "划桨效应 → Y 轴速度误差 (仅有 X 轴角振动 + Y 轴线振动!)\n"
    f"角振幅 {amplitude_angle*57.3:.1f}° @ {omega_freq}Hz, "
    f"线振幅 {amplitude_vel:.2f} m/s² @ {omega_freq}Hz"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── (2,0): Per-epoch compensation breakdown ──
epoch_show = 20
scul_z = np.zeros(epoch_show)
rot_z = np.zeros(epoch_show)
for k in range(epoch_show):
    i0 = k * fine_per_nav
    dtheta_total = np.array([np.sum(wx[i0:i0+fine_per_nav])*dt_fine, 0., 0.])
    dvel_total = np.array([0., np.sum(ay[i0:i0+fine_per_nav])*dt_fine, 0.])
    dtheta1 = np.array([np.sum(wx[i0:i0+half_fine])*dt_fine, 0., 0.])
    dtheta2 = np.array([np.sum(wx[i0+half_fine:i0+fine_per_nav])*dt_fine, 0., 0.])
    dvel1 = np.array([0., np.sum(ay[i0:i0+half_fine])*dt_fine, 0.])
    dvel2 = np.array([0., np.sum(ay[i0+half_fine:i0+fine_per_nav])*dt_fine, 0.])
    scul = (2/3)*(np.cross(dtheta1, dvel2) + np.cross(dvel1, dtheta2))
    rot = 0.5 * np.cross(dtheta_total, dvel_total)
    scul_z[k] = scul[2] * 1e9
    rot_z[k] = rot[2] * 1e9

ax = fig.add_subplot(gs[2, 0])
x_idx = np.arange(epoch_show)
w = 0.35
ax.bar(x_idx - w/2, scul_z, w, color="purple", alpha=0.7, label="划桨 (2/3)(dth1 x dv2 + dv1 x dth2)")
ax.bar(x_idx + w/2, rot_z, w, color="orange", alpha=0.7, label="旋转 (1/2)(dθ×dv)")
ax.set_xlabel("导航历元"); ax.set_ylabel("补偿量 [nm/s]")
ax.set_title("每历元的划桨 vs 旋转补偿量\n划桨 ≈ 2× 旋转 (此参数组合下)")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)

# ── (2,1): Rotation effect explanation ──
ax = fig.add_subplot(gs[2, 1])
# Draw a rotating frame illustration
theta_rot = np.linspace(0, np.pi/4, 50)
r = 0.3
x_circle = r * np.cos(theta_rot)
y_circle = r * np.sin(theta_rot)
ax.plot(x_circle, y_circle, "gray", linewidth=0.8, linestyle="--")
ax.arrow(0.5, 0, 0, 0, color="blue", width=0.005, head_width=0.012, label="起点 body 系")
ax.arrow(0.44, 0.12, 0.02, 0.08, color="red", width=0.005, head_width=0.012,
         label="途中 body 系 (在转!)")
# Frame axes at start
ax.arrow(0.5, 0, 0.1, 0, color="black", width=0.003)
ax.arrow(0.5, 0, 0, 0.1, color="black", width=0.003)
# Frame axes at mid-rotation
mid_x, mid_y = 0.44, 0.12
ax.arrow(mid_x, mid_y, 0.08, 0.04, color="gray", width=0.003)
ax.arrow(mid_x, mid_y, -0.02, 0.08, color="gray", width=0.003)
ax.set_xlim(0.2, 0.7); ax.set_ylim(-0.1, 0.4)
ax.set_aspect("equal")
for spine in ax.spines.values(): spine.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("旋转效应\nbody 系在 dt 内旋转 →\ndv 的方向不是恒定的\n简单相加忽略了方向变化")
ax.legend(fontsize=7)

# ── (2,2): Position error accumulation ──
ax = fig.add_subplot(gs[2, 2])
pos_single_y = np.cumsum(single_hist[:, 1]) * nav_T * 100  # cm
pos_double_y = np.cumsum(double_hist[:, 1]) * nav_T * 100
pos_true_y = np.cumsum(true_at_nav[:, 1]) * nav_T * 100
ax.plot(t_nav, (pos_single_y - pos_true_y), "gray", linewidth=1.2, label="单子样")
ax.plot(t_nav, (pos_double_y - pos_true_y), "r", linewidth=1.2, label="双子样")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("Y 位置误差 [cm]")
ax.set_title("位置漂移 (速度误差的积分)\n1 秒内单子样漂 ~cm 级, 双子样几乎零")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("划桨效应 (Sculling) 与旋转效应 (Rotation): 角振动 × 线振动 → 虚假的常值速度",
             fontsize=14, fontweight="bold", y=0.98)

out = os.path.join(os.path.dirname(__file__), "outputs",
                   "sculling_rotation_error.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(out)}")
plt.close(fig)
