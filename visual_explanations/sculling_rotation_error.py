r"""划桨效应 (Sculling) 与旋转效应 (Rotation) 图解。

划桨: 角振动 × 线振动 → 正交方向的虚假常值速度
旋转: dt 内 body 系旋转 → 速度增量方向变化未补偿

运行: python visual_explanations/sculling_rotation_error.py
输出: visual_explanations/outputs/sculling_rotation_error.png
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
               "PingFang SC", "Heiti SC"]
_avail = {f.name for f in fm.fontManager.ttflist}
_ch = next((n for n in _CANDIDATES if n in _avail), None)
if _ch:
    matplotlib.rcParams["font.sans-serif"] = [_ch, "DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

# ── simulation ───────────────────────────────────────────────────
# sculling: X-axis angular oscillation + Z-axis linear oscillation
#   → produces Y-axis net velocity (dth_x × dv_z = Y direction)
# rotation: total rotation during dt changes direction of dv

T, dt_fine, nav_T = 1.0, 0.0001, 0.01
omega = 10.0          # 10 Hz vibration (clearly visible)
amp_ang = 0.05        # angular amp [rad] ≈ 2.9°
amp_vel = 0.5         # velocity amp [m/s] (larger for visible effect)

t = np.arange(0, T, dt_fine)
n_fine = len(t)

# X-axis angular oscillation
wx = amp_ang * omega * np.cos(omega * t)
wy = np.zeros_like(t)
wz = np.zeros_like(t)

# Z-axis linear oscillation (90° phase → max sculling)
az = amp_vel * omega * np.cos(omega * t + np.pi/2)
ax = np.zeros_like(t)
ay = np.zeros_like(t)

# Truth: integrate at fine dt, account for body rotation
theta_x = np.cumsum(wx) * dt_fine
v_true = np.zeros(3)
v_true_hist = np.zeros((n_fine, 3))
for i in range(n_fine):
    dv_body = np.array([ax[i], ay[i], az[i]]) * dt_fine
    ct, st = np.cos(theta_x[i]), np.sin(theta_x[i])
    dv_nav = np.array([dv_body[0],
                       ct*dv_body[1] - st*dv_body[2],
                       st*dv_body[1] + ct*dv_body[2]])
    v_true += dv_nav
    v_true_hist[i] = v_true

# navigation at 100 Hz
steps = int(T / nav_T)
fpn = int(nav_T / dt_fine)
hf = fpn // 2

v_single = np.zeros(3)
v_double = np.zeros(3)
s_hist = np.zeros((steps, 3))
d_hist = np.zeros((steps, 3))
vt_hist = np.zeros((steps, 3))

for k in range(steps):
    i0 = k * fpn

    dth_t = np.array([np.sum(wx[i0:i0+fpn])*dt_fine, 0., 0.])
    dv_t  = np.array([0., 0., np.sum(az[i0:i0+fpn])*dt_fine])

    # single
    v_single += dv_t

    # two-sample
    dth1 = np.array([np.sum(wx[i0:i0+hf])*dt_fine, 0., 0.])
    dth2 = np.array([np.sum(wx[i0+hf:i0+fpn])*dt_fine, 0., 0.])
    dv1  = np.array([0., 0., np.sum(az[i0:i0+hf])*dt_fine])
    dv2  = np.array([0., 0., np.sum(az[i0+hf:i0+fpn])*dt_fine])

    scul = (2./3.)*(np.cross(dth1, dv2) + np.cross(dv1, dth2))
    rot  = 0.5 * np.cross(dth_t, dv_t)
    v_double += dv_t + scul + rot

    s_hist[k] = v_single
    d_hist[k] = v_double
    vt_hist[k] = v_true_hist[i0]

# Sculling produces Y-axis velocity (dth_x × dv_z → Y)
tn = np.arange(steps) * nav_T
err_s = (s_hist[:, 1] - vt_hist[:, 1]) * 1000   # mm/s
err_d = (d_hist[:, 1] - vt_hist[:, 1]) * 1000

# cross-product diagnostics
scul_y = np.zeros(steps)
rot_y = np.zeros(steps)
for k in range(steps):
    i0 = k*fpn
    dth1 = np.array([np.sum(wx[i0:i0+hf])*dt_fine, 0., 0.])
    dth2 = np.array([np.sum(wx[i0+hf:i0+fpn])*dt_fine, 0., 0.])
    dv1  = np.array([0., 0., np.sum(az[i0:i0+hf])*dt_fine])
    dv2  = np.array([0., 0., np.sum(az[i0+hf:i0+fpn])*dt_fine])
    dth_t = dth1+dth2; dv_t = dv1+dv2
    scul_y[k] = (2./3.)*(np.cross(dth1, dv2)+np.cross(dv1, dth2))[1]*1000
    rot_y[k]  = 0.5*np.cross(dth_t, dv_t)[1]*1000

# ── 2x3 layout ───────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
((ax0, ax1, ax2), (ax3, ax4, ax5)) = axes

# (0,0) ω_x and a_z signals
ax = ax0
t_show = t[:2000]
ax2_ = ax.twinx()
ax.plot(t_show, wx[:2000]*57.3, "red", lw=0.8, label="ω_x (角速度)")
ax2_.plot(t_show, az[:2000], "blue", lw=0.8, label="a_z (线加速度)")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("角速度 [°/s]", color="red")
ax2_.set_ylabel("加速度 [m/s²]", color="blue")
ax.set_title("X 轴角振动 + Z 轴线振动", fontsize=10)

# (0,1) Sculling comp per epoch (mm/s scale)
ax = ax1
x_idx = np.arange(min(30, steps))
w = 0.35
ax.bar(x_idx-w/2, scul_y[:30], w, color="purple", alpha=0.7, label="划桨 (2/3)(dth1×dv2+dv1×dth2)")
ax.bar(x_idx+w/2, rot_y[:30], w, color="orange", alpha=0.7, label="旋转 (1/2)(dth×dv)")
ax.set_xlabel("导航历元"); ax.set_ylabel("补偿量 Y [mm/s]")
ax.set_title("每历元划桨 vs 旋转补偿量", fontsize=10)
ax.legend(fontsize=6); ax.grid(axis="y", alpha=0.3)

# (0,2) Physical explanation
ax = ax2
ax.axis("off")
msg = (
    "划桨效应 — 物理类比\n\n"
    "手摇船桨 (角振动)\n"
    "桨片划水 (线振动)\n"
    "→ 船获得常值前进速度\n\n"
    "数学: dth_x × dv_z → Y 方向速度\n\n"
    f"角振幅: {amp_ang*57.3:.1f}°\n"
    f"线振幅: {amp_vel:.2f} m/s²\n"
    f"频率: {omega} Hz"
)
ax.text(0.05, 0.95, msg, transform=ax.transAxes, fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# (1,0) Velocity error — Y axis
ax = ax3
ax.plot(tn, err_s, "gray", lw=1.2, label="单子样 (不补偿)")
ax.plot(tn, err_d, "r", lw=1.2, label="双子样 (划桨+旋转补偿)")
ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.set_xlabel("时间 [s]"); ax.set_ylabel("Y 轴速度误差 [mm/s]")
ax.set_title("划桨 → Y 轴速度误差", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (1,1) Position drift (integrate velocity error)
ax = ax4
pos_s = np.cumsum(err_s) * nav_T * 1e-3   # m
pos_d = np.cumsum(err_d) * nav_T * 1e-3
ax.plot(tn, pos_s*1000, "gray", lw=1.2, label="单子样")
ax.plot(tn, pos_d*1000, "r", lw=1.2, label="双子样")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("Y 位置漂移 [mm]")
ax.set_title("位置漂移 (速度误差积分)", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (1,2) Rotation effect diagram
ax = ax5
theta_demo = np.linspace(0, np.pi/6, 50)
r = 0.3
ax.plot(r*np.cos(theta_demo), r*np.sin(theta_demo), "gray", lw=0.8, ls="--")
# body frame at start
ax.arrow(0.55, 0, 0.12, 0, color="black", width=0.003, head_width=0.008)
ax.arrow(0.55, 0, 0, 0.12, color="black", width=0.003, head_width=0.008)
ax.text(0.70, 0.03, "body(t)", fontsize=7)
# body frame mid
cx, cy = 0.52, 0.06
ax.arrow(cx, cy, 0.10, 0.03, color="gray", width=0.003, head_width=0.008)
ax.arrow(cx, cy, -0.01, 0.11, color="gray", width=0.003, head_width=0.008)
ax.text(cx+0.08, cy+0.03, "body(t+dt/2)", fontsize=7, color="gray")
# dv arrow
ax.arrow(0.55, -0.05, -0.05, 0.15, color="red", width=0.004, head_width=0.012, length_includes_head=True)
ax.text(0.40, 0.10, "dv 方向在变!", fontsize=7, color="red")
ax.set_xlim(0.3, 0.8); ax.set_ylim(-0.1, 0.3)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect("equal")
ax.set_title("旋转效应: body系在dt内旋转", fontsize=10)

fig.suptitle("划桨效应 (Sculling) 与旋转效应 (Rotation): 角振动 × 线振动 → 虚假常值速度",
             fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = os.path.join(os.path.dirname(__file__), "outputs", "sculling_rotation_error.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(out)}")
plt.close(fig)
