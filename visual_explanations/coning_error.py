r"""圆锥效应 (Coning) 图解。

X-Y 轴同频率角振动 → Z 轴产生净旋转 (即使 Z 角速度=0)。

运行: python visual_explanations/coning_error.py
输出: visual_explanations/outputs/coning_error.png
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

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])

def aa2quat(axis, angle):
    axis = np.asarray(axis); n = np.linalg.norm(axis)
    if n < 1e-15: return np.array([1.,0.,0.,0.])
    u = axis/n; h = angle*0.5
    return np.array([np.cos(h), u[0]*np.sin(h), u[1]*np.sin(h), u[2]*np.sin(h)])

def q2yaw(q):
    w,x,y,z = q
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))

T, dt_fine, nav_T = 2.0, 0.0001, 0.01
omega, amp = 5.0, 0.02
t = np.arange(0, T, dt_fine); n = len(t)

wx = amp*omega*np.cos(omega*t)
wy = amp*omega*np.cos(omega*t + np.pi/2)
wz = np.zeros_like(t)

# truth (fine integration)
q_true = np.array([1.,0.,0.,0.])
q_hist = np.zeros((n,4))
for i in range(n):
    dth = np.array([wx[i], wy[i], wz[i]])*dt_fine
    ang = np.linalg.norm(dth)
    dq = aa2quat(dth/ang, ang) if ang>1e-15 else np.array([1.,0.,0.,0.])
    q_true = quat_mul(q_true, dq); q_true/=np.linalg.norm(q_true)
    q_hist[i] = q_true

# navigation at 100 Hz
steps = int(T/nav_T); fpn = int(nav_T/dt_fine); hf = fpn//2
q1 = np.array([1.,0.,0.,0.]); q2 = np.array([1.,0.,0.,0.])
y1 = np.zeros(steps); y2 = np.zeros(steps); yt = np.zeros(steps)

for k in range(steps):
    i0 = k*fpn
    dth_t = np.array([np.sum(wx[i0:i0+fpn])*dt_fine,
                      np.sum(wy[i0:i0+fpn])*dt_fine,
                      np.sum(wz[i0:i0+fpn])*dt_fine])

    # single-sample
    a = np.linalg.norm(dth_t)
    dq = aa2quat(dth_t/a, a) if a>1e-15 else np.array([1.,0.,0.,0.])
    q1 = quat_mul(q1, dq); q1/=np.linalg.norm(q1); y1[k] = q2yaw(q1)

    # two-sample with coning comp
    dth1 = np.array([np.sum(wx[i0:i0+hf])*dt_fine, np.sum(wy[i0:i0+hf])*dt_fine, 0.])
    dth2 = np.array([np.sum(wx[i0+hf:i0+fpn])*dt_fine, np.sum(wy[i0+hf:i0+fpn])*dt_fine, 0.])
    dth_2 = dth1+dth2 + (2./3.)*np.cross(dth1, dth2)
    a = np.linalg.norm(dth_2)
    dq = aa2quat(dth_2/a, a) if a>1e-15 else np.array([1.,0.,0.,0.])
    q2 = quat_mul(q2, dq); q2/=np.linalg.norm(q2); y2[k] = q2yaw(q2)

    yt[k] = q2yaw(q_hist[i0])

tn = np.arange(steps)*nav_T
err1 = np.rad2deg(y1 - yt)*3600    # deg/h
err2 = np.rad2deg(y2 - yt)*3600

# cross-product diagnostics
cross_z = np.zeros(steps)
for k in range(steps):
    i0 = k*fpn
    d1 = np.array([np.sum(wx[i0:i0+hf])*dt_fine, np.sum(wy[i0:i0+hf])*dt_fine, 0.])
    d2 = np.array([np.sum(wx[i0+hf:i0+fpn])*dt_fine, np.sum(wy[i0+hf:i0+fpn])*dt_fine, 0.])
    cross_z[k] = np.cross(d1, d2)[2]*1e6  # urad

# ── 2x3 layout, no 3D ────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
((ax0, ax1, ax2), (ax3, ax4, ax5)) = axes

# (0,0) Phase diagram
ax = ax0
ax.plot(np.sin(omega*t[:3000]), np.cos(omega*t[:3000]+np.pi/2), "b-", lw=0.8)
ax.set_xlabel("X 轴角度 (归一化)"); ax.set_ylabel("Y 轴角度 (归一化)")
ax.set_title("X-Y 角振动轨迹 (圆形 → 圆锥运动)", fontsize=10)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

# (0,1) ω_x vs ω_y scatter
ax = ax1
skip=30
ax.scatter(wx[::skip]*57.3, wy[::skip]*57.3, c=t[::skip], s=2, alpha=0.6, cmap="coolwarm")
ax.set_xlabel("ω_x [°/s]"); ax.set_ylabel("ω_y [°/s]")
ax.set_title("角速度相图 (同频正交 → 叉积≠0)", fontsize=10)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

# (0,2) Cross-product per epoch
ax = ax2
ax.bar(range(min(30, steps)), cross_z[:30], color="purple", alpha=0.7)
ax.set_xlabel("导航历元"); ax.set_ylabel("dth1 x dth2 的 Z 分量 [µrad]")
ax.set_title("每历元圆锥叉积项 (dth1 x dth2)_z", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# (1,0) Yaw drift — span 2 cols
ax = ax3
ax.plot(tn, err1, "gray", lw=1.2, label="单子样 (不补偿)")
ax.plot(tn, err2, "r", lw=1.2, label="双子样 (圆锥补偿)")
ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.set_xlabel("时间 [s]"); ax.set_ylabel("航向误差 [°/h]")
ax.set_title("Z 轴航向漂移 — 无 Z 输入, 仅 X-Y 角振动", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (1,1) — zoom first 0.5s
ax = ax4
mask = tn <= 0.5
ax.plot(tn[mask], err1[mask], "gray", lw=1.2, label="单子样")
ax.plot(tn[mask], err2[mask], "r", lw=1.2, label="双子样")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("航向误差 [°/h]")
ax.set_title("前 0.5s 放大", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (1,2) — explanation text
ax = ax5
ax.axis("off")
msg = (
    "圆锥效应 — 为什么发生?\n\n"
    "两个正交轴上的角振动 (X 和 Y)\n"
    "在第三个轴 (Z) 产生净旋转。\n\n"
    f"振幅: {amp*57.3:.1f}° @ {omega} Hz\n"
    f"最大漂移率 (单子样): {np.max(np.abs(err1)):.1f} °/h\n"
    f"补偿后残余: {np.max(np.abs(err2)):.4f} °/h\n\n"
    "补偿公式:\n"
    "Δθ = Δθ1+Δθ2 + (2/3)(Δθ1×Δθ2)\n\n"
    "系数 2/3 源于线性角速度假设\n"
    "(Savage, Strapdown Analytics)"
)
ax.text(0.05, 0.95, msg, transform=ax.transAxes, fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("圆锥效应 (Coning): X-Y 角振动 → Z 轴净旋转",
             fontsize=13, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = os.path.join(os.path.dirname(__file__), "outputs", "coning_error.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(out)}")
plt.close(fig)
