r"""圆锥效应 (Coning) 图解 — 三张独立大图。

图1: 物理机制 — 为什么两个轴的角振动会在第三轴产生净旋转
图2: 三维示意 — 圆锥在桌面滚动的空间图像
图3: 实际影响 — 单子样 vs 双子样航向漂移对比

运行: python visual_explanations/coning_error.py
输出: visual_explanations/outputs/coning_{1,2,3}_*.png
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

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ── quaternion utils ──
def qmul(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])
def aa2q(axis, angle):
    a=np.asarray(axis); n=np.linalg.norm(a)
    if n<1e-15: return np.array([1.,0.,0.,0.])
    u=a/n; h=angle*.5
    return np.array([np.cos(h),u[0]*np.sin(h),u[1]*np.sin(h),u[2]*np.sin(h)])
def q2y(q): w,x,y,z=q; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))

# ── simulation params ──
T, dt_fine, nav_T = 2.0, 0.0001, 0.01
omega, amp = 5.0, 0.02
t = np.arange(0, T, dt_fine); n = len(t)
wx = amp*omega*np.cos(omega*t)
wy = amp*omega*np.cos(omega*t+np.pi/2); wz = np.zeros_like(t)

# truth
q_t = np.array([1.,0.,0.,0.]); q_hist = np.zeros((n,4))
for i in range(n):
    dth = np.array([wx[i],wy[i],wz[i]])*dt_fine
    a = np.linalg.norm(dth)
    dq = aa2q(dth/a,a) if a>1e-15 else np.array([1.,0.,0.,0.])
    q_t = qmul(q_t,dq); q_t/=np.linalg.norm(q_t); q_hist[i]=q_t

# navigation
steps = int(T/nav_T); fpn = int(nav_T/dt_fine); hf = fpn//2
q1=np.array([1.,0.,0.,0.]); q2=np.array([1.,0.,0.,0.])
y1=np.zeros(steps); y2=np.zeros(steps); yt=np.zeros(steps)
cross_z = np.zeros(steps)
for k in range(steps):
    i0=k*fpn
    dth_t=np.array([np.sum(wx[i0:i0+fpn])*dt_fine,
                    np.sum(wy[i0:i0+fpn])*dt_fine,0.])
    a=np.linalg.norm(dth_t)
    dq=aa2q(dth_t/a,a) if a>1e-15 else np.array([1.,0.,0.,0.])
    q1=qmul(q1,dq); q1/=np.linalg.norm(q1); y1[k]=q2y(q1)

    dth1=np.array([np.sum(wx[i0:i0+hf])*dt_fine,np.sum(wy[i0:i0+hf])*dt_fine,0.])
    dth2=np.array([np.sum(wx[i0+hf:i0+fpn])*dt_fine,np.sum(wy[i0+hf:i0+fpn])*dt_fine,0.])
    cross_z[k]=np.cross(dth1,dth2)[2]*1e6  # urad
    dth_2=dth1+dth2+(2./3.)*np.cross(dth1,dth2)
    a=np.linalg.norm(dth_2)
    dq=aa2q(dth_2/a,a) if a>1e-15 else np.array([1.,0.,0.,0.])
    q2=qmul(q2,dq); q2/=np.linalg.norm(q2); y2[k]=q2y(q2)
    yt[k]=q2y(q_hist[i0])

tn=np.arange(steps)*nav_T
err1=np.rad2deg(y1-yt)*3600
err2=np.rad2deg(y2-yt)*3600

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Physical Mechanism
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
((ax0, ax1), (ax2, ax3)) = axes

# (0,0) ω signals
ax = ax0
ax.plot(t[:2000], wx[:2000]*57.3, "red", lw=0.8, label="ω_x")
ax.plot(t[:2000], wy[:2000]*57.3, "blue", lw=0.8, label="ω_y")
ax.plot(t[:2000], wz[:2000], "k--", lw=0.5, label="ω_z (=0!)")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("角速度 [°/s]")
ax.set_title("① 角速度输入: X 和 Y 有同频振动, Z 恒为零", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (0,1) Phase diagram X vs Y
ax = ax1
ax.plot(np.sin(omega*t[:3000]), np.cos(omega*t[:3000]+np.pi/2), "b-", lw=0.6)
ax.set_xlabel("X 轴角度 (归一化)"); ax.set_ylabel("Y 轴角度 (归一化)")
ax.set_title("② X-Y 相位图: 圆形轨迹 → 圆锥运动", fontsize=11)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.annotate("轨迹是圆形\n意味着圆锥运动", xy=(0.5, 0.5), fontsize=10, ha="center",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))

# (2,0) Cross product explanation
ax = ax2
dth_sample = np.array([0.001, 0.001, 0.0])
# draw dth1 and dth2
ax.arrow(0, 0, 5, 0, color="red", width=0.05, head_width=0.3, label="dth1 (子样1)")
ax.arrow(0, 0, 0, 5, color="blue", width=0.05, head_width=0.3, label="dth2 (子样2)")
ax.arrow(0, 0, 0, 0, color="purple", width=0.05, head_width=0.3)  # placeholder
# dth1+dth2
ax.arrow(0, 0, 5, 5, color="gray", width=0.03, head_width=0.2, alpha=0.5)
ax.text(5.3, 5.3, "dth1+dth2\n(简单相加)", fontsize=8, color="gray")
# cross product (out of page → Z)
ax.annotate("dth1×dth2\n指向 Z 轴\n(垂直纸面向外)",
            xy=(2.5, 2.5), fontsize=10, ha="center", color="purple",
            xytext=(8, 2), arrowprops=dict(arrowstyle="->", color="purple", lw=1.5))
ax.set_xlim(-1, 10); ax.set_ylim(-1, 7)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.set_title("③ 叉积的物理含义\n"
             "dth1 × dth2 = 两个旋转的'不可交换'部分 → 指向 Z 轴", fontsize=11)
ax.legend(fontsize=8, loc="lower right")

# (2,1) Cross product values
ax = ax3
colors = ["green" if v>0 else "red" for v in cross_z[:40]]
ax.bar(range(40), cross_z[:40], color=colors, alpha=0.7)
ax.set_xlabel("导航历元"); ax.set_ylabel("(dth1×dth2) 的 Z 分量 [µrad]")
ax.set_title("④ 每历元的圆锥叉积项\n正=Z轴正转, 负=Z轴反转 — 交替但累积", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="gray", lw=0.5)

fig.suptitle("圆锥效应 — 物理机制: 角振动 × 角振动 → 正交轴净旋转",
             fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "coning_1_mechanism.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: coning_1_mechanism.png")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Spatial Example
# ══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 8))

# Left: 3D cone trajectory
ax = fig.add_subplot(1, 2, 1, projection="3d")
# cone surface
n_cone = 3000; r_cone = 0.2
theta_c = 2*np.pi*omega*t[:n_cone]
z_c = np.sin(theta_c)*r_cone
x_c = np.cos(theta_c)*r_cone
y_c = np.cos(theta_c+np.pi/2)*r_cone*0.6
ax.plot(x_c, y_c, z_c, "b", lw=0.8)
# Z axis
ax.quiver(0, 0, -0.3, 0, 0, 0.5, color="red", lw=2, arrow_length_ratio=0.08)
ax.text(0, 0, 0.25, "Z 轴净旋转", color="red", fontsize=10)
# X and Y axes at base
ax.quiver(0, 0, -0.3, 0.3, 0, 0, color="gray", lw=1, arrow_length_ratio=0.1, alpha=0.5)
ax.quiver(0, 0, -0.3, 0, 0.3, 0, color="gray", lw=1, arrow_length_ratio=0.1, alpha=0.5)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("圆锥运动的 3D 轨迹\n"
             "X-Y 振荡画出椭圆底, Z 轴获得净旋转\n"
             "(类比: 桌面滚动的圆锥)", fontsize=11)
ax.view_init(elev=20, azim=-50)

# Right: explanation with 2D projection views
ax = fig.add_subplot(1, 2, 2)
ax.axis("off")
msg = (
    "圆锥效应 — 空间直观\n\n"
    "想象一个圆锥在桌面上滚动:\n\n"
    "• 圆锥的尖端着地, 在桌面画圈\n"
    "  (X-Y 平面上的圆形 = 角振动)\n\n"
    "• 圆锥绕着自身的 Z 轴在转动\n"
    "  (即使桌面是平的, Z 轴角速度=0!)\n\n"
    "• 桌面画一圈 → 圆锥自转一小圈\n"
    f"  旋转量 ≈ (振幅)²/(2·频率) 每周期\n\n"
    "这就是 '不可交换误差' 的空间直觉:\n"
    "X 转 + Y 转 ≠ Y 转 + X 转\n"
    "其差值 = X×Y 叉积 → Z 轴净旋转\n\n"
    "数值参数:\n"
    f"  角振幅: {amp*57.3:.1f}°\n"
    f"  频率: {omega} Hz\n"
    f"  每历元叉积: {np.mean(np.abs(cross_z)):.1f} µrad\n"
    f"  2 秒累积漂移 (单): {np.max(np.abs(err1)):.2f} °/h"
)
ax.text(0.05, 0.95, msg, transform=ax.transAxes, fontsize=11, va="top",
        linespacing=1.5,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("圆锥效应 — 空间直观: 3D 轨迹 + 物理类比", fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "coning_2_spatial.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: coning_2_spatial.png")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Practical Impact
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (0) Full yaw drift
ax = axes[0]
ax.plot(tn, err1, "gray", lw=1.2, label="单子样 (不补偿)")
ax.plot(tn, err2, "r", lw=1.2, label="双子样 (圆锥补偿)")
ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.set_xlabel("时间 [s]"); ax.set_ylabel("航向误差 [°/h]")
ax.set_title("全时段航向漂移", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (1) Zoom first 0.5s
ax = axes[1]
mask = tn <= 0.5
ax.plot(tn[mask], err1[mask], "gray", lw=1.5, label="单子样")
ax.plot(tn[mask], err2[mask], "r", lw=1.5, label="双子样")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("航向误差 [°/h]")
ax.set_title("前 0.5s 放大 — 误差从零开始累积", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (2) Accumulated cross product vs yaw error
ax = axes[2]
cum_cross = np.cumsum(cross_z)*1e-6*57.3*3600  # cumulative effect in deg/h
ax.plot(tn, cum_cross, "purple", lw=1.2, label="累积叉积 (理论预测)")
ax.plot(tn, err1, "gray", lw=1.2, alpha=0.7, label="实际航向误差 (单子样)")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("航向 [°/h]")
ax.set_title("累积叉积 = 理论漂移量 ≈ 实际误差", fontsize=11)
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig.suptitle("圆锥效应 — 实际影响: 单子样 °/h 级航向漂移 vs 双子样消除",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "coning_3_impact.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: coning_3_impact.png")
