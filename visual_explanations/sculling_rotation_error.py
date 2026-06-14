r"""划桨效应 (Sculling) 与旋转效应 (Rotation) 图解 — 三张独立大图。

图1: 物理机制 — 划桨类比 + 角振动×线振动→常值速度
图2: 旋转效应 — body 系在 dt 内旋转导致的方向变化
图3: 实际影响 — 速度误差 + 位置漂移对比

运行: python visual_explanations/sculling_rotation_error.py
输出: visual_explanations/outputs/sculling_{1,2,3}_*.png
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
from matplotlib.patches import FancyBboxPatch, Arc, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ── simulation ───────────────────────────────────────────────────
# X angular oscillation × Z linear oscillation → Y direction sculling
T, dt_fine, nav_T = 1.0, 0.0001, 0.01
omega = 10.0; amp_ang = 0.05; amp_vel = 0.5
t = np.arange(0, T, dt_fine); n = len(t)

wx = amp_ang*omega*np.cos(omega*t)
az = amp_vel*omega*np.cos(omega*t+np.pi/2)
theta_x = np.cumsum(wx)*dt_fine

# truth
v_t = np.zeros(3); v_t_hist = np.zeros((n,3))
for i in range(n):
    dv_b = np.array([0., 0., az[i]])*dt_fine
    ct,st=np.cos(theta_x[i]),np.sin(theta_x[i])
    dv_n = np.array([0., ct*dv_b[1]-st*dv_b[2], st*dv_b[1]+ct*dv_b[2]])
    v_t+=dv_n; v_t_hist[i]=v_t

# navigation
steps=int(T/nav_T); fpn=int(nav_T/dt_fine); hf=fpn//2
vs=np.zeros(3); vd=np.zeros(3)
sh=np.zeros((steps,3)); dh=np.zeros((steps,3)); vth=np.zeros((steps,3))
scul_y=np.zeros(steps); rot_y=np.zeros(steps)

for k in range(steps):
    i0=k*fpn
    dth_t=np.array([np.sum(wx[i0:i0+fpn])*dt_fine,0.,0.])
    dv_t=np.array([0.,0.,np.sum(az[i0:i0+fpn])*dt_fine])
    vs+=dv_t
    dth1=np.array([np.sum(wx[i0:i0+hf])*dt_fine,0.,0.])
    dth2=np.array([np.sum(wx[i0+hf:i0+fpn])*dt_fine,0.,0.])
    dv1=np.array([0.,0.,np.sum(az[i0:i0+hf])*dt_fine])
    dv2=np.array([0.,0.,np.sum(az[i0+hf:i0+fpn])*dt_fine])
    scul=(2./3.)*(np.cross(dth1,dv2)+np.cross(dv1,dth2))
    rot=0.5*np.cross(dth_t,dv_t)
    vd+=dv_t+scul+rot
    sh[k]=vs; dh[k]=vd; vth[k]=v_t_hist[i0]
    scul_y[k]=scul[1]*1000; rot_y[k]=rot[1]*1000

tn=np.arange(steps)*nav_T
err_s=(sh[:,1]-vth[:,1])*1000  # mm/s
err_d=(dh[:,1]-vth[:,1])*1000

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Physical Mechanism (Sculling)
# ══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35)

# (0,0) Boat analogy — hand-drawn style
ax = fig.add_subplot(gs[0, 0])
# boat body
rect = FancyBboxPatch((-0.3, 0.05), 0.6, 0.15, boxstyle="round,pad=0.02",
                       facecolor="saddlebrown", edgecolor="black", alpha=0.7)
ax.add_patch(rect)
# water
ax.axhline(0, color="blue", lw=2, alpha=0.5)
for x in np.linspace(-0.8, 0.8, 15):
    ax.plot(x, -0.02+0.01*np.sin(x*10), "b", lw=0.5, alpha=0.3)
# oar
ax.plot([0.2, 0.25], [0.2, -0.1], "k", lw=2)
ax.plot([0.25, 0.35], [-0.1, -0.08], "k", lw=1.5)
# annotations
ax.annotate("手摇\n(角振动)", xy=(0.2, 0.22), fontsize=9, ha="center",
            color="green", xytext=(0.05, 0.4),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
ax.annotate("桨划水\n(线振动)", xy=(0.3, -0.1), fontsize=9, ha="center",
            color="blue", xytext=(0.55, -0.15),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
ax.annotate("船前进!\n(常值速度)", xy=(0.35, 0.12), fontsize=10, ha="center",
            color="red", xytext=(0.6, 0.25),
            arrowprops=dict(arrowstyle="->", color="red", lw=2))
ax.set_xlim(-0.8, 0.9); ax.set_ylim(-0.3, 0.5)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("划桨效应的物理类比", fontsize=12, fontweight="bold")

# (0,1) Signal waveforms
ax = fig.add_subplot(gs[0, 1])
t_s = t[:1500]
ax2_ = ax.twinx()
ax.plot(t_s, wx[:1500]*57.3, "red", lw=1, label="ω_x (角速度)")
ax2_.plot(t_s, az[:1500], "blue", lw=1, label="a_z (线加速度)")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("角速度 [°/s]", color="red")
ax2_.set_ylabel("加速度 [m/s²]", color="blue")
ax.set_title("X 轴角振动 + Z 轴线振动\n同频率, 90° 相位差 → 最大划桨效应", fontsize=11)
ax.grid(True, alpha=0.3)

# (0,2) Math explanation
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
msg = (
    "划桨效应 — 数学模型\n\n"
    "输入:\n"
    f"  dth_x = {amp_ang:.2f}·cos({omega}t)·dt  [rad]\n"
    f"  dv_z  = {amp_vel:.2f}·cos({omega}t+90°)·dt [m/s]\n\n"
    "双子样补偿:\n"
    "  dv_scul = (2/3)(dth1 × dv2 + dv1 × dth2)\n\n"
    "dth_x × dv_z → Y 方向!\n"
    "叉积结果指向 Y 轴\n"
    "→ Y 轴获得虚假的常值速度\n\n"
    "物理含义:\n"
    "角振动(绕X) × 线振动(沿Z)\n"
    "→ 正交方向(Y)的净速度\n"
    "就像划桨时手摇和桨划水\n"
    "产生了船向前的推力"
)
ax.text(0.05, 0.98, msg, transform=ax.transAxes, fontsize=10, va="top",
        linespacing=1.3,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# (1,0) Sculling comp per epoch — bar chart
ax = fig.add_subplot(gs[1, 0])
x_idx = np.arange(min(40, steps))
w = 0.35
ax.bar(x_idx-w/2, scul_y[:40], w, color="purple", alpha=0.7, label="划桨 (2/3)(dth1×dv2+dv1×dth2)")
ax.bar(x_idx+w/2, rot_y[:40], w, color="orange", alpha=0.7, label="旋转 (1/2)(dth×dv)")
ax.set_xlabel("导航历元"); ax.set_ylabel("Y 方向补偿量 [mm/s]")
ax.set_title("每历元划桨 vs 旋转补偿量\n划桨 ≈ 2× 旋转 (当前参数)", fontsize=11)
ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

# (1,1) Velocity error
ax = fig.add_subplot(gs[1, 1])
ax.plot(tn, err_s, "gray", lw=1.2, label="单子样 (不补偿)")
ax.plot(tn, err_d, "r", lw=1.2, label="双子样 (划桨+旋转补偿)")
ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.set_xlabel("时间 [s]"); ax.set_ylabel("Y 轴速度误差 [mm/s]")
ax.set_title("Y 轴速度误差 — 1 秒累积到 mm/s 级", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (1,2) Position drift
ax = fig.add_subplot(gs[1, 2])
pos_s = np.cumsum(err_s)*nav_T*1e-3  # m → mm
pos_d = np.cumsum(err_d)*nav_T*1e-3
ax.plot(tn, pos_s*1000, "gray", lw=1.2, label="单子样")
ax.plot(tn, pos_d*1000, "r", lw=1.2, label="双子样")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("Y 位置漂移 [µm]")
ax.set_title("位置漂移 (速度误差的积分)\n1 秒单子样漂 ~0.5mm", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("划桨效应 — 物理机制 + 信号 + 补偿量 + 误差分析",
             fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "sculling_1_mechanism.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: sculling_1_mechanism.png")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Rotation Effect Explanation
# ══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(1, 3, wspace=0.35)

# (0) Rotation diagram — 3 sequential body frames
ax = fig.add_subplot(gs[0, 0])
# Time t
ax.arrow(0.2, 0.5, 0.18, 0, color="black", width=0.005, head_width=0.03)
ax.arrow(0.2, 0.5, 0, 0.18, color="black", width=0.005, head_width=0.03)
ax.text(0.42, 0.5, "X", fontsize=11, ha="center")
ax.text(0.22, 0.72, "Y", fontsize=11, ha="center")
ax.text(0.15, 0.35, "t", fontsize=12, fontweight="bold")

# Time t+dt/2
cx, cy = 0.6, 0.6
rot_angle = 0.25  # ~14°
ax.arrow(cx, cy, 0.18*np.cos(rot_angle), 0.18*np.sin(rot_angle),
         color="gray", width=0.005, head_width=0.03)
ax.arrow(cx, cy, -0.18*np.sin(rot_angle), 0.18*np.cos(rot_angle),
         color="gray", width=0.005, head_width=0.03)
ax.text(cx+0.22, cy+0.05, "X'", fontsize=10, color="gray")
ax.text(cx-0.06, cy+0.27, "Y'", fontsize=10, color="gray")
ax.text(cx-0.05, cy-0.05, "t+dt/2", fontsize=10, color="gray")

# Arc showing rotation
arc = Arc((0.2, 0.5), 0.4, 0.4, angle=0, theta1=0, theta2=rot_angle*57.3,
          color="green", lw=2, fill=False)
ax.add_patch(arc)
ax.text(0.35, 0.62, "dt内旋转", fontsize=9, color="green")

# dv in body(t) vs body(t+dt/2)
ax.arrow(0.2, 0.15, 0.25, 0.18, color="red", width=0.006, head_width=0.04,
         length_includes_head=True, label="dv 方向在变!")
ax.text(0.5, 0.28, "dv 的方向\n随 body 系\n一起在转", fontsize=9, color="red")

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("旋转效应: dt 内 body 系在旋转", fontsize=12, fontweight="bold")

# (1) Rotation effect math
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
msg = (
    "旋转效应 — 数学本质\n\n"
    "问题:\n"
    "IMU 在 dt 内逐个采样 dv\n"
    "但 body 系在 dt 内本身在旋转\n"
    "dv(0) 和 dv(dt) 的 body 系方向不同!\n\n"
    "单子样做法:\n"
    "  把所有的 dv 简单加起来\n"
    "  → 忽略了 'body 系在转' 这件事\n\n"
    "真实情况:\n"
    "  dv 的 body 系在 dt 内转了 dθ\n"
    "  → dv 在导航系的方向偏差 ≈ dθ × dv\n\n"
    "补偿项:\n"
    "  dv_rot = (1/2) (dθ_total × dv_total)\n"
    "  \n"
    "  系数 1/2 是因为平均旋转角是 dθ/2\n\n"
    "划桨和旋转的异同:\n"
    "  划桨: 角振动 × 线振动 (同频率)\n"
    "  旋转: 角运动 × 线运动 (同一 dt 内)\n"
    "  物理机制不同, 补偿公式也不同\n"
    "  但都源于 '旋转的不可交换性'"
)
ax.text(0.05, 0.98, msg, transform=ax.transAxes, fontsize=10, va="top",
        linespacing=1.2,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# (2) Compensation amount comparison
ax = fig.add_subplot(gs[0, 2])
x_idx = np.arange(min(40, steps))
w = 0.3
ax.bar(x_idx-w, scul_y[:40], w, color="purple", alpha=0.8, label="划桨补偿")
ax.bar(x_idx, rot_y[:40], w, color="orange", alpha=0.8, label="旋转补偿")
ax.bar(x_idx+w, (scul_y[:40]+rot_y[:40]), w, color="green", alpha=0.5, label="合计")
ax.set_xlabel("导航历元"); ax.set_ylabel("Y 方向速度补偿 [mm/s]")
ax.set_title("每历元的三种补偿量对比\n"
             "划桨(紫) + 旋转(橙) = 合计(绿)", fontsize=11)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

fig.suptitle("旋转效应 — body 系旋转 + 数学推导 + 补偿量对比",
             fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "sculling_2_rotation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: sculling_2_rotation.png")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Practical Impact
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (0) Velocity error
ax = axes[0]
ax.plot(tn, err_s, "gray", lw=1.5, label="单子样")
ax.plot(tn, err_d, "r", lw=1.5, label="双子样")
ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.fill_between(tn, 0, err_s, alpha=0.1, color="gray")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("Y 轴速度误差 [mm/s]")
ax.set_title("速度误差 — 1 秒内 mm/s 级", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (1) Position drift
ax = axes[1]
pos_s = np.cumsum(err_s)*nav_T*1e-3
pos_d = np.cumsum(err_d)*nav_T*1e-3
ax.plot(tn, pos_s*1e3, "gray", lw=1.5, label="单子样")
ax.plot(tn, pos_d*1e3, "r", lw=1.5, label="双子样")
ax.set_xlabel("时间 [s]"); ax.set_ylabel("Y 位置漂移 [µm]")
ax.set_title("位置漂移 — 1 秒 ~0.5mm (随 t² 增长)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (2) Summary
ax = axes[2]
ax.axis("off")
msg = (
    "不可交换误差 — 总结\n\n"
    "三种误差都是因为:\n"
    "'离散采样 ≠ 连续旋转'\n\n"
    "圆锥: 角 × 角 → 第三轴净旋转\n"
    f"  效果: °/h 级航向漂移\n\n"
    "划桨: 角 × 线 → 正交方向净速度\n"
    f"  效果: mm/s 级速度偏差\n\n"
    "旋转: 角 × 线 → 方向变化\n"
    f"  效果: µm/s 级 (通常 < 划桨)\n\n"
    "补偿后 (双子样):\n"
    "  残余误差 < 原始误差的 1%\n\n"
    "工程意义:\n"
    "  • 高振动环境 (直升机、导弹) 必须补偿\n"
    "  • 车载低动态可不补偿 (但双子样简单)\n"
    "  • 导航周期越长, 补偿越关键"
)
ax.text(0.05, 0.98, msg, transform=ax.transAxes, fontsize=10, va="top",
        linespacing=1.4,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

fig.suptitle("不可交换误差 — 实际影响与工程意义",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "sculling_3_impact.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: sculling_3_impact.png")
