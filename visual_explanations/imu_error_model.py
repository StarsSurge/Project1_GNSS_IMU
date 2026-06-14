r"""IMU 误差模型图解 — 三张独立大图。

图1: 系统性误差 (零偏、比例因子、交轴耦合)
图2: 随机误差 (白噪声、偏置不稳定性、随机游走) 时域对比
图3: Allan 方差曲线 — 所有噪声在一张图上

运行: python visual_explanations/imu_error_model.py
输出: visual_explanations/outputs/imu_error_{1,2,3}_*.png
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

rng = np.random.RandomState(42)
fs = 200; dt = 1/fs; T = 30
t = np.arange(0, T, dt); n = len(t)
true_signal = 2.0 * np.sin(2*np.pi*0.3*t) + 3.0

# ── signals ──
white = rng.randn(n)*0.5
rw_sigma = 0.03; rw = np.cumsum(rng.randn(n))*rw_sigma*np.sqrt(dt)
bias_drift = 0.8 + rw
gm_tau = 10.0; gm_sigma = 0.3
alpha = np.exp(-dt/gm_tau); gm_drive = gm_sigma*np.sqrt(1-alpha**2)
gm = np.zeros(n); gm[0] = rng.randn()*gm_sigma
for i in range(1,n): gm[i] = alpha*gm[i-1] + gm_drive*rng.randn()
sf = 1.03; sf_meas = true_signal*sf
composite = true_signal + bias_drift + white

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Systematic Errors
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
((ax0, ax1), (ax2, ax3)) = axes

# (0,0) Bias
ax = ax0
ax.plot(t, true_signal, "k--", lw=1, alpha=0.6)
ax.plot(t, true_signal+0.8, "gray", lw=1, label="恒定偏置 +0.8")
ax.plot(t, true_signal+bias_drift, "r", lw=1, label="偏置 + 随机游走漂移")
ax.set_ylabel("信号值"); ax.set_xlabel("时间 [s]")
ax.set_title("① 零偏 (Bias)\n常值偏移 + 缓慢随机游走漂移", fontsize=11)
ax.legend(fontsize=8)

# (0,1) Scale Factor
ax = ax1
ax.plot(t[:500], true_signal[:500], "k--", lw=1, alpha=0.6)
ax.plot(t[:500], sf_meas[:500], "b", lw=1)
ax.fill_between(t[:500], true_signal[:500], sf_meas[:500], alpha=0.12, color="blue")
ax.set_ylabel("信号值"); ax.set_xlabel("时间 [s]")
ax.set_title("② 比例因子 (Scale Factor)\n输出 = 真实 × (1+3%), 误差与信号幅值成正比", fontsize=11)
ax.annotate("误差 = 3% × 真实值", xy=(2, sf_meas[400]),
            xytext=(4, 0.5), fontsize=9, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue", alpha=0.5))

# (2,0) Composite
ax = ax2
ax.plot(t, true_signal, "k--", lw=1, alpha=0.6, label="真实值")
ax.plot(t, composite, "r", lw=0.6, alpha=0.8, label="含全部误差的测量值")
ax.set_ylabel("信号值"); ax.set_xlabel("时间 [s]")
ax.set_title("③ 综合叠加\n偏置 + 白噪声 + 比例因子 + 偏置漂移 — 真实信号被淹没", fontsize=11)
ax.legend(fontsize=8)

# (2,1) Misalignment diagram
ax = ax3
ax.axis("equal"); ax.set_xlim(-0.3, 1.5); ax.set_ylim(-0.3, 1.5)
# ideal axes
ax.arrow(0, 0, 1.2, 0, color="black", width=0.005, head_width=0.04, label="X (理想)")
ax.arrow(0, 0, 0, 1.2, color="black", width=0.005, head_width=0.04)
# misaligned axes
eps = 0.08
ax.arrow(0, 0, 1.2, eps*1.2, color="red", width=0.005, head_width=0.04)
ax.arrow(0, 0, -eps*1.2, 1.2, color="red", width=0.005, head_width=0.04)
# arc
theta = np.linspace(0, np.arctan(eps), 20)
ax.plot(0.6*np.cos(theta), 0.6*np.sin(theta), "red", lw=0.8)
ax.text(0.65, 0.04, f"~{eps*57:.0f}°", fontsize=8, color="red")
ax.text(1.25, 0, "X", fontsize=11)
ax.text(0, 1.25, "Y", fontsize=11)
ax.text(1.0, 0.20, "X' (实际)", fontsize=9, color="red")
ax.text(-0.20, 1.0, "Y' (实际)", fontsize=9, color="red")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("④ 交轴耦合 (Misalignment)\n三轴不完全正交 → 一个轴的运动泄漏到另一轴", fontsize=11)

fig.suptitle("IMU 系统性误差 (可标定补偿)", fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "imu_error_1_systematic.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: imu_error_1_systematic.png")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Stochastic Errors
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
((ax0, ax1), (ax2, ax3)) = axes

# (0,0) White noise
ax = ax0
ax.plot(t[:500], white[:500], "gray", lw=0.5)
ax.set_ylabel("噪声幅度"); ax.set_xlabel("时间 [s]")
ax.set_title("① 白噪声\n逐历元独立, 方差恒定", fontsize=11)
# inset: power spectrum
axin = ax.inset_axes([0.55, 0.55, 0.40, 0.40])
freq = np.fft.rfftfreq(2000, dt)
psd = np.abs(np.fft.rfft(white[:2000]))**2
axin.semilogy(freq[1:], psd[1:], "gray", lw=0.5)
axin.set_title("功率谱 (平坦)", fontsize=7)
axin.tick_params(labelsize=6)

# (0,1) Integrated white noise → Random Walk
ax = ax1
integrated = np.cumsum(white)*dt
ax.plot(t[:1000], integrated[:1000], "b", lw=0.6)
ax.set_ylabel("积分值 (随机游走)"); ax.set_xlabel("时间 [s]")
ax.set_title("② 白噪声积分 → 角度/速度随机游走 (ARW/VRW)\n"
             "白噪声积分后不再是白噪声 — 方差随时间线性增长", fontsize=11)
ax.axhline(0, color="gray", ls=":", lw=0.5)

# (2,0) Bias Instability (1st-order GM)
ax = ax2
ax.plot(t, gm, "orange", lw=0.8)
ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.axhline(+gm_sigma, color="red", ls=":", lw=0.5)
ax.axhline(-gm_sigma, color="red", ls=":", lw=0.5, label=f"±σ = ±{gm_sigma:.1f}")
ax.set_ylabel("偏置值"); ax.set_xlabel("时间 [s]")
ax.set_title(f"③ 偏置不稳定性 (Bias Instability)\n"
             f"一阶高斯-马尔可夫过程, 时间常数 τ={gm_tau}s — "
             "介于白噪声和随机游走之间", fontsize=11)
ax.legend(fontsize=8)

# (2,1) Comparison of all three
ax = ax3
ax.plot(t[:500], white[:500]*0.3, "gray", lw=0.5, alpha=0.6, label="白噪声 (×0.3)")
ax.plot(t[:500], integrated[:500]*0.3, "b", lw=0.8, label="积分白噪声 (随机游走)")
ax.plot(t[:500], gm[:500]*0.5, "orange", lw=0.8, label="偏置不稳定性 (GM)")
ax.set_ylabel("归一化幅度"); ax.set_xlabel("时间 [s]")
ax.set_title("④ 三种随机误差时域对比\n白噪声(抖动) vs 随机游走(漂走) vs GM(缓慢波动)", fontsize=11)
ax.legend(fontsize=7)

fig.suptitle("IMU 随机误差 (统计描述, 不可消除)", fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "imu_error_2_stochastic.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: imu_error_2_stochastic.png")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Allan Variance
# ══════════════════════════════════════════════════════════════════

def simple_allan(signal, fs, tau_max=30.0, n_taus=60):
    taus = np.logspace(-1, np.log10(tau_max), n_taus)
    devs = np.zeros_like(taus)
    for j, tau_s in enumerate(taus):
        m = int(tau_s*fs)
        if m < 2 or 2*m > len(signal): devs[j] = np.nan; continue
        nc = len(signal)//m
        clusters = signal[:nc*m].reshape(nc, m)
        means = np.mean(clusters, axis=1)
        devs[j] = np.sqrt(0.5*np.mean(np.diff(means)**2))
    return taus, devs

taus, ad = simple_allan(composite, fs, tau_max=30, n_taus=80)
valid = ~np.isnan(ad)

fig, ax = plt.subplots(figsize=(14, 8))

ax.loglog(taus[valid], ad[valid], "k-", lw=2, label="Allan 标准差 σ(τ)", zorder=3)

# slope annotations
tau_r = np.array([0.12, 0.4])
ax.loglog(tau_r, tau_r**(-0.5)*0.4, "r--", lw=1.5, label="斜率 −½: 角度/速度随机游走 (ARW/VRW)")
tau_r2 = np.array([3, 20])
ax.loglog(tau_r2, tau_r2**0*0.22, "orange", ls="--", lw=1.5, label="斜率 0: 偏置不稳定性 (最小点)")
tau_r3 = np.array([8, 25])
ax.loglog(tau_r3, tau_r3**0.5*0.08, "purple", ls="--", lw=1.5, label="斜率 +½: 速率随机游走")

# find bias instability minimum
bi_idx = np.nanargmin(ad)
ax.annotate(f"偏置不稳定性\nσ={ad[bi_idx]:.3f} @ τ={taus[bi_idx]:.1f}s",
            xy=(taus[bi_idx], ad[bi_idx]),
            xytext=(taus[bi_idx]*3, ad[bi_idx]*2.5),
            fontsize=10, color="orange",
            arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))

ax.set_xlabel("积分时间 τ [s]", fontsize=12)
ax.set_ylabel("Allan 标准差", fontsize=12)
ax.set_title("Allan 方差 — 一张图诊断所有 IMU 随机噪声\n"
             "不同 τ 段的斜率对应不同噪声类型, 谷底 = 偏置不稳定性",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower left")
ax.grid(True, alpha=0.3, which="both")
ax.set_xlim(taus[valid][0]*0.8, taus[valid][-1]*1.2)

txt = ("如何读 Allan 方差图:\n\n"
       "• τ < 0.5s: 白噪声主导\n"
       "  斜率 −½ → ARW/VRW 系数\n\n"
       "• τ ≈ 1-10s: 谷底\n"
       "  高度 = 偏置不稳定性\n\n"
       "• τ > 20s: 随机游走主导\n"
       "  斜率 +½ → 速率随机游走系数")
ax.text(0.98, 0.98, txt, transform=ax.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(OUT, "imu_error_3_allan.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: imu_error_3_allan.png")
