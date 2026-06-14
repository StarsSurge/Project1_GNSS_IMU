r"""IMU 误差模型图解。

用 6 张子图系统展示 IMU 的五类误差来源:
  1. 零偏 (Bias)
  2. 比例因子 (Scale Factor)
  3. 白噪声 → ARW / VRW
  4. 偏置不稳定性 (Bias Instability)
  5. 随机游走 (Random Walk)

外加 Allan 方差曲线标注各噪声区域。

运行: python visual_explanations/imu_error_model.py
输出: visual_explanations/outputs/imu_error_model.png
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import matplotlib
matplotlib.use("Agg")

# ── 中文字体 (须在 import pyplot 之前设置 rcParams) ──
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
# 1. Generate demo signals
# ══════════════════════════════════════════════════════════════════

rng = np.random.RandomState(42)
fs = 100          # Hz
t = np.arange(0, 60, 1/fs)
dt = 1/fs
n = len(t)

# --- pure white noise ---
white_noise = rng.randn(n) * 0.5

# --- bias + random walk ---
sigma_rw = 0.02
random_walk = np.cumsum(rng.randn(n)) * sigma_rw * np.sqrt(dt)
bias = 0.8 + random_walk

# --- bias instability (1st-order GM) ---
tau = 10.0
sigma_bi = 0.3
alpha = np.exp(-dt / tau)
sigma_drive = sigma_bi * np.sqrt(1 - alpha**2)
gm = np.zeros(n)
gm[0] = rng.randn() * sigma_bi
for i in range(1, n):
    gm[i] = alpha * gm[i-1] + sigma_drive * rng.randn()

# --- scale factor error ---
true_input = 2.0 * np.sin(2 * np.pi * 0.3 * t) + 3.0
scale_factor = 1.03           # 3% error
measured_sf = true_input * scale_factor

# --- composite signal ---
composite = true_input + bias + white_noise

# --- Allan deviation (simplified computation) ---
def simple_allan_dev(signal, fs, tau_max=10.0, n_taus=50):
    taus = np.logspace(-1, np.log10(tau_max), n_taus)
    devs = np.zeros_like(taus)
    for j, tau_s in enumerate(taus):
        m = int(tau_s * fs)
        if m < 2 or 2*m > len(signal):
            devs[j] = np.nan
            continue
        n_clusters = len(signal) // m
        clusters = signal[:n_clusters*m].reshape(n_clusters, m)
        means = np.mean(clusters, axis=1)
        devs[j] = np.sqrt(0.5 * np.mean(np.diff(means)**2))
    return taus, devs

taus, ad = simple_allan_dev(composite, fs, tau_max=30)

# ══════════════════════════════════════════════════════════════════
# 2. Plot
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 2, figsize=(16, 13))

# (0,0) ── Bias ──
ax = axes[0, 0]
ax.plot(t, true_input, "k--", linewidth=1, alpha=0.6, label="真实值")
ax.plot(t, true_input + 0.8, "gray", linewidth=1, label="恒定偏置 (+0.8)")
ax.plot(t, true_input + bias, "r", linewidth=1, label="偏置 + 随机游走")
ax.set_ylabel("信号值"); ax.set_xlabel("时间 [s]")
ax.set_title("① 零偏 (Bias)\n常值偏移 + 缓慢漂移 (随机游走)")
ax.legend(fontsize=8, loc="upper right")

# (0,1) ── Scale Factor ──
ax = axes[0, 1]
ax.plot(t, true_input, "k--", linewidth=1, alpha=0.6, label="真实输入")
ax.plot(t, measured_sf, "b", linewidth=1, label="比例因子 1.03 × 真实")
ax.fill_between(t, true_input, measured_sf, alpha=0.15, color="blue")
ax.set_ylabel("信号值"); ax.set_xlabel("时间 [s]")
ax.set_title("② 比例因子 (Scale Factor)\n输出 = 真实 × (1 + s), s = 3%")
ax.legend(fontsize=8)

# (1,0) ── White Noise + ARW ──
ax = axes[1, 0]
ax.plot(t[:300], white_noise[:300], "gray", linewidth=0.5, alpha=0.7)
ax.set_ylabel("噪声幅度"); ax.set_xlabel("时间 [s]")
ax.set_title("③ 白噪声 → 角度/速度随机游走 (ARW/VRW)\n"
             "高频随机抖动, 积分后方差 ∝ dt")
# inset: integrated white noise
axin = ax.inset_axes([0.60, 0.60, 0.35, 0.35])
integrated = np.cumsum(white_noise) * dt
axin.plot(t[:500], integrated[:500], "b", linewidth=0.6)
for spine in axin.spines.values():
    spine.set_edgecolor("blue")
    spine.set_linewidth(0.5)
axin.tick_params(labelsize=6, colors="blue")
axin.set_title("积分后: 随机游走", fontsize=7, color="blue")

# (1,1) ── Bias Instability ──
ax = axes[1, 1]
ax.plot(t, gm, "orange", linewidth=0.8)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax.axhline(y=+sigma_bi, color="red", linewidth=0.5, linestyle=":",
           label=f"±σ_bi = ±{sigma_bi:.1f}")
ax.axhline(y=-sigma_bi, color="red", linewidth=0.5, linestyle=":")
ax.set_ylabel("偏置值"); ax.set_xlabel("时间 [s]")
ax.set_title("④ 偏置不稳定性 (Bias Instability)\n"
             f"一阶高斯-马尔可夫, τ={tau}s, 长时缓慢波动")
ax.legend(fontsize=8)

# (2,0) ── Allan Variance ──
ax = axes[2, 0]
valid = ~np.isnan(ad)
ax.loglog(taus[valid], ad[valid], "k-", linewidth=1.5, label="Allan 标准差")
# annotate zones
yl = ax.get_ylim()
mid_tau = 10**np.mean(np.log10(taus[valid]))
mid_dev = np.interp(mid_tau, taus[valid], ad[valid])
# slope lines
tau_ref = np.array([0.15, 0.5])
ax.loglog(tau_ref, tau_ref**(-0.5)*0.5, "r--", linewidth=1, label="−½: ARW/VRW")
tau_ref2 = np.array([8, 25])
ax.loglog(tau_ref2, tau_ref2**(0)*0.18, "orange", linestyle="--", linewidth=1,
          label="0: 偏置不稳定性")
tau_ref3 = np.array([8, 25])
ax.loglog(tau_ref3, tau_ref3**(0.5)*0.06, "purple", linestyle="--", linewidth=1,
          label="+½: 速率随机游走")
ax.set_xlabel("积分时间 τ [s]"); ax.set_ylabel("Allan 标准差")
ax.set_title("⑤ Allan 方差 — 所有噪声在同一个图上\n"
             "不同斜率对应不同噪声类型")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (2,1) ── Composite signal ──
ax = axes[2, 1]
ax.plot(t, true_input, "k--", linewidth=1, alpha=0.6, label="真实值")
ax.plot(t, composite, "r", linewidth=0.6, alpha=0.8, label="含全部误差的测量值")
ax.set_ylabel("信号值"); ax.set_xlabel("时间 [s]")
ax.set_title("⑥ 综合: 真实 vs 测量\n偏置+白噪声+比例因子+偏置漂移 叠加")
ax.legend(fontsize=8)

fig.tight_layout(pad=2.0, w_pad=2.5, h_pad=3.0)

out = os.path.join(os.path.dirname(__file__), "outputs", "imu_error_model.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.abspath(out)}")
plt.close(fig)
