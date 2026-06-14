r"""IMU error modelling and correction visual guide.

Outputs four figures:
1. deterministic error model and calibration;
2. stochastic error processes and integration growth;
3. Allan deviation from a stationary synthetic rate sequence;
4. the complete correction chain used before INS/ESKF propagation.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

from imu_visualization_math import (
    allan_deviation,
    fit_allan_log_slope,
    overlapping_allan_deviation,
)

matplotlib.use("Agg")
_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "PingFang SC",
]
_AVAILABLE = {font.name for font in fm.fontManager.ttflist}
_CHINESE_FONT = next(
    (name for name in _CANDIDATES if name in _AVAILABLE), None
)
if _CHINESE_FONT:
    matplotlib.rcParams["font.sans-serif"] = [_CHINESE_FONT, "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(os.path.join(OUT, filename), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


def one_over_f_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a normalized real 1/f sequence for Allan-plot education."""
    frequencies = np.fft.rfftfreq(size)
    spectrum = rng.normal(size=frequencies.size) + 1j * rng.normal(
        size=frequencies.size
    )
    scale = np.zeros_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    sequence = np.fft.irfft(spectrum * scale, n=size)
    return (sequence - sequence.mean()) / sequence.std()


# ---------------------------------------------------------------------------
# Figure 1: deterministic errors and calibration
# ---------------------------------------------------------------------------

sample_rate = 200.0
dt = 1.0 / sample_rate
t = np.arange(0.0, 12.0, dt)
true_x = 2.0 + 1.5 * np.sin(2.0 * np.pi * 0.35 * t)
bias = 0.35
scale_factor = 0.04
raw_bias = true_x + bias
raw_scale = (1.0 + scale_factor) * true_x

true_three_axis = np.column_stack(
    [
        1.2 * np.sin(2.0 * np.pi * 0.3 * t),
        0.8 * np.cos(2.0 * np.pi * 0.2 * t),
        0.5 * np.sin(2.0 * np.pi * 0.15 * t + 0.4),
    ]
)
calibration_matrix = np.array(
    [[1.02, 0.015, -0.010], [-0.012, 0.98, 0.018], [0.008, -0.014, 1.03]]
)
bias_vector = np.array([0.12, -0.08, 0.05])
measured_three_axis = (
    calibration_matrix @ true_three_axis.T
).T + bias_vector
corrected_three_axis = (
    np.linalg.inv(calibration_matrix)
    @ (measured_three_axis - bias_vector).T
).T

fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

ax = axes[0, 0]
ax.plot(t, true_x, "k--", lw=1.2, label="真实输入")
ax.plot(t, raw_bias, color="tab:red", lw=1.0, label="测量值")
ax.plot(t, raw_bias - bias, color="tab:green", lw=1.0, label="减去标定 bias")
ax.set_title("① 零偏：与输入无关的加性误差")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("传感器输出")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[0, 1]
ax.plot(t[:800], true_x[:800], "k--", lw=1.2, label="真实输入")
ax.plot(t[:800], raw_scale[:800], color="tab:blue", lw=1.0, label="含 4% 比例因子")
ax.plot(
    t[:800],
    raw_scale[:800] / (1.0 + scale_factor),
    color="tab:green",
    lw=1.0,
    label="除以标定比例",
)
ax.fill_between(
    t[:800], true_x[:800], raw_scale[:800], color="tab:blue", alpha=0.12
)
ax.set_title("② 比例因子：误差随输入幅值增大")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("传感器输出")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1, 0]
axis_only = np.zeros((400, 3))
axis_only[:, 0] = np.linspace(-2.0, 2.0, axis_only.shape[0])
leaked = (calibration_matrix @ axis_only.T).T
ax.plot(axis_only[:, 0], leaked[:, 0], label="X 测量")
ax.plot(axis_only[:, 0], leaked[:, 1], label="泄漏到 Y")
ax.plot(axis_only[:, 0], leaked[:, 2], label="泄漏到 Z")
ax.axhline(0.0, color="black", lw=0.6)
ax.set_title("③ 非正交/交轴耦合：单轴输入泄漏到其他轴")
ax.set_xlabel("真实 X 输入")
ax.set_ylabel("三轴测量")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1, 1]
ax.plot(
    t[:1000],
    measured_three_axis[:1000, 0] - true_three_axis[:1000, 0],
    color="tab:red",
    lw=0.9,
    label="校正前 X 误差",
)
ax.plot(
    t[:1000],
    corrected_three_axis[:1000, 0] - true_three_axis[:1000, 0],
    color="tab:green",
    lw=1.2,
    label="校正后 X 误差",
)
ax.set_title("④ 统一标定模型：先减 bias，再乘逆标定矩阵")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("误差")
ax.legend()
ax.grid(alpha=0.25)
ax.text(
    0.02,
    0.96,
    "测量模型：y_m = K(T) y_true + b(T) + n\n"
    "离线校正：y_hat = inv(K(T)) [y_m - b(T)]\n"
    "K 同时包含比例因子、非正交和安装误差\n"
    "温度项需通过多温点标定建模",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

fig.suptitle(
    "IMU 确定性误差：测量模型、可观测现象与离线标定",
    fontsize=15,
    fontweight="bold",
)
save(fig, "imu_error_1_systematic.png")


# ---------------------------------------------------------------------------
# Figure 2: stochastic errors and integration
# ---------------------------------------------------------------------------

duration = 60.0
t_noise = np.arange(0.0, duration, dt)
white_rate = RNG.normal(0.0, 0.12, size=t_noise.size)
angle_random_walk = np.cumsum(white_rate) * dt

tau_c = 8.0
stationary_sigma = 0.025
alpha = np.exp(-dt / tau_c)
gm = np.zeros_like(t_noise)
gm[0] = RNG.normal(0.0, stationary_sigma)
drive_sigma = stationary_sigma * np.sqrt(1.0 - alpha**2)
for index in range(1, gm.size):
    gm[index] = alpha * gm[index - 1] + drive_sigma * RNG.normal()

bias_rw_drive = 0.002
bias_random_walk = np.cumsum(RNG.normal(size=t_noise.size))
bias_random_walk *= bias_rw_drive * np.sqrt(dt)

ensemble_size = 80
ensemble = np.cumsum(
    RNG.normal(0.0, 0.12, size=(ensemble_size, t_noise.size)), axis=1
) * dt
ensemble_std = ensemble.std(axis=0)

fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

ax = axes[0, 0]
ax.plot(t_noise[:1600], white_rate[:1600], color="0.35", lw=0.6)
ax.set_title("① 测量白噪声：逐样本不可预测，均值近零")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("速率噪声")
ax.grid(alpha=0.25)

ax = axes[0, 1]
for trajectory in ensemble[:16]:
    ax.plot(t_noise, trajectory, color="tab:blue", alpha=0.15, lw=0.6)
ax.plot(t_noise, ensemble_std, color="tab:red", lw=2.0, label="蒙特卡洛标准差")
theory = 0.12 * np.sqrt(np.maximum(t_noise, dt) * dt)
ax.plot(t_noise, theory, "k--", lw=1.2, label="理论 ∝ √t")
ax.set_title("② 白噪声积分：角度/速度随机游走")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("积分误差")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1, 0]
ax.plot(t_noise, gm, color="tab:orange", lw=0.8, label="一阶 Gauss-Markov")
ax.axhline(stationary_sigma, color="tab:red", ls=":", lw=0.8)
ax.axhline(-stationary_sigma, color="tab:red", ls=":", lw=0.8)
ax.set_title(f"③ 相关 bias：有限相关时间 τ={tau_c:.0f}s，不会无界发散")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("bias")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1, 1]
ax.plot(t_noise, bias_random_walk, color="tab:purple", lw=0.9)
ax.set_title("④ Bias 随机游走：驱动噪声积分后无界漂移")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("bias")
ax.grid(alpha=0.25)
ax.text(
    0.02,
    0.96,
    "ESKF 中的处理：\n"
    "• 白噪声进入过程噪声 Qc\n"
    "• db/dt = n_b 建模 bias 随机游走\n"
    "• GNSS 等外部观测通过交叉协方差估计 bias\n"
    "随机误差不能逐样本“校准掉”，只能统计建模与估计",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

fig.suptitle(
    "IMU 随机误差：从测量噪声到积分漂移与 bias 状态模型",
    fontsize=15,
    fontweight="bold",
)
save(fig, "imu_error_2_random.png")


# ---------------------------------------------------------------------------
# Figure 3: Allan deviation
# ---------------------------------------------------------------------------

allan_rate = 100.0
allan_dt = 1.0 / allan_rate
# Two hours gives enough clusters to show the long-tau rising branch without
# making the teaching curve depend on a handful of endpoint clusters.
allan_duration = 7200.0
allan_size = int(allan_rate * allan_duration)
white_density = 0.012
white_component = RNG.normal(
    0.0, white_density / np.sqrt(allan_dt), size=allan_size
)
flicker_component = 0.003 * one_over_f_noise(allan_size, RNG)
random_walk_component = np.cumsum(RNG.normal(size=allan_size))
random_walk_component *= 5.0e-4 * np.sqrt(allan_dt)
rate_sequence = white_component + flicker_component + random_walk_component

requested_taus = np.logspace(-1, 2.8, 85)
taus_non, total_ad_non = allan_deviation(
    rate_sequence, allan_rate, requested_taus
)
taus, total_ad, overlapping_pairs = overlapping_allan_deviation(
    rate_sequence, allan_rate, requested_taus
)
_, white_ad, _ = overlapping_allan_deviation(
    white_component, allan_rate, requested_taus
)
_, flicker_ad, _ = overlapping_allan_deviation(
    flicker_component, allan_rate, requested_taus
)
_, random_walk_ad, _ = overlapping_allan_deviation(
    random_walk_component, allan_rate, requested_taus
)

fig, (ax, ax_count) = plt.subplots(
    2,
    1,
    figsize=(14, 10),
    gridspec_kw={"height_ratios": [4.2, 1.15]},
    constrained_layout=True,
)
ax.loglog(taus, total_ad, color="black", lw=2.2, label="重叠 Allan：主估计")
ax.loglog(
    taus_non,
    total_ad_non,
    color="0.45",
    ls=":",
    lw=1.2,
    label="非重叠 Allan：教学对照",
)
ax.loglog(taus, white_ad, color="tab:blue", lw=1.0, alpha=0.8, label="白噪声分量")
ax.loglog(
    taus, flicker_ad, color="tab:orange", lw=1.0, alpha=0.8, label="1/f 相关 bias 分量"
)
ax.loglog(
    taus,
    random_walk_ad,
    color="tab:purple",
    lw=1.0,
    alpha=0.8,
    label="速率随机游走分量",
)

short_anchor = np.argmin(np.abs(taus - 0.4))
short_tau = np.array([0.12, 1.5])
short_ref = total_ad[short_anchor] * (short_tau / taus[short_anchor]) ** -0.5
ax.loglog(short_tau, short_ref, "b--", lw=1.4, label="参考斜率 −1/2")

long_anchor = np.argmin(np.abs(taus - 150.0))
long_tau = np.array([50.0, 500.0])
long_ref = total_ad[long_anchor] * (long_tau / taus[long_anchor]) ** 0.5
ax.loglog(long_tau, long_ref, color="tab:purple", ls="--", lw=1.4, label="参考斜率 +1/2")

log_slope = np.gradient(np.log10(total_ad), np.log10(taus))
plateau_candidates = np.where((taus >= 25.0) & (taus <= 70.0))[0]
plateau_index = plateau_candidates[
    np.argmin(np.abs(log_slope[plateau_candidates]))
]
ax.scatter(
    taus[plateau_index],
    total_ad[plateau_index],
    color="tab:orange",
    s=55,
    zorder=5,
)
ax.annotate(
    "中等 τ：相关 bias / 1/f 噪声形成近水平区",
    xy=(taus[plateau_index], total_ad[plateau_index]),
    xytext=(2.0, 1.8 * total_ad[plateau_index]),
    arrowprops=dict(arrowstyle="->", color="tab:orange"),
    fontsize=9,
)

ax.axvspan(0.1, 10.0, color="tab:blue", alpha=0.055)
ax.axvspan(10.0, 70.0, color="tab:orange", alpha=0.055)
ax.axvspan(70.0, 630.0, color="tab:purple", alpha=0.045)
ax.text(0.18, ax.get_ylim()[1] / 1.5, "白噪声主导\n斜率约 -1/2", color="tab:blue")
ax.text(13.0, ax.get_ylim()[1] / 1.5, "bias instability 区\n斜率约 0", color="tab:orange")
ax.text(90.0, ax.get_ylim()[1] / 1.5, "rate random walk 主导\n斜率约 +1/2", color="tab:purple")

ax.set_xlabel("聚类时间 τ [s]")
ax.set_ylabel("Allan 标准差 [速率单位]")
ax.set_title(
    "典型近似 U 型 Allan 标准差：白噪声下降、bias 平台、随机游走上升\n"
    "必须使用静止、去趋势的速率数据；区间边界取决于各噪声量级"
)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=9, ncol=2)
ax.text(
    0.02,
    0.04,
    "斜率解释：−1 为量化噪声，−1/2 为白速率噪声，"
    "0 为 flicker/bias instability，+1/2 为速率随机游走。\n"
    "常值 bias 不影响 Allan 方差；真实运动和趋势会污染长 τ 区域。",
    transform=ax.transAxes,
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
)

cluster_sizes_non = np.rint(taus_non * allan_rate).astype(int)
nonoverlap_pairs = allan_size // cluster_sizes_non - 1
ax_count.loglog(
    taus,
    overlapping_pairs,
    color="black",
    lw=1.8,
    label="重叠差分对数量",
)
ax_count.loglog(
    taus_non,
    nonoverlap_pairs,
    color="0.45",
    ls=":",
    lw=1.5,
    label="非重叠差分对数量",
)
ax_count.axhline(20, color="tab:red", ls="--", lw=1.0, label="示例最低门槛 20 对")
ax_count.set_xlabel("聚类时间 τ [s]")
ax_count.set_ylabel("差分对数量")
ax_count.set_title("长 τ 端必须同时检查有效统计样本数")
ax_count.grid(True, which="both", alpha=0.25)
ax_count.legend(fontsize=8, ncol=3)
save(fig, "imu_error_3_allan.png")

short_fit = fit_allan_log_slope(taus, total_ad, 0.15, 1.0)
middle_fit = fit_allan_log_slope(taus, total_ad, 30.0, 70.0)
long_fit = fit_allan_log_slope(taus, total_ad, 80.0, 400.0)
print(
    "Allan local slopes: "
    f"short={short_fit['slope']:.3f}, "
    f"middle={middle_fit['slope']:.3f}, "
    f"long={long_fit['slope']:.3f}"
)


# ---------------------------------------------------------------------------
# Figure 4: correction chain
# ---------------------------------------------------------------------------

raw_rmse = np.sqrt(np.mean((measured_three_axis - true_three_axis) ** 2, axis=0))
corrected_rmse = np.sqrt(
    np.mean((corrected_three_axis - true_three_axis) ** 2, axis=0)
)

fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

ax = axes[0, 0]
ax.axis("off")
ax.text(
    0.02,
    0.96,
    "离线标定层\n\n"
    "原始码值\n"
    "  ↓ 单位、符号、轴顺序\n"
    "温度模型 b(T), K(T)\n"
    "  ↓ y_cal = inv(K(T)) [y_raw - b(T)]\n"
    "得到物理单位的角速度/比力\n\n"
    "典型可标定项：\n"
    "bias、比例因子、非正交、安装角、温度项\n"
    "陀螺 g-sensitivity 需要额外模型和激励",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    linespacing=1.45,
    bbox=dict(boxstyle="round", facecolor="#e8f4ff", alpha=0.95),
)

ax = axes[0, 1]
ax.axis("off")
ax.text(
    0.02,
    0.96,
    "在线导航层（增量型 IMU）\n\n"
    "每个子样先去 bias：\n"
    "Δθ_corr = Δθ_meas - b_g Δt\n"
    "Δv_corr = Δv_meas - b_a Δt\n\n"
    "两个子样再做不可交换补偿：\n"
    "• coning：角 × 角\n"
    "• sculling：角 × 线\n"
    "• rotation：平均姿态旋转\n\n"
    "最后：body → nav、加重力、积分位置速度",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    linespacing=1.45,
    bbox=dict(boxstyle="round", facecolor="#effbea", alpha=0.95),
)

ax = axes[1, 0]
x_positions = np.arange(3)
width = 0.34
ax.bar(x_positions - width / 2, raw_rmse, width, color="tab:red", label="校正前")
ax.bar(
    x_positions + width / 2,
    corrected_rmse,
    width,
    color="tab:green",
    label="校正后",
)
ax.set_xticks(x_positions, ["X", "Y", "Z"])
ax.set_ylabel("RMSE")
ax.set_title("确定性标定可显著降低系统误差")
ax.legend()
ax.grid(axis="y", alpha=0.25)

ax = axes[1, 1]
ax.axis("off")
ax.text(
    0.02,
    0.96,
    "哪些能“改正”，哪些只能“估计”？\n\n"
    "离线确定性补偿：\n"
    "- 固定 bias、比例因子、非正交、温度曲线\n\n"
    "在线状态估计：\n"
    "- 通电后 bias、缓慢漂移、时间变化误差\n\n"
    "统计进入 Q/P：\n"
    "- 白噪声、bias 随机游走、相关噪声\n\n"
    "机械编排补偿：\n"
    "- 圆锥、划桨、旋转效应\n\n"
    "关键原则：不要用同一个“bias”概念混指\n"
    "标定常值、随机过程状态和单次白噪声。",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    linespacing=1.4,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
)

fig.suptitle(
    "IMU 改正全链路：离线标定、在线 bias、随机噪声与机械编排",
    fontsize=15,
    fontweight="bold",
)
save(fig, "imu_error_4_correction_pipeline.png")
