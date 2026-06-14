r"""Sculling and rotation-effect visualizations with navigation-frame truth."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

from imu_visualization_math import (
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
    sculling_rotation_correct,
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


# ---------------------------------------------------------------------------
# Sculling experiment: x-axis angular vibration and z-axis specific force
# ---------------------------------------------------------------------------

duration = 3.0
fine_dt = 0.0002
nav_dt = 0.01
frequency_hz = 4.0
angular_frequency = 2.0 * np.pi * frequency_hz
angle_amplitude = np.deg2rad(2.5)
force_amplitude = 3.0
t = np.arange(0.0, duration, fine_dt)

omega_x = angle_amplitude * angular_frequency * np.cos(angular_frequency * t)
specific_force_z = force_amplitude * np.cos(
    angular_frequency * t + np.pi / 2.0
)
omega_body = np.column_stack(
    [omega_x, np.zeros_like(t), np.zeros_like(t)]
)
specific_force_body = np.column_stack(
    [np.zeros_like(t), np.zeros_like(t), specific_force_z]
)

# Fine truth: rotate every infinitesimal body delta-v into navigation frame.
q_truth = np.array([1.0, 0.0, 0.0, 0.0])
v_truth = np.zeros(3)
p_truth = np.zeros(3)
truth_velocity_history = np.empty((t.size, 3))
truth_position_history = np.empty((t.size, 3))
for index in range(t.size):
    c_bn = quat_to_dcm(q_truth)
    velocity_previous = v_truth.copy()
    v_truth += c_bn @ (specific_force_body[index] * fine_dt)
    p_truth += 0.5 * (velocity_previous + v_truth) * fine_dt
    q_truth = quat_multiply(
        q_truth, rotvec_to_quat(omega_body[index] * fine_dt)
    )
    q_truth /= np.linalg.norm(q_truth)
    truth_velocity_history[index] = v_truth
    truth_position_history[index] = p_truth

samples_per_nav = int(round(nav_dt / fine_dt))
half_samples = samples_per_nav // 2
nav_steps = t.size // samples_per_nav
nav_time = (np.arange(nav_steps) + 1) * nav_dt

q_nav = np.array([1.0, 0.0, 0.0, 0.0])
velocity_simple = np.zeros(3)
velocity_corrected = np.zeros(3)
position_simple = np.zeros(3)
position_corrected = np.zeros(3)
simple_history = np.empty((nav_steps, 3))
corrected_history = np.empty((nav_steps, 3))
simple_position_history = np.empty((nav_steps, 3))
corrected_position_history = np.empty((nav_steps, 3))
truth_velocity_nav = np.empty((nav_steps, 3))
truth_position_nav = np.empty((nav_steps, 3))
sculling_components = np.empty((nav_steps, 3))
rotation_components = np.empty((nav_steps, 3))

for epoch in range(nav_steps):
    start = epoch * samples_per_nav
    middle = start + half_samples
    end = start + samples_per_nav
    dtheta1 = omega_body[start:middle].sum(axis=0) * fine_dt
    dtheta2 = omega_body[middle:end].sum(axis=0) * fine_dt
    dvel1 = specific_force_body[start:middle].sum(axis=0) * fine_dt
    dvel2 = specific_force_body[middle:end].sum(axis=0) * fine_dt

    c_old = quat_to_dcm(q_nav)
    simple_delta_velocity = c_old @ (dvel1 + dvel2)
    corrected_body, sculling, rotation = sculling_rotation_correct(
        dtheta1, dvel1, dtheta2, dvel2
    )
    corrected_delta_velocity = c_old @ corrected_body

    old_simple = velocity_simple.copy()
    old_corrected = velocity_corrected.copy()
    velocity_simple += simple_delta_velocity
    velocity_corrected += corrected_delta_velocity
    position_simple += 0.5 * (old_simple + velocity_simple) * nav_dt
    position_corrected += 0.5 * (
        old_corrected + velocity_corrected
    ) * nav_dt

    q_nav = quat_multiply(
        q_nav, rotvec_to_quat(dtheta1 + dtheta2)
    )
    q_nav /= np.linalg.norm(q_nav)

    simple_history[epoch] = velocity_simple
    corrected_history[epoch] = velocity_corrected
    simple_position_history[epoch] = position_simple
    corrected_position_history[epoch] = position_corrected
    truth_velocity_nav[epoch] = truth_velocity_history[end - 1]
    truth_position_nav[epoch] = truth_position_history[end - 1]
    sculling_components[epoch] = c_old @ sculling
    rotation_components[epoch] = c_old @ rotation

velocity_error_simple = simple_history - truth_velocity_nav
velocity_error_corrected = corrected_history - truth_velocity_nav
position_error_simple = simple_position_history - truth_position_nav
position_error_corrected = corrected_position_history - truth_position_nav


# ---------------------------------------------------------------------------
# Figure 1: sculling mechanism and impact
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

window = t <= 0.8
ax = axes[0, 0]
line1 = ax.plot(
    t[window], np.rad2deg(omega_x[window]), color="tab:red", label="ωx"
)
ax2 = ax.twinx()
line2 = ax2.plot(
    t[window],
    specific_force_z[window],
    color="tab:blue",
    label="fz",
)
ax.set_xlabel("时间 [s]")
ax.set_ylabel("角速度 [deg/s]", color="tab:red")
ax2.set_ylabel("比力 [m/s²]", color="tab:blue")
ax.set_title("① 角振动与线振动相差 90°")
ax.legend(line1 + line2, ["ωx", "fz"], loc="upper right")
ax.grid(alpha=0.25)

ax = axes[0, 1]
epochs = np.arange(min(80, nav_steps))
ax.plot(
    epochs,
    sculling_components[: epochs.size, 1] * 1e6,
    color="tab:purple",
    label="sculling 项",
)
ax.plot(
    epochs,
    rotation_components[: epochs.size, 1] * 1e6,
    color="tab:orange",
    label="rotation 项",
)
ax.set_title("② 每历元两个不同来源的 Y 向速度修正")
ax.set_xlabel("导航历元")
ax.set_ylabel("Δv 修正 [µm/s]")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1, 0]
ax.plot(
    nav_time,
    velocity_error_simple[:, 1] * 1000.0,
    color="0.45",
    label="简单累加",
)
ax.plot(
    nav_time,
    velocity_error_corrected[:, 1] * 1000.0,
    color="tab:red",
    label="两子样修正",
)
ax.set_title("③ 相对细积分真值的 Y 速度误差")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("速度误差 [mm/s]")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1, 1]
ax.plot(
    nav_time,
    position_error_simple[:, 1] * 1000.0,
    color="0.45",
    label="简单累加",
)
ax.plot(
    nav_time,
    position_error_corrected[:, 1] * 1000.0,
    color="tab:red",
    label="两子样修正",
)
ax.set_title("④ 速度误差积分成 Y 位置误差")
ax.set_xlabel("时间 [s]")
ax.set_ylabel("位置误差 [mm]")
ax.legend()
ax.grid(alpha=0.25)
ax.text(
    0.02,
    0.96,
    "Δv_corr = Δv_1 + Δv_2\n"
    "      + 2/3(Δθ_1×Δv_2 + Δv_1×Δθ_2)\n"
    "      + ½(Δθ×Δv)\n\n"
    "补偿后的 body Δv 仍必须用旧姿态 C_bn\n"
    "旋转到导航系后才能与真值比较。",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

fig.suptitle(
    "划桨效应：角振动 × 线振动产生正交方向净速度",
    fontsize=15,
    fontweight="bold",
)
save(fig, "sculling_1_mechanism.png")


# ---------------------------------------------------------------------------
# Rotation-effect experiment: constant turn and constant body specific force
# ---------------------------------------------------------------------------

turn_rate = np.deg2rad(60.0)
body_force = np.array([2.0, 0.0, 0.0])
step_sizes = np.logspace(-3, -0.7, 35)
simple_errors = []
rotation_errors = []
exact_vectors = []
simple_vectors = []
corrected_vectors = []

for step in step_sizes:
    local_dt = min(1e-5, step / 500.0)
    local_count = int(np.ceil(step / local_dt))
    local_dt = step / local_count
    exact_delta = np.zeros(3)
    q_local = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(local_count):
        exact_delta += quat_to_dcm(q_local) @ (body_force * local_dt)
        q_local = quat_multiply(
            q_local, rotvec_to_quat([0.0, 0.0, turn_rate * local_dt])
        )
    dtheta = np.array([0.0, 0.0, turn_rate * step])
    dvel = body_force * step
    simple_delta = dvel
    corrected_delta = dvel + 0.5 * np.cross(dtheta, dvel)
    simple_errors.append(np.linalg.norm(simple_delta - exact_delta))
    rotation_errors.append(np.linalg.norm(corrected_delta - exact_delta))
    exact_vectors.append(exact_delta)
    simple_vectors.append(simple_delta)
    corrected_vectors.append(corrected_delta)

simple_errors = np.asarray(simple_errors)
rotation_errors = np.asarray(rotation_errors)
example_index = np.argmin(abs(step_sizes - 0.1))
example_exact = exact_vectors[example_index]
example_simple = simple_vectors[example_index]
example_corrected = corrected_vectors[example_index]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

ax = axes[0]
for vector, color, label in [
    (example_exact, "black", "细积分真值"),
    (example_simple, "0.55", "忽略旋转"),
    (example_corrected, "tab:orange", "½(Δθ×Δv) 修正"),
]:
    ax.quiver(
        0,
        0,
        vector[0],
        vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        label=label,
    )
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-0.01, max(example_exact[0], example_simple[0]) * 1.15)
ax.set_ylim(-0.01, max(example_exact[1], example_corrected[1]) * 1.35)
ax.set_title(f"① {step_sizes[example_index]*1000:.0f} ms 内 Δv 方向在旋转")
ax.set_xlabel("导航系 X 速度增量 [m/s]")
ax.set_ylabel("导航系 Y 速度增量 [m/s]")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1]
ax.loglog(step_sizes * 1000.0, simple_errors, label="忽略 rotation")
ax.loglog(
    step_sizes * 1000.0,
    rotation_errors,
    color="tab:orange",
    label="一阶 rotation 修正",
)
ax.set_title("② 导航周期越长，旋转效应越明显")
ax.set_xlabel("导航周期 [ms]")
ax.set_ylabel("单步 Δv 误差 [m/s]")
ax.legend()
ax.grid(True, which="both", alpha=0.25)

ax = axes[2]
ax.axis("off")
ax.text(
    0.02,
    0.96,
    "Rotation effect 的来源\n\n"
    "同一导航周期内，body 坐标系持续旋转。\n"
    "早期和晚期采到的 Δv 虽都写在 body 坐标中，\n"
    "对应的导航系方向并不相同。\n\n"
    "常角速度、常比力的一阶积分：\n"
    "Δvⁿ ≈ C_old [Δv + ½(Δθ×Δv)]\n\n"
    "与 sculling 的区别：\n"
    "• rotation：总角运动导致平均方向变化\n"
    "• sculling：子样内角/线变化相关造成额外二阶项\n"
    "二者都要加，但物理来源不能混为一谈。",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    linespacing=1.45,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
)

fig.suptitle(
    "旋转效应：一个导航周期内 body 坐标系方向发生变化",
    fontsize=15,
    fontweight="bold",
)
save(fig, "sculling_2_rotation.png")


# ---------------------------------------------------------------------------
# Figure 3: summary and quantitative checks
# ---------------------------------------------------------------------------

simple_velocity_rmse = np.sqrt(np.mean(velocity_error_simple[:, 1] ** 2))
corrected_velocity_rmse = np.sqrt(
    np.mean(velocity_error_corrected[:, 1] ** 2)
)
simple_position_final = abs(position_error_simple[-1, 1])
corrected_position_final = abs(position_error_corrected[-1, 1])

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

ax = axes[0]
ax.bar(
    ["简单累加", "两子样"],
    [simple_velocity_rmse * 1000.0, corrected_velocity_rmse * 1000.0],
    color=["0.5", "tab:red"],
)
ax.set_yscale("log")
ax.set_ylabel("Y 速度 RMSE [mm/s]")
ax.set_title("划桨实验：速度误差")
ax.grid(axis="y", alpha=0.25)

ax = axes[1]
ax.bar(
    ["简单累加", "两子样"],
    [simple_position_final * 1000.0, corrected_position_final * 1000.0],
    color=["0.5", "tab:red"],
)
ax.set_yscale("log")
ax.set_ylabel("末端 Y 位置误差 [mm]")
ax.set_title("划桨实验：位置误差")
ax.grid(axis="y", alpha=0.25)

ax = axes[2]
ax.axis("off")
velocity_improvement = simple_velocity_rmse / max(
    corrected_velocity_rmse, 1e-15
)
ax.text(
    0.02,
    0.96,
    "不可交换速度误差总结\n\n"
    "Coning：角 × 角 → 姿态误差\n"
    "Sculling：子样角 × 子样线 → 速度误差\n"
    "Rotation：总角 × 总线 → 平均方向修正\n\n"
    f"本划桨实验速度 RMSE 改善：{velocity_improvement:.1f}×\n\n"
    "工程边界：\n"
    "• 两子样公式依赖等间隔和线性变化假设\n"
    "• 高频强振动可能需要多子样或更高阶算法\n"
    "• 先做 bias/scale 校正，再做交叉项补偿\n"
    "• 最后才将 Δv 从 body 转到 navigation",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    linespacing=1.45,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
)

fig.suptitle(
    "划桨与旋转补偿：定量效果、适用条件和正确处理顺序",
    fontsize=15,
    fontweight="bold",
)
save(fig, "sculling_3_impact.png")

position_improvement = simple_position_final / max(
    corrected_position_final, 1e-15
)
print(
    "Sculling velocity RMSE [mm/s]: "
    f"simple={simple_velocity_rmse * 1000.0:.6f}, "
    f"corrected={corrected_velocity_rmse * 1000.0:.6f}; "
    f"improvement={velocity_improvement:.1f}x"
)
print(
    "Sculling final position error [mm]: "
    f"simple={simple_position_final * 1000.0:.6f}, "
    f"corrected={corrected_position_final * 1000.0:.6f}; "
    f"improvement={position_improvement:.1f}x"
)
