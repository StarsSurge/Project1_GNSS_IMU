"""MVP scaffold for a two-sample IMU nominal state update.

This file is intentionally an algorithm-prototype workspace.  The stable base
components live in ``gnss_imu.imu_mechanization``; the state propagation body is
left for the learner to implement step by step.

Run from the repository root:

    $env:PYTHONPATH = "$PWD\\python"
    .\\.venv\\Scripts\\python.exe python\\examples\\demo_imu_state_update_mvp.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from gnss_imu import (
    IMUIncrement,
    NavigationState,
    correct_two_sample_increments,
    euler_zyx_to_quat,
    normalize_quat,
    quat_multiply,
    quat_to_dcm,
    rotvec_to_quat,
)

GRAVITY_NED = np.array([0.0, 0.0, 9.80665])


def propagate_two_sample_mvp(
    state: NavigationState,
    imu1: IMUIncrement,
    imu2: IMUIncrement,
    gravity_ned: np.ndarray = GRAVITY_NED,
) -> NavigationState:
    """Prototype two-sample strapdown state propagation.

    Implement this function as the current MVP exercise.  Keep the code
    explicit and close to the derivation before promoting it into the formal
    library module.
    """
    gravity_ned = np.asarray(gravity_ned, dtype=float).reshape(3)
    if not np.all(np.isfinite(gravity_ned)):
        raise ValueError("gravity_ned must contain only finite values")

    # 去除 IMU 零偏，并做双子样圆锥、划桨和旋转效应修正。
    # The result is still an increment: dtheta [rad], dvel [m/s].
    corrected = correct_two_sample_increments(
        imu1,
        imu2,
        b_g=state.b_g,
        b_a=state.b_a,
    )

    # 将修正后的机体系旋转向量转换成小旋转四元数。
    # corrected.dtheta 表示这一个导航周期内 body frame 的累计转角。
    dq_body = rotvec_to_quat(corrected.dtheta)

    # 姿态采用右乘更新：q_new = q_old ⊗ dq_body。
    # 因为 dq_body 是在机体系下表达的小旋转增量。
    q_new = normalize_quat(quat_multiply(state.q_bn, dq_body), "q_new")

    # 将修正后的机体系速度增量旋转到 NED 导航系。
    # MVP 中先使用更新前的姿态近似整个短时间区间的姿态。
    c_bn = quat_to_dcm(state.q_bn)
    dvel_n = c_bn @ corrected.dvel

    # 加速度计测量的是比力 specific force，不是导航系加速度。
    # 因此把比力积分量转到 NED 后，还需要单独加回重力项。
    v_new = state.v_n + dvel_n + gravity_ned * corrected.dt

    # 用梯形积分更新位置，比简单欧拉积分更符合“速度在周期内变化”的情况。
    p_new = state.p_n + 0.5 * (state.v_n + v_new) * corrected.dt

    return NavigationState(
        p_n=p_new,
        v_n=v_new,
        q_bn=q_new,
        b_a=state.b_a,
        b_g=state.b_g,
    )


def make_static_initial_state() -> NavigationState:
    """Known initial state for the first static smoke test."""
    return NavigationState(
        p_n=np.zeros(3),
        v_n=np.zeros(3),
        q_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )


def make_static_two_sample_imu(
    dt_sub: float = 0.005,
    gravity: float = 9.80665,
) -> tuple[IMUIncrement, IMUIncrement]:
    """Create two static IMU increment samples for an identity attitude.

    Static accelerometers measure specific force, not acceleration.  In NED
    with body aligned to navigation, a stationary sensor has body delta-v
    approximately [0, 0, -g * dt].
    """
    static_dvel = np.array([0.0, 0.0, -gravity * dt_sub])
    imu1 = IMUIncrement(
        dtheta=np.zeros(3),
        dvel=static_dvel,
        dt=dt_sub,
    )
    imu2 = IMUIncrement(
        dtheta=np.zeros(3),
        dvel=static_dvel,
        dt=dt_sub,
    )
    return imu1, imu2


def make_dataset1_measured_case() -> tuple[NavigationState, IMUIncrement, IMUIncrement]:
    """Use two consecutive measured IMU frames from ``data/dataset1``.

    Source rows:
        - ``Leador-A15.txt`` line 10001, time 456300.004412029 s
        - ``Leador-A15.txt`` line 10002, time 456300.009412029 s

    Initial state source:
        - ``truth.nav`` first row, time 456300.004412 s

    The IMU file already stores increments: delta angle [rad] and delta
    velocity [m/s].  The per-sample interval is about 0.005 s.
    """
    state0 = NavigationState(
        p_n=np.zeros(3),
        v_n=np.array([0.0003, -0.0009, -0.0009]),
        q_bn=euler_zyx_to_quat(
            roll=0.85266,
            pitch=-2.03401,
            yaw=185.67273,
            degrees=True,
        ),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )

    imu1 = IMUIncrement(
        dtheta=np.array([0.0000043453, -0.0000018374, -0.0000012908]),
        dvel=np.array([-0.0023051314, -0.0010445054, -0.0488077663]),
        dt=0.005,
    )
    imu2 = IMUIncrement(
        dtheta=np.array([0.0000002888, -0.0000027108, -0.0000018521]),
        dvel=np.array([-0.0021418147, -0.0012695923, -0.0491048507]),
        dt=0.005,
    )
    return state0, imu1, imu2


def print_state(label: str, state: NavigationState) -> None:
    """Print a compact state summary."""
    print(label)
    print(f"  p_n  [m]     : {state.p_n}")
    print(f"  v_n  [m/s]   : {state.v_n}")
    print(f"  q_bn [wxyz]  : {state.q_bn}")
    print(f"  b_a  [m/s^2] : {state.b_a}")
    print(f"  b_g  [rad/s] : {state.b_g}")


def main() -> None:
    """Prepare the MVP inputs and show the expected first validation."""
    state0 = make_static_initial_state()
    imu1, imu2 = make_static_two_sample_imu()

    print("=" * 72)
    print("IMU state update MVP scaffold")
    print("=" * 72)
    print_state("Initial state:", state0)
    print("\nTwo IMU increment samples:")
    print(f"  imu1: {asdict(imu1)}")
    print(f"  imu2: {asdict(imu2)}")

    print("\nExpected after you implement propagate_two_sample_mvp():")
    print("  p_n close to [0, 0, 0]")
    print("  v_n close to [0, 0, 0]")
    print("  q_bn close to [1, 0, 0, 0]")

    try:
        state1 = propagate_two_sample_mvp(state0, imu1, imu2)
    except NotImplementedError as exc:
        print("\nImplementation status:")
        print(f"  {exc}")
        return

    print_state("\nUpdated state:", state1)
    print("\nStatic residual checks:")
    print(f"  |p_n|       : {np.linalg.norm(state1.p_n):.6e} m")
    print(f"  |v_n|       : {np.linalg.norm(state1.v_n):.6e} m/s")
    print(f"  |norm(q)-1| : {abs(np.linalg.norm(state1.q_bn) - 1.0):.6e}")

    measured_state0, measured_imu1, measured_imu2 = make_dataset1_measured_case()
    print("\n" + "=" * 72)
    print("Dataset1 measured two-frame IMU input")
    print("=" * 72)
    print_state("Measured-case initial state from truth.nav:", measured_state0)
    print("\nMeasured IMU increment samples from Leador-A15.txt:")
    print(f"  imu1: {asdict(measured_imu1)}")
    print(f"  imu2: {asdict(measured_imu2)}")

    measured_state1 = propagate_two_sample_mvp(
        measured_state0,
        measured_imu1,
        measured_imu2,
    )
    print_state("\nMeasured-case updated state:", measured_state1)
    print("\nMeasured-case delta summary:")
    print(f"  dp_n [m]   : {measured_state1.p_n - measured_state0.p_n}")
    print(f"  dv_n [m/s] : {measured_state1.v_n - measured_state0.v_n}")
    print(f"  q norm     : {np.linalg.norm(measured_state1.q_bn):.12f}")


if __name__ == "__main__":
    main()
