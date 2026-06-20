"""Run the production-oriented GNSS/IMU loose ESKF baseline on dataset1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from gnss_imu import (  # noqa: E402
    ESKFConfig,
    ESKFState,
    GNSSPositionMeasurement,
    IMUNoiseModel,
    LooselyCoupledESKF,
    TimedIMUIncrement,
    default_initial_covariance,
    euler_zyx_to_quat,
    geodetic_to_ned,
    load_dataset1,
    quat_to_dcm,
    rpy_deg_to_body_to_ned,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay dataset1 through a 15-state GNSS/IMU loose ESKF."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "dataset1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_eskf",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Replay duration from truth start; <=0 processes all available data.",
    )
    parser.add_argument(
        "--imu-profile",
        choices=("mems", "navigation-grade"),
        default="navigation-grade",
    )
    parser.add_argument(
        "--lever-arm-b-m",
        type=float,
        nargs=3,
        metavar=("FORWARD", "RIGHT", "DOWN"),
        default=(0.14722696, -0.29821683, -0.18079014),
        help="Calibrated GNSS antenna minus IMU lever arm in body FRD [m].",
    )
    return parser.parse_args()


def _interpolate_truth(data, query_time_s: np.ndarray) -> dict[str, np.ndarray]:
    truth = data.truth
    return {
        "latitude_deg": np.interp(query_time_s, truth.time_s, truth.latitude_deg),
        "longitude_deg": np.interp(query_time_s, truth.time_s, truth.longitude_deg),
        "height_m": np.interp(query_time_s, truth.time_s, truth.height_m),
        "velocity_ned_mps": np.column_stack(
            [
                np.interp(query_time_s, truth.time_s, truth.velocity_ned_mps[:, i])
                for i in range(3)
            ]
        ),
        "attitude_rpy_deg": np.column_stack(
            [
                np.interp(query_time_s, truth.time_s, truth.attitude_rpy_deg[:, i])
                for i in range(3)
            ]
        ),
    }


def _write_outputs(
    output_dir: Path,
    history: dict[str, list],
    data,
    eskf: LooselyCoupledESKF,
    skipped_gnss: int,
    profile_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(history["time_s"])
    latitude_deg = np.rad2deg(np.asarray(history["latitude_rad"]))
    longitude_deg = np.rad2deg(np.asarray(history["longitude_rad"]))
    height_m = np.asarray(history["height_m"])
    velocity = np.asarray(history["velocity_ned_mps"])
    quaternion = np.asarray(history["q_bn"])
    accel_bias = np.asarray(history["accel_bias_mps2"])
    gyro_bias = np.asarray(history["gyro_bias_rps"])
    covariance_std = np.asarray(history["covariance_std"])

    truth = _interpolate_truth(data, times)
    reference_llh = (
        float(truth["latitude_deg"][0]),
        float(truth["longitude_deg"][0]),
        float(truth["height_m"][0]),
    )
    estimate_ned = geodetic_to_ned(
        latitude_deg,
        longitude_deg,
        height_m,
        reference_llh,
    )
    truth_ned = geodetic_to_ned(
        truth["latitude_deg"],
        truth["longitude_deg"],
        truth["height_m"],
        reference_llh,
    )
    position_error = estimate_ned - truth_ned
    velocity_error = velocity - truth["velocity_ned_mps"]

    estimate_dcm = np.stack([quat_to_dcm(q) for q in quaternion])
    truth_dcm = rpy_deg_to_body_to_ned(
        truth["attitude_rpy_deg"][:, 0],
        truth["attitude_rpy_deg"][:, 1],
        truth["attitude_rpy_deg"][:, 2],
    )
    relative_dcm = np.einsum("nij,njk->nik", np.swapaxes(truth_dcm, 1, 2), estimate_dcm)
    cosine_angle = np.clip((np.trace(relative_dcm, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    attitude_error_deg = np.rad2deg(np.arccos(cosine_angle))

    output_table = np.column_stack(
        [
            times,
            latitude_deg,
            longitude_deg,
            height_m,
            velocity,
            quaternion,
            accel_bias,
            gyro_bias,
            covariance_std,
            position_error,
            velocity_error,
            attitude_error_deg,
        ]
    )
    header = ",".join(
        [
            "time_s", "latitude_deg", "longitude_deg", "height_m",
            "vn_mps", "ve_mps", "vd_mps", "qw", "qx", "qy", "qz",
            "ba_x_mps2", "ba_y_mps2", "ba_z_mps2",
            "bg_x_rps", "bg_y_rps", "bg_z_rps",
            *[f"state_std_{i}" for i in range(15)],
            "error_n_m", "error_e_m", "error_d_m",
            "error_vn_mps", "error_ve_mps", "error_vd_mps",
            "attitude_error_deg",
        ]
    )
    np.savetxt(
        output_dir / "eskf_solution.csv",
        output_table,
        delimiter=",",
        header=header,
        comments="",
    )

    position_rms = np.sqrt(np.mean(position_error**2, axis=0))
    velocity_rms = np.sqrt(np.mean(velocity_error**2, axis=0))
    summary = {
        "maturity": "verified MVP / production-oriented baseline",
        "initialization": "truth-assisted for dataset evaluation only",
        "imu_profile": profile_name,
        "duration_s": float(times[-1] - times[0]),
        "navigation_epochs": int(times.size),
        "accepted_gnss_updates": eskf.accepted_gnss_updates,
        "rejected_gnss_updates": eskf.rejected_gnss_updates,
        "skipped_unsynchronized_gnss": skipped_gnss,
        "position_rms_ned_m": position_rms.tolist(),
        "position_rms_3d_m": float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))),
        "velocity_rms_ned_mps": velocity_rms.tolist(),
        "attitude_error_rms_deg": float(np.sqrt(np.mean(attitude_error_deg**2))),
        "limitations": [
            "No online coarse/fine alignment; replay starts from truth state.",
            "No scale-factor, non-orthogonality, temperature, or vibration-rectification model.",
            "GNSS updates use nearest propagated epoch within the configured tolerance; no delayed-state rewind.",
            "Noise profiles are starting points and must be replaced by calibrated device parameters.",
            "Python implementation is for offline verification, not certified hard real-time deployment.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    elapsed = times - times[0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, label in enumerate(("North", "East", "Down")):
        axes[axis].plot(elapsed, position_error[:, axis])
        axes[axis].set_ylabel(f"{label} [m]")
        axes[axis].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Elapsed time [s]")
    fig.suptitle("ESKF Position Error")
    fig.tight_layout()
    fig.savefig(output_dir / "position_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, label in enumerate(("VN", "VE", "VD")):
        axes[axis].plot(elapsed, velocity_error[:, axis])
        axes[axis].set_ylabel(f"{label} [m/s]")
        axes[axis].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Elapsed time [s]")
    fig.suptitle("ESKF Velocity Error")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_error.png", dpi=180)
    plt.close(fig)


def run_replay(args: argparse.Namespace) -> dict[str, object]:
    data = load_dataset1(args.dataset_dir)
    truth = data.truth
    start_time = float(truth.time_s[0])
    end_time = float(truth.time_s[-1])
    if args.duration_s > 0.0:
        end_time = min(end_time, start_time + args.duration_s)

    noise = (
        IMUNoiseModel.mems_default()
        if args.imu_profile == "mems"
        else IMUNoiseModel.navigation_grade_default()
    )
    config = ESKFConfig(
        imu_noise=noise,
        gnss_lever_arm_b_m=np.asarray(args.lever_arm_b_m, dtype=float),
    )
    state = ESKFState(
        time_s=start_time,
        latitude_rad=np.deg2rad(truth.latitude_deg[0]),
        longitude_rad=np.deg2rad(truth.longitude_deg[0]),
        height_m=float(truth.height_m[0]),
        velocity_ned_mps=truth.velocity_ned_mps[0],
        q_bn=euler_zyx_to_quat(*truth.attitude_rpy_deg[0], degrees=True),
        accel_bias_mps2=np.zeros(3),
        gyro_bias_rps=np.zeros(3),
        covariance=default_initial_covariance(
            position_std_m=0.1,
            velocity_std_mps=0.05,
            attitude_std_deg=0.2,
            accel_bias_std_mps2=0.02,
            gyro_bias_std_deg_s=0.005,
        ),
    )
    eskf = LooselyCoupledESKF(state, config)

    history: dict[str, list] = {
        "time_s": [], "latitude_rad": [], "longitude_rad": [], "height_m": [],
        "velocity_ned_mps": [], "q_bn": [], "accel_bias_mps2": [],
        "gyro_bias_rps": [], "covariance_std": [],
    }

    def record_state() -> None:
        history["time_s"].append(eskf.state.time_s)
        history["latitude_rad"].append(eskf.state.latitude_rad)
        history["longitude_rad"].append(eskf.state.longitude_rad)
        history["height_m"].append(eskf.state.height_m)
        history["velocity_ned_mps"].append(eskf.state.velocity_ned_mps.copy())
        history["q_bn"].append(eskf.state.q_bn.copy())
        history["accel_bias_mps2"].append(eskf.state.accel_bias_mps2.copy())
        history["gyro_bias_rps"].append(eskf.state.gyro_bias_rps.copy())
        history["covariance_std"].append(np.sqrt(np.diag(eskf.state.covariance)))

    record_state()
    imu_start = int(np.searchsorted(data.imu.time_s, start_time, side="left"))
    # A row timestamp marks the end of its increment.  Skip any row whose
    # integration interval starts before the initialized state epoch, even if
    # floating-point timestamp text differs by only a few nanoseconds.
    while imu_start < data.imu.time_s.size:
        if imu_start == 0:
            imu_start += 1
            continue
        candidate_dt = (
            data.imu.time_s[imu_start] - data.imu.time_s[imu_start - 1]
        )
        candidate_start = data.imu.time_s[imu_start] - candidate_dt
        if candidate_start >= start_time - 1e-6:
            break
        imu_start += 1
    gnss_index = int(np.searchsorted(data.rtk.time_s, start_time, side="right"))
    skipped_gnss = 0

    imu_index = imu_start
    while imu_index + 1 < data.imu.time_s.size:
        time1 = float(data.imu.time_s[imu_index])
        time2 = float(data.imu.time_s[imu_index + 1])
        if time2 > end_time:
            break
        dt1 = time1 - float(data.imu.time_s[imu_index - 1])
        dt2 = time2 - time1
        imu1 = TimedIMUIncrement(
            time1,
            data.imu.delta_angle_rad[imu_index],
            data.imu.delta_velocity_mps[imu_index],
            dt1,
        )
        imu2 = TimedIMUIncrement(
            time2,
            data.imu.delta_angle_rad[imu_index + 1],
            data.imu.delta_velocity_mps[imu_index + 1],
            dt2,
        )
        eskf.predict_two_sample(imu1, imu2)

        while (
            gnss_index < data.rtk.time_s.size
            and data.rtk.time_s[gnss_index] <= eskf.state.time_s
        ):
            measurement = GNSSPositionMeasurement(
                time_s=float(data.rtk.time_s[gnss_index]),
                latitude_rad=np.deg2rad(data.rtk.latitude_deg[gnss_index]),
                longitude_rad=np.deg2rad(data.rtk.longitude_deg[gnss_index]),
                height_m=float(data.rtk.height_m[gnss_index]),
                std_ned_m=data.rtk.std_ned_m[gnss_index],
            )
            if abs(eskf.state.time_s - measurement.time_s) <= config.max_gnss_time_error_s:
                eskf.update_gnss_position(measurement)
            else:
                skipped_gnss += 1
            gnss_index += 1

        record_state()
        imu_index += 2

    _write_outputs(
        args.output_dir,
        history,
        data,
        eskf,
        skipped_gnss,
        args.imu_profile,
    )
    return {
        "eskf": eskf,
        "history": history,
        "skipped_gnss": skipped_gnss,
    }


def main() -> None:
    args = parse_args()
    result = run_replay(args)
    eskf = result["eskf"]
    print(f"Wrote ESKF replay artifacts to {args.output_dir}")
    print(f"Accepted GNSS updates: {eskf.accepted_gnss_updates}")
    print(f"Rejected GNSS updates: {eskf.rejected_gnss_updates}")
    print(f"Skipped unsynchronized GNSS: {result['skipped_gnss']}")


if __name__ == "__main__":
    main()
