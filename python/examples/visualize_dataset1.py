"""Visualize dataset1 RTK, IMU increment, and truth navigation data."""

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

from gnss_imu.dataset_visualization import (  # noqa: E402
    Dataset1,
    decimation_indices,
    fit_body_lever_arm_from_residuals,
    geodetic_to_ned,
    increments_to_rates,
    interpolate_columns,
    load_dataset1,
    relative_time_seconds,
    rpy_deg_to_body_to_ned,
    sampling_summary,
)

AXIS_LABELS = ("x", "y", "z")
NED_LABELS = ("north", "east", "down")
ATTITUDE_LABELS = ("roll", "pitch", "yaw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate first-pass plots for data/dataset1 RTK, IMU increments, "
            "and truth navigation states."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "dataset1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_visualization",
    )
    parser.add_argument(
        "--max-imu-points",
        type=int,
        default=8000,
        help="Maximum points per IMU curve after deterministic decimation.",
    )
    parser.add_argument(
        "--max-truth-points",
        type=int,
        default=12000,
        help="Maximum points per truth curve after deterministic decimation.",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trajectory(
    data: Dataset1,
    truth_ned_m: np.ndarray,
    rtk_ned_m: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.plot(
        truth_ned_m[:, 1],
        truth_ned_m[:, 0],
        color="tab:blue",
        linewidth=1.2,
        label="truth",
    )
    ax.scatter(
        rtk_ned_m[:, 1],
        rtk_ned_m[:, 0],
        s=9,
        color="tab:orange",
        alpha=0.75,
        label="RTK",
    )
    ax.scatter(
        truth_ned_m[0, 1],
        truth_ned_m[0, 0],
        marker="o",
        s=35,
        color="tab:green",
        label="truth start",
    )
    ax.set_title("Dataset1 Local Trajectory")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_dir / "trajectory_ned.png")


def plot_position_components(
    data: Dataset1,
    truth_ned_m: np.ndarray,
    rtk_ned_m: np.ndarray,
    output_dir: Path,
) -> None:
    origin = data.truth.time_s[0]
    truth_t = relative_time_seconds(data.truth.time_s, origin)
    rtk_t = relative_time_seconds(data.rtk.time_s, origin)
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 9.0), sharex=True)
    for axis, label in enumerate(NED_LABELS):
        axes[axis].plot(truth_t, truth_ned_m[:, axis], label="truth")
        axes[axis].scatter(rtk_t, rtk_ned_m[:, axis], s=8, label="RTK")
        axes[axis].set_ylabel(f"{label} [m]")
        axes[axis].grid(True, alpha=0.3)
    axes[3].plot(truth_t, data.truth.height_m, label="truth")
    axes[3].scatter(rtk_t, data.rtk.height_m, s=8, label="RTK")
    axes[3].set_ylabel("height [m]")
    axes[3].set_xlabel("Time from truth start [s]")
    axes[3].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle("RTK and Truth Position Components")
    save_figure(fig, output_dir / "position_components.png")


def plot_rtk_std(data: Dataset1, output_dir: Path) -> None:
    t = relative_time_seconds(data.rtk.time_s, data.rtk.time_s[0])
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for axis, label in enumerate(NED_LABELS):
        ax.plot(t, data.rtk.std_ned_m[:, axis], label=f"{label} std")
    ax.set_title("RTK Reported Position Standard Deviation")
    ax.set_xlabel("RTK elapsed time [s]")
    ax.set_ylabel("1-sigma [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_dir / "rtk_position_std.png")


def plot_rtk_truth_residuals(
    data: Dataset1,
    truth_ned_m: np.ndarray,
    rtk_ned_m: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    mask = (
        (data.rtk.time_s >= data.truth.time_s[0])
        & (data.rtk.time_s <= data.truth.time_s[-1])
    )
    if np.count_nonzero(mask) < 2:
        return {"overlap_count": 0.0}

    rtk_t = data.rtk.time_s[mask]
    truth_at_rtk = interpolate_columns(data.truth.time_s, truth_ned_m, rtk_t)
    residual_ned = rtk_ned_m[mask] - truth_at_rtk
    elapsed = relative_time_seconds(rtk_t, rtk_t[0])
    attitude_at_rtk = interpolate_columns(
        data.truth.time_s, data.truth.attitude_rpy_deg, rtk_t
    )
    body_to_ned = rpy_deg_to_body_to_ned(
        attitude_at_rtk[:, 0],
        attitude_at_rtk[:, 1],
        attitude_at_rtk[:, 2],
    )
    lever_arm_b_m, _, corrected_residual_ned = (
        fit_body_lever_arm_from_residuals(body_to_ned, residual_ned)
    )

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True)
    for axis, label in enumerate(NED_LABELS):
        axes[axis].plot(elapsed, residual_ned[:, axis])
        axes[axis].axhline(0.0, color="black", linewidth=0.7)
        axes[axis].set_ylabel(f"{label} error [m]")
        axes[axis].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Overlap elapsed time [s]")
    fig.suptitle("RTK minus Interpolated Truth Position")
    save_figure(fig, output_dir / "rtk_minus_truth_ned.png")

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True)
    for axis, label in enumerate(NED_LABELS):
        axes[axis].plot(
            elapsed,
            residual_ned[:, axis],
            alpha=0.35,
            label="raw",
        )
        axes[axis].plot(
            elapsed,
            corrected_residual_ned[:, axis],
            label="after fitted lever arm",
        )
        axes[axis].axhline(0.0, color="black", linewidth=0.7)
        axes[axis].set_ylabel(f"{label} error [m]")
        axes[axis].grid(True, alpha=0.3)
        axes[axis].legend(loc="best")
    axes[-1].set_xlabel("Overlap elapsed time [s]")
    fig.suptitle(
        "RTK minus Truth Before/After Fitted Body-Frame Lever Arm"
    )
    save_figure(fig, output_dir / "rtk_minus_truth_lever_arm_check.png")

    rms = np.sqrt(np.mean(residual_ned**2, axis=0))
    corrected_rms = np.sqrt(np.mean(corrected_residual_ned**2, axis=0))
    return {
        "overlap_count": float(rtk_t.size),
        "rms_north_m": float(rms[0]),
        "rms_east_m": float(rms[1]),
        "rms_down_m": float(rms[2]),
        "max_abs_3d_m": float(np.max(np.linalg.norm(residual_ned, axis=1))),
        "fitted_body_lever_arm_forward_right_down_m": lever_arm_b_m.tolist(),
        "corrected_rms_north_m": float(corrected_rms[0]),
        "corrected_rms_east_m": float(corrected_rms[1]),
        "corrected_rms_down_m": float(corrected_rms[2]),
        "corrected_rms_3d_m": float(
            np.sqrt(np.mean(np.sum(corrected_residual_ned**2, axis=1)))
        ),
        "corrected_max_abs_3d_m": float(
            np.max(np.linalg.norm(corrected_residual_ned, axis=1))
        ),
        "diagnostic_note": (
            "Large raw RTK-truth residual is mostly explained by a fixed "
            "body-frame lever arm, consistent with RTK antenna position "
            "versus truth IMU position."
        ),
    }


def plot_truth_velocity_attitude(
    data: Dataset1,
    output_dir: Path,
    max_points: int,
) -> None:
    indices = decimation_indices(data.truth.time_s.size, max_points)
    t = relative_time_seconds(data.truth.time_s[indices], data.truth.time_s[0])
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    for axis, label in enumerate(NED_LABELS):
        axes[0].plot(t, data.truth.velocity_ned_mps[indices, axis], label=label)
    for axis, label in enumerate(ATTITUDE_LABELS):
        axes[1].plot(t, data.truth.attitude_rpy_deg[indices, axis], label=label)
    axes[0].set_title("Truth Velocity and Attitude")
    axes[0].set_ylabel("Velocity [m/s]")
    axes[1].set_ylabel("Attitude [deg]")
    axes[1].set_xlabel("Truth elapsed time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    save_figure(fig, output_dir / "truth_velocity_attitude.png")


def plot_imu(
    data: Dataset1,
    output_dir: Path,
    max_points: int,
) -> None:
    gyro_rate_time_s, gyro_rate_rps = increments_to_rates(
        data.imu.time_s, data.imu.delta_angle_rad
    )
    accel_time_s, specific_force_mps2 = increments_to_rates(
        data.imu.time_s, data.imu.delta_velocity_mps
    )
    inc_indices = decimation_indices(data.imu.time_s.size, max_points)
    rate_indices = decimation_indices(gyro_rate_time_s.size, max_points)

    inc_t = relative_time_seconds(
        data.imu.time_s[inc_indices], data.imu.time_s[0]
    )
    rate_t = relative_time_seconds(gyro_rate_time_s[rate_indices], data.imu.time_s[0])

    fig, axes = plt.subplots(4, 1, figsize=(10.0, 10.0), sharex=True)
    for axis, label in enumerate(AXIS_LABELS):
        axes[0].plot(
            inc_t,
            data.imu.delta_angle_rad[inc_indices, axis],
            label=f"dtheta {label}",
        )
        axes[1].plot(
            inc_t,
            data.imu.delta_velocity_mps[inc_indices, axis],
            label=f"dvel {label}",
        )
        axes[2].plot(
            rate_t,
            gyro_rate_rps[rate_indices, axis],
            label=f"gyro {label}",
        )
        axes[3].plot(
            rate_t,
            specific_force_mps2[rate_indices, axis],
            label=f"accel {label}",
        )

    axes[0].set_title("IMU Increments and Derived Rates")
    axes[0].set_ylabel("Delta angle [rad]")
    axes[1].set_ylabel("Delta velocity [m/s]")
    axes[2].set_ylabel("Angular rate [rad/s]")
    axes[3].set_ylabel("Specific force [m/s^2]")
    axes[3].set_xlabel("IMU elapsed time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncols=3)
    save_figure(fig, output_dir / "imu_increments_and_rates.png")


def write_summary(
    data: Dataset1,
    residual_summary: dict[str, float],
    output_dir: Path,
) -> None:
    gyro_rate_time_s, gyro_rate_rps = increments_to_rates(
        data.imu.time_s, data.imu.delta_angle_rad
    )
    accel_time_s, specific_force_mps2 = increments_to_rates(
        data.imu.time_s, data.imu.delta_velocity_mps
    )
    summary = {
        "rtk": sampling_summary(data.rtk.time_s),
        "imu": sampling_summary(data.imu.time_s),
        "truth": sampling_summary(data.truth.time_s),
        "rtk_minus_truth_ned": residual_summary,
        "imu_derived_rate_units": {
            "gyro": "rad/s from delta_angle_rad / dt",
            "accelerometer": "m/s^2 from delta_velocity_mps / dt",
        },
        "imu_derived_rate_mean": {
            "gyro_rps_xyz": np.mean(gyro_rate_rps, axis=0).tolist(),
            "specific_force_mps2_xyz": np.mean(
                specific_force_mps2, axis=0
            ).tolist(),
            "gyro_rate_time_start_s": float(gyro_rate_time_s[0]),
            "accel_rate_time_start_s": float(accel_time_s[0]),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    data = load_dataset1(args.dataset_dir)
    reference_llh = (
        float(data.truth.latitude_deg[0]),
        float(data.truth.longitude_deg[0]),
        float(data.truth.height_m[0]),
    )
    truth_ned_m = geodetic_to_ned(
        data.truth.latitude_deg,
        data.truth.longitude_deg,
        data.truth.height_m,
        reference_llh,
    )
    rtk_ned_m = geodetic_to_ned(
        data.rtk.latitude_deg,
        data.rtk.longitude_deg,
        data.rtk.height_m,
        reference_llh,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectory(data, truth_ned_m, rtk_ned_m, args.output_dir)
    plot_position_components(data, truth_ned_m, rtk_ned_m, args.output_dir)
    plot_rtk_std(data, args.output_dir)
    residual_summary = plot_rtk_truth_residuals(
        data, truth_ned_m, rtk_ned_m, args.output_dir
    )
    plot_truth_velocity_attitude(data, args.output_dir, args.max_truth_points)
    plot_imu(data, args.output_dir, args.max_imu_points)
    write_summary(data, residual_summary, args.output_dir)

    print(f"Wrote dataset1 plots and summary to {args.output_dir}")
    for path in sorted(args.output_dir.glob("*")):
        print(f"- {path}")


if __name__ == "__main__":
    main()
