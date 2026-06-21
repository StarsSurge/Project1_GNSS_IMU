"""Jointly calibrate Dataset1 GNSS lever arm and constant time offset."""

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
    antenna_velocity_ned,
    calibrate_lever_arm_and_time_offset,
    calibrate_time_offset_with_fixed_lever,
    fit_body_lever_arm_from_residuals,
    earth_rate_ned,
    geodetic_to_ned,
    interpolate_columns,
    load_dataset1,
    rpy_deg_to_body_to_ned,
    residual_correlation_diagnostics,
    transport_rate_ned,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit body-frame GNSS lever arm and constant timestamp offset from "
            "RTK-minus-truth position residuals."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data" / "dataset1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_lever_time_calibration",
    )
    parser.add_argument("--start-offset-s", type=float, default=0.0)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Calibration duration; <=0 uses all overlapping data.",
    )
    parser.add_argument("--huber-threshold", type=float, default=2.5)
    parser.add_argument(
        "--lever-prior-b-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("FORWARD", "RIGHT", "DOWN"),
        help="Independent mechanical lever-arm prior [m].",
    )
    parser.add_argument(
        "--lever-prior-std-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("STD_FORWARD", "STD_RIGHT", "STD_DOWN"),
        help="One-sigma uncertainty of the independent lever prior [m].",
    )
    parser.add_argument(
        "--fixed-lever-b-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("FORWARD", "RIGHT", "DOWN"),
        help="Fix an independently measured lever arm and estimate only time.",
    )
    parser.add_argument(
        "--omit-rotational-velocity",
        action="store_true",
        help="A/B baseline using IMU-point velocity instead of antenna velocity.",
    )
    return parser.parse_args()


def run_calibration(args: argparse.Namespace) -> dict[str, object]:
    fixed_lever_b_m = getattr(args, "fixed_lever_b_m", None)
    lever_prior_b_m = getattr(args, "lever_prior_b_m", None)
    lever_prior_std_m = getattr(args, "lever_prior_std_m", None)
    omit_rotational_velocity = bool(
        getattr(args, "omit_rotational_velocity", False)
    )
    if fixed_lever_b_m is not None and (
        lever_prior_b_m is not None or lever_prior_std_m is not None
    ):
        raise ValueError("choose either a fixed lever or a finite lever prior")
    data = load_dataset1(args.dataset_dir)
    truth = data.truth
    start_time_s = float(truth.time_s[0] + args.start_offset_s)
    end_time_s = float(truth.time_s[-1])
    if args.duration_s > 0.0:
        end_time_s = min(end_time_s, start_time_s + args.duration_s)
    mask = (
        (data.rtk.time_s >= max(start_time_s, float(truth.time_s[0])))
        & (data.rtk.time_s <= min(end_time_s, float(truth.time_s[-1])))
    )
    if np.count_nonzero(mask) < 4:
        raise ValueError("joint calibration interval has fewer than four GNSS epochs")

    reference_llh = (
        float(truth.latitude_deg[0]),
        float(truth.longitude_deg[0]),
        float(truth.height_m[0]),
    )
    truth_ned = geodetic_to_ned(
        truth.latitude_deg,
        truth.longitude_deg,
        truth.height_m,
        reference_llh,
    )
    rtk_ned = geodetic_to_ned(
        data.rtk.latitude_deg[mask],
        data.rtk.longitude_deg[mask],
        data.rtk.height_m[mask],
        reference_llh,
    )
    measurement_times = data.rtk.time_s[mask]
    truth_position = interpolate_columns(truth.time_s, truth_ned, measurement_times)
    truth_velocity = interpolate_columns(
        truth.time_s,
        truth.velocity_ned_mps,
        measurement_times,
    )
    truth_attitude = interpolate_columns(
        truth.time_s,
        truth.attitude_rpy_deg,
        measurement_times,
    )
    body_to_ned = rpy_deg_to_body_to_ned(
        truth_attitude[:, 0],
        truth_attitude[:, 1],
        truth_attitude[:, 2],
    )
    imu_dt = np.diff(data.imu.time_s)
    if np.any(imu_dt <= 0.0):
        raise ValueError("IMU timestamps must be strictly increasing")
    imu_rate_times = 0.5 * (data.imu.time_s[1:] + data.imu.time_s[:-1])
    gyro_rate_ib_b = data.imu.delta_angle_rad[1:] / imu_dt[:, None]
    gyro_rate_at_gnss = interpolate_columns(
        imu_rate_times,
        gyro_rate_ib_b,
        measurement_times,
    )
    latitude_at_gnss_rad = np.deg2rad(
        np.interp(measurement_times, truth.time_s, truth.latitude_deg)
    )
    height_at_gnss_m = np.interp(
        measurement_times,
        truth.time_s,
        truth.height_m,
    )
    navigation_rate_ned = np.vstack(
        [
            earth_rate_ned(latitude_at_gnss_rad[index])
            + transport_rate_ned(
                latitude_at_gnss_rad[index],
                height_at_gnss_m[index],
                truth_velocity[index],
            )
            for index in range(measurement_times.size)
        ]
    )
    navigation_rate_b = np.einsum(
        "nji,nj->ni",
        body_to_ned,
        navigation_rate_ned,
    )
    angular_rate_nb_b = gyro_rate_at_gnss - navigation_rate_b
    angular_rate_for_model = (
        None if omit_rotational_velocity else angular_rate_nb_b
    )
    raw_residual = rtk_ned - truth_position
    if fixed_lever_b_m is None:
        result = calibrate_lever_arm_and_time_offset(
            body_to_ned,
            truth_velocity,
            raw_residual,
            data.rtk.std_ned_m[mask],
            angular_rate_b_rps=angular_rate_for_model,
            lever_prior_b_m=(
                None
                if lever_prior_b_m is None
                else np.asarray(lever_prior_b_m, dtype=float)
            ),
            lever_prior_std_m=(
                None
                if lever_prior_std_m is None
                else np.asarray(lever_prior_std_m, dtype=float)
            ),
            huber_threshold=float(args.huber_threshold),
        )
        calibration_mode = (
            "joint-unconstrained"
            if lever_prior_b_m is None
            else "joint-with-lever-prior"
        )
        time_offset_std = float(result.parameter_std[3])
        lever_std = result.parameter_std[:3].tolist()
        time_lever_correlation = result.correlation[:3, 3].tolist()
        condition_number = result.condition_number
        parameter_covariance = result.covariance.tolist()
        parameter_correlation = result.correlation.tolist()
    else:
        result = calibrate_time_offset_with_fixed_lever(
            body_to_ned,
            truth_velocity,
            raw_residual,
            np.asarray(fixed_lever_b_m, dtype=float),
            data.rtk.std_ned_m[mask],
            angular_rate_b_rps=angular_rate_for_model,
            huber_threshold=float(args.huber_threshold),
        )
        calibration_mode = "fixed-independent-lever"
        time_offset_std = result.time_offset_std_s
        lever_std = None
        time_lever_correlation = None
        condition_number = None
        parameter_covariance = None
        parameter_correlation = None
    lever_only, _, lever_only_corrected = fit_body_lever_arm_from_residuals(
        body_to_ned,
        raw_residual,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_table = np.column_stack(
        [
            measurement_times,
            raw_residual,
            result.predicted_residual_ned_m,
            result.corrected_residual_ned_m,
            result.robust_epoch_weights,
        ]
    )
    np.savetxt(
        args.output_dir / "lever_time_residuals.csv",
        output_table,
        delimiter=",",
        header=(
            "time_s,raw_n_m,raw_e_m,raw_d_m,predicted_n_m,predicted_e_m,"
            "predicted_d_m,corrected_n_m,corrected_e_m,corrected_d_m,robust_weight"
        ),
        comments="",
    )

    elapsed = measurement_times - measurement_times[0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, label in enumerate(("North", "East", "Down")):
        axes[axis].plot(elapsed, raw_residual[:, axis], alpha=0.35, label="raw")
        axes[axis].plot(
            elapsed,
            result.corrected_residual_ned_m[:, axis],
            label="after lever + time",
        )
        axes[axis].set_ylabel(f"{label} [m]")
        axes[axis].grid(True, alpha=0.3)
        axes[axis].legend()
    axes[-1].set_xlabel("Elapsed time [s]")
    fig.suptitle("RTK Minus Truth Residual: Joint Lever/Time Calibration")
    fig.tight_layout()
    fig.savefig(args.output_dir / "lever_time_residuals.png", dpi=180)
    plt.close(fig)

    if calibration_mode != "fixed-independent-lever":
        labels = ("lever_forward", "lever_right", "lever_down", "time_offset")
        fig, axis = plt.subplots(figsize=(6.5, 5.5))
        image = axis.imshow(result.correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(np.arange(4), labels, rotation=30, ha="right")
        axis.set_yticks(np.arange(4), labels)
        for row in range(4):
            for column in range(4):
                axis.text(
                    column,
                    row,
                    f"{result.correlation[row, column]:.2f}",
                    ha="center",
                    va="center",
                )
        fig.colorbar(image, ax=axis, label="Correlation coefficient")
        axis.set_title("Joint Calibration Parameter Correlation")
        fig.tight_layout()
        fig.savefig(args.output_dir / "parameter_correlation.png", dpi=180)
        plt.close(fig)

    corrected_rms_3d = float(
        np.sqrt(np.mean(np.sum(result.corrected_residual_ned_m**2, axis=1)))
    )
    antenna_velocity = antenna_velocity_ned(
        body_to_ned,
        truth_velocity,
        angular_rate_nb_b,
        result.lever_arm_b_m,
    )
    rotational_velocity = antenna_velocity - truth_velocity
    correlation_diagnostics = residual_correlation_diagnostics(
        result.corrected_residual_ned_m,
        max_lag=min(200, measurement_times.size - 1),
    )
    minimum_effective_samples = float(
        np.min(correlation_diagnostics.effective_sample_size)
    )
    correlation_inflation = float(
        np.sqrt(measurement_times.size / minimum_effective_samples)
    )
    correlation_inflated_time_std = time_offset_std * correlation_inflation
    np.savetxt(
        args.output_dir / "residual_autocorrelation.csv",
        np.column_stack(
            [
                np.arange(correlation_diagnostics.max_lag + 1),
                correlation_diagnostics.autocorrelation,
            ]
        ),
        delimiter=",",
        header="lag,autocorrelation_n,autocorrelation_e,autocorrelation_d",
        comments="",
    )
    fig, axis = plt.subplots(figsize=(9, 4.5))
    for component, label in enumerate(("North", "East", "Down")):
        axis.plot(
            np.arange(correlation_diagnostics.max_lag + 1),
            correlation_diagnostics.autocorrelation[:, component],
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("Lag [GNSS epochs]")
    axis.set_ylabel("Autocorrelation [-]")
    axis.set_title("Corrected Position Residual Autocorrelation")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "residual_autocorrelation.png", dpi=180)
    plt.close(fig)
    lever_only_rms_3d = float(
        np.sqrt(np.mean(np.sum(lever_only_corrected**2, axis=1)))
    )
    summary = {
        "maturity": "truth-referenced offline joint calibration MVP",
        "calibration_mode": calibration_mode,
        "model": (
            "residual_ned = C_bn * lever_b + "
            "[velocity_ned + C_bn(omega_nb_b cross lever_b)] * time_offset"
        ),
        "uses_rotational_velocity": not omit_rotational_velocity,
        "offset_convention": "effective_time = reported_time + offset",
        "start_time_s": float(measurement_times[0]),
        "end_time_s": float(measurement_times[-1]),
        "epoch_count": int(measurement_times.size),
        "lever_prior_b_m": lever_prior_b_m,
        "lever_prior_std_m": lever_prior_std_m,
        "fixed_lever_b_m": fixed_lever_b_m,
        "lever_arm_b_m": result.lever_arm_b_m.tolist(),
        "lever_arm_std_m": lever_std,
        "time_offset_s": result.time_offset_s,
        "time_offset_std_s": time_offset_std,
        "time_offset_ci95_s": [
            result.time_offset_s - 1.959963984540054 * time_offset_std,
            result.time_offset_s + 1.959963984540054 * time_offset_std,
        ],
        "correlation_inflated_time_std_s": correlation_inflated_time_std,
        "correlation_inflated_time_ci95_s": [
            result.time_offset_s - 1.959963984540054 * correlation_inflated_time_std,
            result.time_offset_s + 1.959963984540054 * correlation_inflated_time_std,
        ],
        "residual_lag1_autocorrelation_ned": (
            correlation_diagnostics.lag1_autocorrelation.tolist()
        ),
        "integrated_autocorrelation_time_ned": (
            correlation_diagnostics.integrated_autocorrelation_time.tolist()
        ),
        "effective_sample_size_ned": (
            correlation_diagnostics.effective_sample_size.tolist()
        ),
        "correlation_std_inflation_factor": correlation_inflation,
        "time_lever_correlation": time_lever_correlation,
        "condition_number": condition_number,
        "iteration_count": result.iteration_count,
        "downweighted_epoch_count": result.downweighted_epoch_count,
        "angular_rate_rms_rps": float(
            np.sqrt(np.mean(np.sum(angular_rate_nb_b**2, axis=1)))
        ),
        "rotational_lever_velocity_rms_mps": float(
            np.sqrt(np.mean(np.sum(rotational_velocity**2, axis=1)))
        ),
        "rotational_lever_velocity_max_mps": float(
            np.max(np.linalg.norm(rotational_velocity, axis=1))
        ),
        "raw_rms_ned_m": result.raw_rms_ned_m.tolist(),
        "corrected_rms_ned_m": result.corrected_rms_ned_m.tolist(),
        "corrected_rms_3d_m": corrected_rms_3d,
        "lever_only_solution_b_m": lever_only.tolist(),
        "lever_only_corrected_rms_3d_m": lever_only_rms_3d,
        "parameter_covariance": parameter_covariance,
        "parameter_correlation": parameter_correlation,
        "limitations": [
            "Truth position, velocity, and attitude are used; this is an offline reference calibration.",
            "The time term is first-order; angular rate comes from raw IMU increments without an independently calibrated gyro-bias correction.",
            "Reported RTK standard deviations do not include truth uncertainty or temporal correlation.",
            "A full deployment calibration requires independent geometry measurement and repeated datasets.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_calibration(args)
    print(f"Lever arm [FRD m]: {summary['lever_arm_b_m']}")
    print(f"Time offset: {summary['time_offset_s'] * 1e3:+.3f} ms")
    print(f"Wrote joint calibration artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
