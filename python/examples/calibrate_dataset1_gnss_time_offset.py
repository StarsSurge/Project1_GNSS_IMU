"""Estimate a constant Dataset1 GNSS timestamp offset by delayed ESKF replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

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
    TimedIMUIncrement,
    calibrate_constant_gnss_time_offset,
    default_initial_covariance,
    euler_zyx_to_quat,
    load_dataset1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a constant GNSS time offset using robust ESKF innovation scores."
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data" / "dataset1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_time_offset",
    )
    parser.add_argument("--start-offset-s", type=float, default=70.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--candidate-min-s", type=float, default=-0.05)
    parser.add_argument("--candidate-max-s", type=float, default=0.05)
    parser.add_argument("--candidate-step-s", type=float, default=0.005)
    parser.add_argument("--fixed-lag-s", type=float, default=1.0)
    parser.add_argument(
        "--lever-arm-b-m",
        type=float,
        nargs=3,
        default=(0.14722696, -0.29821683, -0.18079014),
        metavar=("FORWARD", "RIGHT", "DOWN"),
    )
    return parser.parse_args()


def run_calibration(args: argparse.Namespace) -> dict[str, object]:
    if args.duration_s <= 0.0 or args.candidate_step_s <= 0.0:
        raise ValueError("duration_s and candidate_step_s must be positive")
    candidates = np.arange(
        args.candidate_min_s,
        args.candidate_max_s + 0.5 * args.candidate_step_s,
        args.candidate_step_s,
    )
    candidates[np.isclose(candidates, 0.0, atol=1e-12)] = 0.0
    if candidates.size < 2:
        raise ValueError("candidate range must contain at least two offsets")

    data = load_dataset1(args.dataset_dir)
    requested_start = float(data.truth.time_s[0] + args.start_offset_s)
    start_imu_index = int(np.searchsorted(data.imu.time_s, requested_start, side="left"))
    if start_imu_index <= 0 or start_imu_index + 1 >= data.imu.time_s.size:
        raise ValueError("calibration start is outside the usable IMU range")
    start_time_s = float(data.imu.time_s[start_imu_index])
    end_time_s = min(start_time_s + args.duration_s, float(data.imu.time_s[-1]))

    truth = data.truth
    attitude_rpy_deg = np.array(
        [
            np.interp(start_time_s, truth.time_s, truth.attitude_rpy_deg[:, axis])
            for axis in range(3)
        ]
    )
    initial_state = ESKFState(
        time_s=start_time_s,
        latitude_rad=np.deg2rad(np.interp(start_time_s, truth.time_s, truth.latitude_deg)),
        longitude_rad=np.deg2rad(np.interp(start_time_s, truth.time_s, truth.longitude_deg)),
        height_m=float(np.interp(start_time_s, truth.time_s, truth.height_m)),
        velocity_ned_mps=np.array(
            [
                np.interp(start_time_s, truth.time_s, truth.velocity_ned_mps[:, axis])
                for axis in range(3)
            ]
        ),
        q_bn=euler_zyx_to_quat(*attitude_rpy_deg, degrees=True),
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

    imu_samples = []
    for index in range(start_imu_index + 1, data.imu.time_s.size):
        time_s = float(data.imu.time_s[index])
        if time_s > end_time_s:
            break
        imu_samples.append(
            TimedIMUIncrement(
                time_s,
                data.imu.delta_angle_rad[index],
                data.imu.delta_velocity_mps[index],
                time_s - float(data.imu.time_s[index - 1]),
            )
        )

    margin_s = float(np.max(np.abs(candidates))) + 0.02
    gnss_indices = np.flatnonzero(
        (data.rtk.time_s >= start_time_s + margin_s)
        & (data.rtk.time_s <= end_time_s - margin_s)
    )
    gnss_measurements = [
        GNSSPositionMeasurement(
            float(data.rtk.time_s[index]),
            np.deg2rad(data.rtk.latitude_deg[index]),
            np.deg2rad(data.rtk.longitude_deg[index]),
            float(data.rtk.height_m[index]),
            data.rtk.std_ned_m[index],
        )
        for index in gnss_indices
    ]
    config = ESKFConfig(
        imu_noise=IMUNoiseModel.navigation_grade_default(),
        gnss_lever_arm_b_m=np.asarray(args.lever_arm_b_m, dtype=float),
    )
    result = calibrate_constant_gnss_time_offset(
        initial_state,
        config,
        imu_samples,
        gnss_measurements,
        candidates,
        lag_s=args.fixed_lag_s,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_table = np.array(
        [
            [
                score.offset_s,
                score.robust_mean_nis,
                score.median_nis,
                score.measurement_count,
                score.accepted_count,
                score.rejected_count,
            ]
            for score in result.scores
        ]
    )
    np.savetxt(
        args.output_dir / "time_offset_scores.csv",
        score_table,
        delimiter=",",
        header=(
            "offset_s,robust_mean_nis,median_nis,measurement_count,"
            "accepted_count,rejected_count"
        ),
        comments="",
    )
    ordered_scores = sorted(result.scores, key=lambda item: item.robust_mean_nis)
    score_margin = (
        ordered_scores[1].robust_mean_nis - ordered_scores[0].robust_mean_nis
    )
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(score_table[:, 0] * 1e3, score_table[:, 1], marker="o", label="Clipped mean NIS")
    axis.plot(score_table[:, 0] * 1e3, score_table[:, 2], marker="s", label="Median NIS")
    axis.axvline(result.best_offset_s * 1e3, color="tab:red", linestyle="--", label="Selected offset")
    axis.set_xlabel("GNSS time offset [ms]")
    axis.set_ylabel("Innovation score [-]")
    axis.set_title("GNSS Time-Offset Candidate Scan")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "time_offset_scores.png", dpi=180)
    plt.close(fig)
    summary = {
        "maturity": "offline calibration MVP",
        "offset_convention": "effective_time = reported_time + offset",
        "best_offset_s": result.best_offset_s,
        "best_robust_mean_nis": ordered_scores[0].robust_mean_nis,
        "runner_up_score_margin": score_margin,
        "peak_speed_mps": result.peak_speed_mps,
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "imu_sample_count": len(imu_samples),
        "gnss_measurement_count": len(gnss_measurements),
        "candidate_count": len(result.scores),
        "limitations": [
            "Truth-assisted initial state is used to isolate timing during this evaluation workflow.",
            "Only one constant offset is scanned; clock drift and latency jitter are not modeled.",
            "The selected offset requires cross-window and independent-dataset validation.",
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
    print(f"Best GNSS time offset: {summary['best_offset_s']:+.6f} s")
    print(f"Wrote calibration artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
