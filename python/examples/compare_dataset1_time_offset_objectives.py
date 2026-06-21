"""Compare frozen-trajectory and sequential-ESKF time-offset objectives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
EXAMPLES_DIR = PYTHON_DIR / "examples"
for path in (PYTHON_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrate_dataset1_gnss_time_offset import run_calibration as run_eskf_calibration  # noqa: E402
from gnss_imu import (  # noqa: E402
    TimeOffsetCalibrationResult,
    TimeOffsetCandidateScore,
    estimate_time_offset_profile_interval,
    bootstrap_frozen_time_offset,
    frozen_trajectory_nis_matrix,
    geodetic_to_ned,
    load_dataset1,
    interpolate_columns,
    residual_correlation_diagnostics,
    rpy_deg_to_body_to_ned,
    score_frozen_trajectory_time_offsets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed truth trajectory scoring with sequential ESKF NIS scoring."
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data" / "dataset1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_time_objective_comparison",
    )
    parser.add_argument("--start-offset-s", type=float, default=70.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--candidate-min-s", type=float, default=-0.05)
    parser.add_argument("--candidate-max-s", type=float, default=0.05)
    parser.add_argument("--candidate-step-s", type=float, default=0.005)
    parser.add_argument("--fixed-lag-s", type=float, default=1.0)
    parser.add_argument("--ungated-nis-threshold", type=float, default=1e12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--correlation-max-lag", type=int, default=10)
    parser.add_argument(
        "--lever-arm-b-m",
        type=float,
        nargs=3,
        default=(0.14722696, -0.29821683, -0.18079014),
        metavar=("FORWARD", "RIGHT", "DOWN"),
    )
    return parser.parse_args()


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eskf_output = args.output_dir / "eskf_profile"
    eskf_summary = run_eskf_calibration(
        SimpleNamespace(
            dataset_dir=args.dataset_dir,
            output_dir=eskf_output,
            start_offset_s=float(args.start_offset_s),
            duration_s=float(args.duration_s),
            candidate_min_s=float(args.candidate_min_s),
            candidate_max_s=float(args.candidate_max_s),
            candidate_step_s=float(args.candidate_step_s),
            fixed_lag_s=float(args.fixed_lag_s),
            lever_arm_b_m=tuple(args.lever_arm_b_m),
            imu_noise_scale=1.0,
        )
    )
    eskf_table = np.genfromtxt(
        eskf_output / "time_offset_scores.csv",
        delimiter=",",
        names=True,
    )
    candidates = np.atleast_1d(eskf_table["offset_s"])
    ungated_output = args.output_dir / "eskf_profile_ungated"
    ungated_summary = run_eskf_calibration(
        SimpleNamespace(
            dataset_dir=args.dataset_dir,
            output_dir=ungated_output,
            start_offset_s=float(args.start_offset_s),
            duration_s=float(args.duration_s),
            candidate_min_s=float(args.candidate_min_s),
            candidate_max_s=float(args.candidate_max_s),
            candidate_step_s=float(args.candidate_step_s),
            fixed_lag_s=float(args.fixed_lag_s),
            lever_arm_b_m=tuple(args.lever_arm_b_m),
            imu_noise_scale=1.0,
            gnss_nis_threshold=float(args.ungated_nis_threshold),
        )
    )
    ungated_table = np.genfromtxt(
        ungated_output / "time_offset_scores.csv",
        delimiter=",",
        names=True,
    )

    data = load_dataset1(args.dataset_dir)
    truth = data.truth
    reference_llh = (
        float(truth.latitude_deg[0]),
        float(truth.longitude_deg[0]),
        float(truth.height_m[0]),
    )
    truth_imu_ned = geodetic_to_ned(
        truth.latitude_deg,
        truth.longitude_deg,
        truth.height_m,
        reference_llh,
    )
    truth_rotation = rpy_deg_to_body_to_ned(
        truth.attitude_rpy_deg[:, 0],
        truth.attitude_rpy_deg[:, 1],
        truth.attitude_rpy_deg[:, 2],
    )
    lever = np.asarray(args.lever_arm_b_m, dtype=float)
    truth_antenna_ned = truth_imu_ned + np.einsum(
        "nij,j->ni",
        truth_rotation,
        lever,
    )
    margin_s = float(np.max(np.abs(candidates))) + 0.02
    start_time_s = float(eskf_summary["start_time_s"])
    end_time_s = float(eskf_summary["end_time_s"])
    measurement_mask = (
        (data.rtk.time_s >= start_time_s + margin_s)
        & (data.rtk.time_s <= end_time_s - margin_s)
    )
    measurement_times = data.rtk.time_s[measurement_mask]
    measured_ned = geodetic_to_ned(
        data.rtk.latitude_deg[measurement_mask],
        data.rtk.longitude_deg[measurement_mask],
        data.rtk.height_m[measurement_mask],
        reference_llh,
    )
    frozen_scores = score_frozen_trajectory_time_offsets(
        truth.time_s,
        truth_antenna_ned,
        measurement_times,
        measured_ned,
        data.rtk.std_ned_m[measurement_mask],
        candidates,
    )
    frozen_calibration = TimeOffsetCalibrationResult(
        best_offset_s=min(
            frozen_scores,
            key=lambda item: (item.robust_mean_nis, item.median_nis),
        ).offset_s,
        scores=tuple(
            TimeOffsetCandidateScore(
                score.offset_s,
                score.robust_mean_nis,
                score.median_nis,
                score.measurement_count,
                score.measurement_count,
                0,
            )
            for score in frozen_scores
        ),
        peak_speed_mps=float(
            np.max(np.linalg.norm(truth.velocity_ned_mps, axis=1))
        ),
    )
    frozen_interval = estimate_time_offset_profile_interval(frozen_calibration)
    frozen_nis_matrix = frozen_trajectory_nis_matrix(
        truth.time_s,
        truth_antenna_ned,
        measurement_times,
        measured_ned,
        data.rtk.std_ned_m[measurement_mask],
        candidates,
    )
    frozen_best_prediction = interpolate_columns(
        truth.time_s,
        truth_antenna_ned,
        measurement_times + frozen_interval.best_offset_s,
    )
    frozen_best_residual = measured_ned - frozen_best_prediction
    correlation = residual_correlation_diagnostics(
        frozen_best_residual,
        max_lag=min(int(args.correlation_max_lag), measurement_times.size - 1),
    )
    unconstrained_block_length = int(
        np.ceil(np.max(correlation.integrated_autocorrelation_time))
    )
    block_length = max(
        1,
        min(unconstrained_block_length, max(2, measurement_times.size // 2)),
    )
    bootstrap = bootstrap_frozen_time_offset(
        candidates,
        frozen_nis_matrix,
        block_length_epochs=block_length,
        replicate_count=int(args.bootstrap_replicates),
        random_seed=int(args.bootstrap_seed),
    )
    np.savetxt(
        args.output_dir / "frozen_bootstrap_offsets.csv",
        bootstrap.bootstrap_offsets_s[:, None],
        delimiter=",",
        header="bootstrap_offset_s",
        comments="",
    )
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(bootstrap.bootstrap_offsets_s * 1e3, bins=30, alpha=0.8)
    axis.axvline(bootstrap.point_estimate_s * 1e3, color="black", linestyle="--", label="Point estimate")
    axis.axvline(bootstrap.lower_offset_s * 1e3, color="tab:red", linestyle=":", label="95% interval")
    axis.axvline(bootstrap.upper_offset_s * 1e3, color="tab:red", linestyle=":")
    axis.set_xlabel("Frozen-trajectory time offset [ms]")
    axis.set_ylabel("Bootstrap count")
    axis.set_title("Moving-Block Bootstrap Time Offset")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "frozen_bootstrap_histogram.png", dpi=180)
    plt.close(fig)

    frozen_mean = np.asarray([item.robust_mean_nis for item in frozen_scores])
    eskf_mean = np.atleast_1d(eskf_table["robust_mean_nis"])
    ungated_mean = np.atleast_1d(ungated_table["robust_mean_nis"])
    frozen_objective = frozen_mean * measurement_times.size
    eskf_objective = eskf_mean * np.atleast_1d(eskf_table["measurement_count"])
    frozen_delta = frozen_objective - np.min(frozen_objective)
    eskf_delta = eskf_objective - np.min(eskf_objective)
    ungated_objective = ungated_mean * np.atleast_1d(
        ungated_table["measurement_count"]
    )
    ungated_delta = ungated_objective - np.min(ungated_objective)
    frozen_normalized = frozen_delta / max(float(np.max(frozen_delta)), 1e-15)
    eskf_normalized = eskf_delta / max(float(np.max(eskf_delta)), 1e-15)
    ungated_normalized = ungated_delta / max(
        float(np.max(ungated_delta)),
        1e-15,
    )
    output_table = np.column_stack(
        [
            candidates,
            frozen_mean,
            eskf_mean,
            frozen_delta,
            eskf_delta,
            frozen_normalized,
            eskf_normalized,
            ungated_mean,
            ungated_delta,
            ungated_normalized,
            np.atleast_1d(eskf_table["accepted_count"]),
            np.atleast_1d(eskf_table["rejected_count"]),
        ]
    )
    np.savetxt(
        args.output_dir / "objective_comparison.csv",
        output_table,
        delimiter=",",
        header=(
            "offset_s,frozen_robust_mean_nis,eskf_robust_mean_nis,"
            "frozen_delta_total_nis,eskf_delta_total_nis,frozen_normalized_delta,"
            "eskf_normalized_delta,ungated_eskf_robust_mean_nis,"
            "ungated_eskf_delta_total_nis,ungated_eskf_normalized_delta,"
            "eskf_accepted_count,eskf_rejected_count"
        ),
        comments="",
    )

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(candidates * 1e3, frozen_normalized, marker="o", label="Frozen truth trajectory")
    axis.plot(candidates * 1e3, eskf_normalized, marker="s", label="Sequential ESKF")
    axis.plot(
        candidates * 1e3,
        ungated_normalized,
        marker="^",
        label="Sequential ESKF, gate disabled",
    )
    axis.axvline(frozen_interval.best_offset_s * 1e3, color="tab:blue", linestyle="--")
    axis.axvline(eskf_summary["best_offset_s"] * 1e3, color="tab:orange", linestyle="--")
    axis.axvline(
        ungated_summary["best_offset_s"] * 1e3,
        color="tab:green",
        linestyle=":",
    )
    axis.set_xlabel("GNSS time offset [ms]")
    axis.set_ylabel("Normalized objective increase [-]")
    axis.set_title("Frozen Trajectory vs Sequential ESKF Time Objective")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "objective_comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "maturity": "offline objective-function diagnostic MVP",
        "offset_convention": "effective_time = reported_time + offset",
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "measurement_count": int(measurement_times.size),
        "lever_arm_b_m": lever.tolist(),
        "frozen_best_offset_s": frozen_interval.best_offset_s,
        "frozen_profile_interval": {
            "lower_offset_s": frozen_interval.lower_offset_s,
            "upper_offset_s": frozen_interval.upper_offset_s,
            "resolution_limited": frozen_interval.resolution_limited,
        },
        "frozen_block_bootstrap": {
            "point_estimate_s": bootstrap.point_estimate_s,
            "lower_offset_s": bootstrap.lower_offset_s,
            "upper_offset_s": bootstrap.upper_offset_s,
            "confidence_level": bootstrap.confidence_level,
            "replicate_count": bootstrap.replicate_count,
            "block_length_epochs": bootstrap.block_length_epochs,
            "unconstrained_correlation_block_length_epochs": (
                unconstrained_block_length
            ),
            "block_length_was_capped": bool(
                block_length < unconstrained_block_length
            ),
            "boundary_hit_fraction": bootstrap.boundary_hit_fraction,
            "grid_resolution_limited": bool(
                bootstrap.grid_resolution_limited
            ),
            "residual_lag1_autocorrelation_ned": (
                correlation.lag1_autocorrelation.tolist()
            ),
            "effective_sample_size_ned": (
                correlation.effective_sample_size.tolist()
            ),
        },
        "eskf_best_offset_s": eskf_summary["best_offset_s"],
        "ungated_eskf_best_offset_s": ungated_summary["best_offset_s"],
        "objective_best_difference_s": (
            eskf_summary["best_offset_s"] - frozen_interval.best_offset_s
        ),
        "ungated_vs_frozen_best_difference_s": (
            ungated_summary["best_offset_s"] - frozen_interval.best_offset_s
        ),
        "normalized_curve_correlation": float(
            np.corrcoef(frozen_normalized, eskf_normalized)[0, 1]
        ),
        "frozen_vs_ungated_curve_correlation": float(
            np.corrcoef(frozen_normalized, ungated_normalized)[0, 1]
        ),
        "gated_vs_ungated_curve_correlation": float(
            np.corrcoef(eskf_normalized, ungated_normalized)[0, 1]
        ),
        "eskf_best_candidate_accepted_count": int(
            np.atleast_1d(eskf_table["accepted_count"])[np.argmin(eskf_mean)]
        ),
        "eskf_best_candidate_rejected_count": int(
            np.atleast_1d(eskf_table["rejected_count"])[np.argmin(eskf_mean)]
        ),
        "ungated_best_candidate_accepted_count": int(
            np.atleast_1d(ungated_table["accepted_count"])[np.argmin(ungated_mean)]
        ),
        "ungated_best_candidate_rejected_count": int(
            np.atleast_1d(ungated_table["rejected_count"])[np.argmin(ungated_mean)]
        ),
        "interpretation": (
            "Frozen scoring isolates timestamp-to-trajectory alignment. Sequential "
            "ESKF scoring also includes state correction, covariance evolution, and gating."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_comparison(args)
    print(f"Frozen best offset: {summary['frozen_best_offset_s'] * 1e3:+.3f} ms")
    print(f"ESKF best offset: {summary['eskf_best_offset_s'] * 1e3:+.3f} ms")
    print(f"Wrote objective comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
