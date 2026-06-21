"""Cross-validate Dataset1 GNSS time offset and compare clock models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from calibrate_dataset1_gnss_time_offset import run_calibration  # noqa: E402
from gnss_imu import compare_clock_offset_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate GNSS time offset in multiple motion windows, then compare "
            "constant-offset and linear clock-drift models."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data" / "dataset1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_time_offset_cross_validation",
    )
    parser.add_argument(
        "--window-start-offsets-s",
        type=float,
        nargs="+",
        default=(70.0, 650.0, 1230.0, 1810.0, 2390.0, 2970.0),
        help="Window starts relative to truth start [s]; at least four are required.",
    )
    parser.add_argument("--window-duration-s", type=float, default=30.0)
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


def run_cross_validation(args: argparse.Namespace) -> dict[str, object]:
    starts = np.asarray(args.window_start_offsets_s, dtype=float)
    if starts.ndim != 1 or starts.size < 4 or not np.all(np.isfinite(starts)):
        raise ValueError("at least four finite window start offsets are required")
    if np.unique(starts).size != starts.size:
        raise ValueError("window start offsets must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    windows: list[dict[str, object]] = []
    initial_states = getattr(args, "initial_states_by_start_offset", {})
    noise_scale = float(getattr(args, "imu_noise_scale", 1.0))
    initialization_label = getattr(args, "initialization_label", "truth-assisted")
    for index, start_offset_s in enumerate(starts):
        window_output = args.output_dir / "windows" / f"window_{index:02d}"
        single_args = SimpleNamespace(
            dataset_dir=args.dataset_dir,
            output_dir=window_output,
            start_offset_s=float(start_offset_s),
            duration_s=float(args.window_duration_s),
            candidate_min_s=float(args.candidate_min_s),
            candidate_max_s=float(args.candidate_max_s),
            candidate_step_s=float(args.candidate_step_s),
            fixed_lag_s=float(args.fixed_lag_s),
            lever_arm_b_m=tuple(args.lever_arm_b_m),
            imu_noise_scale=noise_scale,
            initial_state_override=initial_states.get(float(start_offset_s)),
            initialization_label=initialization_label,
        )
        print(f"Calibrating window {index + 1}/{starts.size} at +{start_offset_s:.1f} s")
        try:
            summary = run_calibration(single_args)
            interval = summary["profile_interval"]
            bounded = bool(interval["lower_bounded"] and interval["upper_bounded"])
            windows.append(
                {
                    "window_index": index,
                    "requested_start_offset_s": float(start_offset_s),
                    "status": "valid" if bounded else "unbounded-profile",
                    **summary,
                }
            )
        except ValueError as error:
            windows.append(
                {
                    "window_index": index,
                    "requested_start_offset_s": float(start_offset_s),
                    "status": "failed",
                    "reason": str(error),
                }
            )

    valid = [window for window in windows if window["status"] == "valid"]
    if len(valid) < 4:
        raise ValueError(
            "fewer than four windows have bounded profile intervals; "
            "increase duration or candidate range"
        )
    center_times = np.asarray(
        [(window["start_time_s"] + window["end_time_s"]) / 2.0 for window in valid]
    )
    offsets = np.asarray([window["best_offset_s"] for window in valid])
    uncertainties = np.asarray(
        [window["profile_interval"]["standard_uncertainty_s"] for window in valid]
    )
    comparison = compare_clock_offset_models(center_times, offsets, uncertainties)

    rows = []
    for window in windows:
        if window["status"] == "valid":
            interval = window["profile_interval"]
            rows.append(
                [
                    window["window_index"],
                    window["requested_start_offset_s"],
                    window["start_time_s"],
                    window["end_time_s"],
                    window["best_offset_s"],
                    interval["lower_offset_s"],
                    interval["upper_offset_s"],
                    interval["standard_uncertainty_s"],
                    window["best_robust_mean_nis"],
                    window["gnss_measurement_count"],
                ]
            )
    np.savetxt(
        args.output_dir / "window_estimates.csv",
        np.asarray(rows),
        delimiter=",",
        header=(
            "window_index,requested_start_offset_s,start_time_s,end_time_s,"
            "best_offset_s,lower_offset_s,upper_offset_s,standard_uncertainty_s,"
            "best_robust_mean_nis,gnss_measurement_count"
        ),
        comments="",
    )

    elapsed = center_times - comparison.reference_time_s
    lower_errors = offsets - np.asarray(
        [window["profile_interval"]["lower_offset_s"] for window in valid]
    )
    upper_errors = np.asarray(
        [window["profile_interval"]["upper_offset_s"] for window in valid]
    ) - offsets
    plot_time = np.linspace(float(np.min(elapsed)), float(np.max(elapsed)), 200)
    drift_line = comparison.linear_offset_at_reference_s + comparison.drift_s_per_s * plot_time
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(
        elapsed,
        offsets * 1e3,
        yerr=np.vstack([lower_errors, upper_errors]) * 1e3,
        fmt="o",
        capsize=4,
        label="Window estimate and profile interval",
    )
    axis.axhline(
        comparison.constant_offset_s * 1e3,
        color="tab:green",
        linestyle="--",
        label="Constant model",
    )
    axis.plot(plot_time, drift_line * 1e3, color="tab:red", label="Linear drift model")
    axis.set_xlabel("Time from weighted reference epoch [s]")
    axis.set_ylabel("GNSS time offset [ms]")
    axis.set_title("Cross-Window GNSS Time-Offset Validation")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "clock_model_comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "maturity": "cross-window offline timing calibration MVP",
        "offset_convention": "effective_time = reported_time + offset",
        "requested_window_count": int(starts.size),
        "valid_bounded_window_count": len(valid),
        "initialization_source": initialization_label,
        "imu_noise_scale": noise_scale,
        "lever_arm_b_m": np.asarray(args.lever_arm_b_m, dtype=float).tolist(),
        "clock_model": asdict(comparison),
        "windows": windows,
        "limitations": [
            "Window intervals are approximate profile-NIS diagnostics, not coverage-certified confidence intervals.",
            "The linear model distinguishes only constant offset from first-order drift; jumps and nonlinear clock behavior remain unmodeled.",
            "Truth-assisted window initialization is retained to isolate timing-model behavior.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_cross_validation(args)
    model = summary["clock_model"]
    print(f"Preferred clock model: {model['preferred_model']}")
    print(f"Estimated drift: {model['drift_ppm']:+.6f} ppm")
    print(f"Wrote cross-validation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
