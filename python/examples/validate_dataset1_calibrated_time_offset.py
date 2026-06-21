"""Validate a frozen-profile time calibration on a disjoint Dataset1 segment."""

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
EXAMPLES_DIR = REPO_ROOT / "python" / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from run_dataset1_eskf import run_replay  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare zero and pre-calibrated GNSS offsets on held-out data."
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data" / "dataset1")
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "dataset1_time_objective_comparison"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_time_offset_validation",
    )
    parser.add_argument("--validation-duration-s", type=float, default=60.0)
    parser.add_argument(
        "--lever-arm-b-m",
        type=float,
        nargs=3,
        default=(0.14722696, -0.29821683, -0.18079014),
        metavar=("FORWARD", "RIGHT", "DOWN"),
    )
    return parser.parse_args()


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    if args.validation_duration_s <= 0.0:
        raise ValueError("validation_duration_s must be positive")
    calibration = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    calibrated_offset_s = float(
        calibration["frozen_block_bootstrap"]["point_estimate_s"]
    )
    bootstrap_lower_s = float(
        calibration["frozen_block_bootstrap"]["lower_offset_s"]
    )
    bootstrap_upper_s = float(
        calibration["frozen_block_bootstrap"]["upper_offset_s"]
    )
    calibration_start_s = float(calibration["start_time_s"])

    common = dict(
        dataset_dir=args.dataset_dir,
        duration_s=float(args.validation_duration_s),
        imu_profile="navigation-grade",
        lever_arm_b_m=tuple(args.lever_arm_b_m),
        initialization="gyrocompass",
        initial_yaw_deg=None,
        alignment_duration_s=30.0,
        gnss_update_mode="delayed-replay",
        fixed_lag_s=2.0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "zero_offset": 0.0,
        "calibrated_offset": calibrated_offset_s,
    }
    summaries: dict[str, dict[str, object]] = {}
    for name, offset_s in scenarios.items():
        output_dir = args.output_dir / name
        run_replay(
            SimpleNamespace(
                **common,
                output_dir=output_dir,
                gnss_time_offset_s=offset_s,
            )
        )
        summaries[name] = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )

    validation_end_s = (
        float(summaries["zero_offset"]["initialization"]["gnss_time_s"])
        + float(args.validation_duration_s)
    )
    if validation_end_s >= calibration_start_s:
        raise ValueError(
            "validation segment overlaps calibration window; shorten validation duration"
        )

    metric_names = (
        "position_rms_3d_m",
        "attitude_error_rms_deg",
    )
    comparison = {}
    for metric in metric_names:
        baseline = float(summaries["zero_offset"][metric])
        calibrated = float(summaries["calibrated_offset"][metric])
        comparison[metric] = {
            "zero_offset": baseline,
            "calibrated_offset": calibrated,
            "change": calibrated - baseline,
            "relative_change": (
                (calibrated - baseline) / baseline if baseline > 0.0 else None
            ),
        }
    for axis, label in enumerate(("north", "east", "down")):
        metric = f"velocity_rms_{label}_mps"
        baseline = float(summaries["zero_offset"]["velocity_rms_ned_mps"][axis])
        calibrated = float(
            summaries["calibrated_offset"]["velocity_rms_ned_mps"][axis]
        )
        comparison[metric] = {
            "zero_offset": baseline,
            "calibrated_offset": calibrated,
            "change": calibrated - baseline,
        }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["zero", "calibrated"]
    axes[0].bar(
        labels,
        [
            comparison["position_rms_3d_m"]["zero_offset"],
            comparison["position_rms_3d_m"]["calibrated_offset"],
        ],
    )
    axes[0].set_ylabel("3D position RMS [m]")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(
        labels,
        [
            comparison["attitude_error_rms_deg"]["zero_offset"],
            comparison["attitude_error_rms_deg"]["calibrated_offset"],
        ],
    )
    axes[1].set_ylabel("Attitude error RMS [deg]")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Held-Out GNSS Time-Offset Validation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "validation_comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "maturity": "held-out offline validation MVP",
        "calibration_summary": str(args.calibration_summary),
        "calibrated_offset_s": calibrated_offset_s,
        "validation_duration_s": float(args.validation_duration_s),
        "calibration_start_s": calibration_start_s,
        "validation_end_s": validation_end_s,
        "segments_are_disjoint": True,
        "calibration_interval_contains_zero": (
            bootstrap_lower_s <= 0.0 <= bootstrap_upper_s
        ),
        "held_out_position_improved": (
            comparison["position_rms_3d_m"]["change"] < 0.0
        ),
        "configuration_recommendation": (
            "retain-zero-offset"
            if bootstrap_lower_s <= 0.0 <= bootstrap_upper_s
            and comparison["position_rms_3d_m"]["change"] >= 0.0
            else "requires-more-validation"
        ),
        "comparison": comparison,
        "gnss_updates": {
            name: {
                "accepted": scenario["accepted_gnss_updates"],
                "rejected": scenario["rejected_gnss_updates"],
            }
            for name, scenario in summaries.items()
        },
        "interpretation": (
            "The calibrated offset is fixed before replay. No parameter is selected "
            "from the held-out navigation metrics."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_validation(args)
    position = summary["comparison"]["position_rms_3d_m"]
    print(f"Calibrated GNSS offset: {summary['calibrated_offset_s'] * 1e3:+.3f} ms")
    print(
        "Held-out 3D position RMS: "
        f"{position['zero_offset']:.6f} -> {position['calibrated_offset']:.6f} m"
    )
    print(f"Wrote held-out validation to {args.output_dir}")


if __name__ == "__main__":
    main()
