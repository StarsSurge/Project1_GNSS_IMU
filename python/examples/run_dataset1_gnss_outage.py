"""Compare baseline, direct reacquisition, and cautious GNSS recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_dataset1_eskf import REPO_ROOT, run_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject a GNSS outage into dataset1 and compare recovery policies."
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data" / "dataset1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_gnss_outage",
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--outage-start-s", type=float, default=10.0)
    parser.add_argument("--outage-end-s", type=float, default=20.0)
    parser.add_argument(
        "--imu-profile",
        choices=("mems", "navigation-grade"),
        default="navigation-grade",
    )
    parser.add_argument(
        "--initialization",
        choices=("truth", "gyrocompass", "external-yaw"),
        default="truth",
        help="Truth initialization isolates outage behavior and is evaluation-only.",
    )
    parser.add_argument("--initial-yaw-deg", type=float, default=None)
    return parser.parse_args()


def _scenario_args(
    args: argparse.Namespace,
    name: str,
    *,
    outage: bool,
    integrity_mode: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir / name,
        duration_s=args.duration_s,
        imu_profile=args.imu_profile,
        lever_arm_b_m=(0.14722696, -0.29821683, -0.18079014),
        initialization=args.initialization,
        initial_yaw_deg=args.initial_yaw_deg,
        alignment_duration_s=30.0,
        gnss_update_mode="delayed-replay",
        gnss_time_offset_s=0.0,
        fixed_lag_s=2.0,
        gnss_outage=(
            [(args.outage_start_s, args.outage_end_s)] if outage else []
        ),
        gnss_integrity_mode=integrity_mode,
        gnss_outage_timeout_s=2.0,
        gnss_recovery_accepts=3,
        gnss_recovery_initial_std_scale=10.0,
        gnss_recovery_scale_decay=0.5,
    )


def _metrics(path: Path, outage_start_s: float, outage_end_s: float) -> dict[str, float]:
    solution = np.genfromtxt(path / "eskf_solution.csv", delimiter=",", names=True)
    elapsed_s = solution["time_s"] - solution["time_s"][0]
    position_error_m = np.sqrt(
        solution["error_n_m"] ** 2
        + solution["error_e_m"] ** 2
        + solution["error_d_m"] ** 2
    )
    outage_mask = (elapsed_s >= outage_start_s) & (elapsed_s < outage_end_s)
    post_mask = (elapsed_s >= outage_end_s) & (elapsed_s < outage_end_s + 5.0)
    reacquisition_mask = (
        (elapsed_s[1:] >= outage_end_s)
        & (elapsed_s[1:] < outage_end_s + 1.5)
    )
    error_vectors_m = np.column_stack(
        [solution["error_n_m"], solution["error_e_m"], solution["error_d_m"]]
    )
    error_steps_m = np.linalg.norm(np.diff(error_vectors_m, axis=0), axis=1)
    end_index = int(np.argmin(np.abs(elapsed_s - outage_end_s)))
    return {
        "outage_peak_position_error_m": float(np.max(position_error_m[outage_mask])),
        "outage_end_position_error_m": float(position_error_m[end_index]),
        "post_5s_position_rms_m": float(
            np.sqrt(np.mean(position_error_m[post_mask] ** 2))
        ),
        "max_reacquisition_error_step_m": float(
            np.max(error_steps_m[reacquisition_mask])
        ),
        "final_position_error_m": float(position_error_m[-1]),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.outage_start_s < args.outage_end_s < args.duration_s:
        raise ValueError("require 0 <= outage_start < outage_end < duration")
    if args.imu_profile == "mems" and args.initialization == "gyrocompass":
        raise ValueError("MEMS profile cannot use gyrocompass initialization")

    scenarios = {
        "baseline": _scenario_args(args, "baseline", outage=False, integrity_mode="recovery"),
        "outage_direct": _scenario_args(args, "outage_direct", outage=True, integrity_mode="off"),
        "outage_recovery": _scenario_args(args, "outage_recovery", outage=True, integrity_mode="recovery"),
    }
    metrics = {}
    for name, scenario_args in scenarios.items():
        run_replay(scenario_args)
        metrics[name] = _metrics(
            scenario_args.output_dir,
            args.outage_start_s,
            args.outage_end_s,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "maturity": "fault-injection MVP; not field validation",
        "imu_profile": args.imu_profile,
        "initialization": args.initialization,
        "outage_interval_elapsed_s": [args.outage_start_s, args.outage_end_s],
        "scenarios": metrics,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    for name, scenario_args in scenarios.items():
        solution = np.genfromtxt(
            scenario_args.output_dir / "eskf_solution.csv",
            delimiter=",",
            names=True,
        )
        elapsed_s = solution["time_s"] - solution["time_s"][0]
        error_m = np.sqrt(
            solution["error_n_m"] ** 2
            + solution["error_e_m"] ** 2
            + solution["error_d_m"] ** 2
        )
        axis.plot(elapsed_s, error_m, label=name.replace("_", " "))
    axis.axvspan(
        args.outage_start_s,
        args.outage_end_s,
        color="tab:red",
        alpha=0.12,
        label="GNSS outage",
    )
    axis.set_xlabel("Elapsed time [s]")
    axis.set_ylabel("3D position error [m]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "position_error_comparison.png", dpi=180)
    plt.close(fig)

    print(f"Wrote GNSS outage comparison to {args.output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
