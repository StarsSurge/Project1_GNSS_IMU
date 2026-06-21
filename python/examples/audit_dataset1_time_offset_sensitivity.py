"""Audit Dataset1 time-offset sensitivity to initialization and filter parameters."""

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

from cross_validate_dataset1_gnss_time_offset import run_cross_validation  # noqa: E402
from run_dataset1_eskf import _build_initial_state  # noqa: E402
from gnss_imu import (  # noqa: E402
    ESKFConfig,
    FixedLagGNSSFusion,
    GNSSPositionMeasurement,
    IMUNoiseModel,
    LooselyCoupledESKF,
    TimedIMUIncrement,
    clone_eskf_state,
    load_dataset1,
)

SCENARIO_NAMES = (
    "truth-baseline",
    "operational-initialization",
    "zero-lever-arm",
    "lever-arm-perturbed",
    "imu-noise-half",
    "imu-noise-double",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how GNSS time-offset calibration changes under controlled perturbations."
    )
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data" / "dataset1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_time_offset_sensitivity",
    )
    parser.add_argument(
        "--window-start-offsets-s",
        type=float,
        nargs="+",
        default=(70.0, 190.0, 250.0, 310.0),
    )
    parser.add_argument("--window-duration-s", type=float, default=10.0)
    parser.add_argument("--candidate-min-s", type=float, default=-0.05)
    parser.add_argument("--candidate-max-s", type=float, default=0.05)
    parser.add_argument("--candidate-step-s", type=float, default=0.01)
    parser.add_argument("--fixed-lag-s", type=float, default=1.0)
    parser.add_argument("--max-acceptable-shift-ms", type=float, default=5.0)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_NAMES,
        default=SCENARIO_NAMES,
    )
    parser.add_argument(
        "--lever-arm-b-m",
        type=float,
        nargs=3,
        default=(0.14722696, -0.29821683, -0.18079014),
        metavar=("FORWARD", "RIGHT", "DOWN"),
    )
    return parser.parse_args()


def build_operational_window_states(
    dataset_dir: Path,
    requested_start_offsets_s: np.ndarray,
    lever_arm_b_m: np.ndarray,
) -> tuple[dict[float, object], dict[str, object]]:
    """Continuously navigate from static alignment to each requested window."""
    data = load_dataset1(dataset_dir)
    evaluation_start_s = float(data.truth.time_s[0])
    config = ESKFConfig(
        imu_noise=IMUNoiseModel.navigation_grade_default(),
        gnss_lever_arm_b_m=lever_arm_b_m,
    )
    initialization_args = SimpleNamespace(
        initialization="gyrocompass",
        imu_profile="navigation-grade",
        alignment_duration_s=30.0,
        initial_yaw_deg=None,
    )
    state, metadata, consumed_gnss_index = _build_initial_state(
        data,
        initialization_args,
        config,
        evaluation_start_s,
    )
    eskf = LooselyCoupledESKF(state, config)
    fusion = FixedLagGNSSFusion(eskf, lag_s=2.0, gnss_time_offset_s=0.0)

    target_times: dict[float, float] = {}
    for offset in requested_start_offsets_s:
        query = evaluation_start_s + float(offset)
        index = int(np.searchsorted(data.imu.time_s, query, side="left"))
        if index <= 0 or index >= data.imu.time_s.size:
            raise ValueError(f"operational target +{offset} s is outside IMU data")
        target_times[float(offset)] = float(data.imu.time_s[index])

    imu_index = int(np.searchsorted(data.imu.time_s, eskf.state.time_s, side="left"))
    while imu_index < data.imu.time_s.size:
        if imu_index == 0:
            imu_index += 1
            continue
        dt_s = float(data.imu.time_s[imu_index] - data.imu.time_s[imu_index - 1])
        interval_start = float(data.imu.time_s[imu_index] - dt_s)
        if interval_start >= eskf.state.time_s - 1e-6:
            break
        imu_index += 1
    gnss_index = int(consumed_gnss_index) + 1
    captured: dict[float, object] = {}
    maximum_target = max(target_times.values())
    while imu_index < data.imu.time_s.size:
        time_s = float(data.imu.time_s[imu_index])
        if time_s > maximum_target + 1e-9:
            break
        dt_s = time_s - float(data.imu.time_s[imu_index - 1])
        fusion.process_imu(
            TimedIMUIncrement(
                time_s,
                data.imu.delta_angle_rad[imu_index],
                data.imu.delta_velocity_mps[imu_index],
                dt_s,
            )
        )
        while (
            gnss_index < data.rtk.time_s.size
            and data.rtk.time_s[gnss_index] <= eskf.state.time_s + 1e-9
        ):
            fusion.process_gnss(
                GNSSPositionMeasurement(
                    float(data.rtk.time_s[gnss_index]),
                    np.deg2rad(data.rtk.latitude_deg[gnss_index]),
                    np.deg2rad(data.rtk.longitude_deg[gnss_index]),
                    float(data.rtk.height_m[gnss_index]),
                    data.rtk.std_ned_m[gnss_index],
                ),
                arrival_time_s=eskf.state.time_s,
            )
            gnss_index += 1
        for offset, target_time in target_times.items():
            if offset not in captured and abs(eskf.state.time_s - target_time) <= 1e-8:
                captured[offset] = clone_eskf_state(eskf.state)
        imu_index += 1
    if len(captured) != len(target_times):
        missing = sorted(set(target_times) - set(captured))
        raise RuntimeError(f"failed to capture operational states for offsets {missing}")
    return captured, metadata


def run_sensitivity_audit(args: argparse.Namespace) -> dict[str, object]:
    starts = np.asarray(args.window_start_offsets_s, dtype=float)
    if starts.ndim != 1 or starts.size < 4 or not np.all(np.isfinite(starts)):
        raise ValueError("sensitivity audit requires at least four finite windows")
    threshold_s = float(args.max_acceptable_shift_ms) * 1e-3
    if not np.isfinite(threshold_s) or threshold_s <= 0.0:
        raise ValueError("max_acceptable_shift_ms must be positive and finite")
    baseline_lever = np.asarray(args.lever_arm_b_m, dtype=float)
    requested_scenarios = tuple(dict.fromkeys(args.scenarios))
    if "truth-baseline" not in requested_scenarios:
        raise ValueError("truth-baseline must be included to define scenario shifts")

    operational_states: dict[float, object] = {}
    operational_metadata: dict[str, object] | None = None
    if "operational-initialization" in requested_scenarios:
        operational_states, operational_metadata = build_operational_window_states(
            args.dataset_dir,
            starts,
            baseline_lever,
        )

    scenario_definitions = {
        "truth-baseline": (baseline_lever, 1.0, {}, "truth-assisted"),
        "operational-initialization": (
            baseline_lever,
            1.0,
            operational_states,
            "continuous-gyrocompass-GNSS-IMU",
        ),
        "zero-lever-arm": (np.zeros(3), 1.0, {}, "truth-assisted"),
        "lever-arm-perturbed": (
            baseline_lever + np.array([0.1, -0.1, 0.1]),
            1.0,
            {},
            "truth-assisted",
        ),
        "imu-noise-half": (baseline_lever, 0.5, {}, "truth-assisted"),
        "imu-noise-double": (baseline_lever, 2.0, {}, "truth-assisted"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_results: list[dict[str, object]] = []
    for scenario_name in requested_scenarios:
        lever, noise_scale, initial_states, initialization_label = scenario_definitions[
            scenario_name
        ]
        print(f"Running sensitivity scenario: {scenario_name}", flush=True)
        scenario_args = SimpleNamespace(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir / "scenarios" / scenario_name,
            window_start_offsets_s=starts.tolist(),
            window_duration_s=float(args.window_duration_s),
            candidate_min_s=float(args.candidate_min_s),
            candidate_max_s=float(args.candidate_max_s),
            candidate_step_s=float(args.candidate_step_s),
            fixed_lag_s=float(args.fixed_lag_s),
            lever_arm_b_m=lever.tolist(),
            imu_noise_scale=float(noise_scale),
            initial_states_by_start_offset=initial_states,
            initialization_label=initialization_label,
        )
        try:
            result = run_cross_validation(scenario_args)
            clock = result["clock_model"]
            scenario_results.append(
                {
                    "scenario": scenario_name,
                    "status": "valid",
                    "constant_offset_s": clock["constant_offset_s"],
                    "constant_offset_ci95_s": clock["constant_offset_ci95_s"],
                    "drift_ppm": clock["drift_ppm"],
                    "drift_ci95_ppm": clock["drift_ci95_ppm"],
                    "preferred_model": clock["preferred_model"],
                    "valid_window_count": result["valid_bounded_window_count"],
                    "lever_arm_b_m": lever.tolist(),
                    "imu_noise_scale": float(noise_scale),
                    "initialization_source": initialization_label,
                }
            )
        except ValueError as error:
            scenario_results.append(
                {"scenario": scenario_name, "status": "failed", "reason": str(error)}
            )

    baseline_candidates = [
        item
        for item in scenario_results
        if item["scenario"] == "truth-baseline" and item["status"] == "valid"
    ]
    if not baseline_candidates:
        failure_summary = {
            "maturity": "offline sensitivity audit MVP",
            "status": "failed",
            "reason": "truth-baseline scenario did not produce four bounded windows",
            "scenario_results": scenario_results,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(failure_summary, indent=2),
            encoding="utf-8",
        )
        raise ValueError(failure_summary["reason"])
    baseline = baseline_candidates[0]
    baseline_offset = float(baseline["constant_offset_s"])
    valid_results = [item for item in scenario_results if item["status"] == "valid"]
    for item in valid_results:
        shift = float(item["constant_offset_s"] - baseline_offset)
        item["shift_from_truth_baseline_s"] = shift
        item["exceeds_shift_threshold"] = abs(shift) > threshold_s

    table = np.asarray(
        [
            [
                index,
                item["constant_offset_s"],
                item["shift_from_truth_baseline_s"],
                item["drift_ppm"],
                item["valid_window_count"],
                float(item["exceeds_shift_threshold"]),
            ]
            for index, item in enumerate(valid_results)
        ]
    )
    np.savetxt(
        args.output_dir / "sensitivity_summary.csv",
        table,
        delimiter=",",
        header=(
            "scenario_index,constant_offset_s,shift_from_truth_baseline_s,"
            "drift_ppm,valid_window_count,exceeds_shift_threshold"
        ),
        comments="",
    )
    labels = [item["scenario"] for item in valid_results]
    shifts_ms = np.asarray(
        [item["shift_from_truth_baseline_s"] for item in valid_results]
    ) * 1e3
    fig, axis = plt.subplots(figsize=(10, 5))
    colors = ["tab:red" if item["exceeds_shift_threshold"] else "tab:blue" for item in valid_results]
    axis.bar(np.arange(len(labels)), shifts_ms, color=colors)
    axis.axhline(args.max_acceptable_shift_ms, color="black", linestyle="--")
    axis.axhline(-args.max_acceptable_shift_ms, color="black", linestyle="--")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel("Offset shift from truth baseline [ms]")
    axis.set_title("GNSS Time-Offset Sensitivity Audit")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "sensitivity_shifts.png", dpi=180)
    plt.close(fig)

    summary = {
        "maturity": "offline sensitivity audit MVP",
        "shift_threshold_ms": float(args.max_acceptable_shift_ms),
        "scenario_results": scenario_results,
        "operational_initialization": operational_metadata,
        "interpretation": (
            "A threshold exceedance indicates model sensitivity, not an automatic "
            "sensor failure or proof that the perturbed parameter is wrong."
        ),
        "limitations": [
            "Operational states assume zero GNSS offset during the preceding continuous navigation, so initialization and timing are not fully independent.",
            "The default windows cover only the early trajectory to keep continuous propagation reproducible in Python.",
            "Temperature, hardware timestamp diagnostics, and independent datasets are still required.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    try:
        summary = run_sensitivity_audit(args)
    except ValueError as error:
        print(f"Sensitivity audit failed: {error}")
        raise SystemExit(2) from None
    for item in summary["scenario_results"]:
        if item["status"] == "valid":
            print(
                f"{item['scenario']}: offset={item['constant_offset_s'] * 1e3:+.3f} ms, "
                f"shift={item['shift_from_truth_baseline_s'] * 1e3:+.3f} ms"
            )
        else:
            print(f"{item['scenario']}: FAILED ({item['reason']})")
    print(f"Wrote sensitivity artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
