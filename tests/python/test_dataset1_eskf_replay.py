"""Short real-data integration test for the dataset1 ESKF main flow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "python" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from run_dataset1_eskf import run_replay  # noqa: E402


def test_dataset1_short_replay_writes_finite_auditable_outputs(tmp_path) -> None:
    result = run_replay(
        SimpleNamespace(
            dataset_dir=PROJECT_ROOT / "data" / "dataset1",
            output_dir=tmp_path,
            duration_s=1.1,
            imu_profile="navigation-grade",
            lever_arm_b_m=(0.14722696, -0.29821683, -0.18079014),
            initialization="truth",
            initial_yaw_deg=None,
            alignment_duration_s=30.0,
        )
    )

    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))
    solution = np.loadtxt(
        tmp_path / "eskf_solution.csv",
        delimiter=",",
        skiprows=1,
    )

    assert summary["maturity"] == "verified MVP / production-oriented baseline"
    assert summary["accepted_gnss_updates"] >= 1
    assert summary["skipped_unsynchronized_gnss"] == 0
    assert solution.shape[0] >= 100
    assert np.all(np.isfinite(solution))
    assert result["eskf"].accepted_gnss_updates >= 1


def test_dataset1_gyrocompass_initialization_replaces_truth_attitude(tmp_path) -> None:
    result = run_replay(
        SimpleNamespace(
            dataset_dir=PROJECT_ROOT / "data" / "dataset1",
            output_dir=tmp_path,
            duration_s=1.1,
            imu_profile="navigation-grade",
            lever_arm_b_m=(0.14722696, -0.29821683, -0.18079014),
            initialization="gyrocompass",
            initial_yaw_deg=None,
            alignment_duration_s=30.0,
        )
    )

    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))

    assert summary["initialization"]["mode"] == "gyrocompass"
    assert summary["initialization"]["yaw_source"] == "gyrocompass"
    assert summary["initialization"]["sample_count"] > 5000
    assert summary["accepted_gnss_updates"] >= 1
    assert summary["attitude_error_rms_deg"] < 2.0
    assert result["initialization"]["mode"] == "gyrocompass"


def test_mems_replay_requires_external_yaw(tmp_path) -> None:
    with np.testing.assert_raises_regex(ValueError, "must use external-yaw"):
        run_replay(
            SimpleNamespace(
                dataset_dir=PROJECT_ROOT / "data" / "dataset1",
                output_dir=tmp_path,
                duration_s=0.1,
                imu_profile="mems",
                lever_arm_b_m=(0.14722696, -0.29821683, -0.18079014),
                initialization="gyrocompass",
                initial_yaw_deg=None,
                alignment_duration_s=30.0,
            )
        )


def test_mems_external_yaw_initialization_runs(tmp_path) -> None:
    result = run_replay(
        SimpleNamespace(
            dataset_dir=PROJECT_ROOT / "data" / "dataset1",
            output_dir=tmp_path,
            duration_s=1.1,
            imu_profile="mems",
            lever_arm_b_m=(0.14722696, -0.29821683, -0.18079014),
            initialization="external-yaw",
            initial_yaw_deg=185.67273,
            alignment_duration_s=10.0,
        )
    )
    summary = json.loads((tmp_path / "summary.json").read_text("utf-8"))

    assert summary["initialization"]["mode"] == "external-yaw"
    assert summary["initialization"]["yaw_source"] == "external"
    assert summary["accepted_gnss_updates"] >= 1
    assert result["eskf"].accepted_gnss_updates >= 1
