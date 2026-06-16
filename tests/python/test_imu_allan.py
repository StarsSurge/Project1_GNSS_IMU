"""Engineering-boundary tests for real-log IMU Allan analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gnss_imu.imu_allan import (
    allan_deviation,
    extract_allan_parameters,
    load_imu_rate_csv,
    load_imu_rate_feather,
    overlapping_allan_deviation,
    validate_uniform_sampling,
)


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_allan_estimators_validate_inputs_and_deduplicate_taus() -> None:
    signal = np.arange(1000, dtype=float)
    requested = np.array([0.010, 0.011, 0.020])

    taus_non, _ = allan_deviation(signal, 100.0, requested)
    taus_over, _, _ = overlapping_allan_deviation(
        signal, 100.0, requested
    )
    np.testing.assert_allclose(taus_non, [0.01, 0.02])
    np.testing.assert_allclose(taus_over, [0.01, 0.02])

    with pytest.raises(ValueError, match="finite"):
        allan_deviation([0.0, np.nan, 1.0], 100.0, [0.1])
    with pytest.raises(ValueError, match="positive"):
        overlapping_allan_deviation(signal, 0.0, [0.1])
    with pytest.raises(ValueError, match="positive"):
        allan_deviation(signal, 100.0, [-0.1])


def test_uniform_sampling_rejects_gap_and_timestamp_reversal() -> None:
    timestamps = np.arange(100, dtype=float) * 0.01
    statistics = validate_uniform_sampling(timestamps)
    assert statistics["sample_rate_hz"] == pytest.approx(100.0)

    with_gap = timestamps.copy()
    with_gap[50:] += 0.04
    with pytest.raises(ValueError, match="exceeds"):
        validate_uniform_sampling(with_gap)

    reversed_time = timestamps.copy()
    reversed_time[50] = reversed_time[49]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_uniform_sampling(reversed_time)


def test_load_rate_csv_applies_time_and_value_units(tmp_path: Path) -> None:
    csv_path = tmp_path / "gyro.csv"
    write_csv(
        csv_path,
        [
            "timestamp_ms,gx_dps,gy_dps",
            "0,1.0,-2.0",
            "10,1.1,-2.1",
            "20,1.2,-2.2",
            "30,1.3,-2.3",
        ],
    )
    timestamps, rates, statistics = load_imu_rate_csv(
        csv_path,
        timestamp_column="timestamp_ms",
        value_columns=("gx_dps", "gy_dps"),
        timestamp_scale_to_seconds=1e-3,
        value_scale_to_si=np.pi / 180.0,
    )

    np.testing.assert_allclose(timestamps, [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(rates[0], np.deg2rad([1.0, -2.0]))
    assert statistics["sample_rate_hz"] == pytest.approx(100.0)


def test_load_increment_csv_divides_by_each_interval(tmp_path: Path) -> None:
    csv_path = tmp_path / "delta_angle.csv"
    write_csv(
        csv_path,
        [
            "time_s,dx",
            "0.00,0.000",
            "0.01,0.010",
            "0.02,0.020",
            "0.03,0.030",
        ],
    )
    timestamps, rates, _ = load_imu_rate_csv(
        csv_path,
        timestamp_column="time_s",
        value_columns=("dx",),
        input_kind="increment",
    )

    np.testing.assert_allclose(timestamps, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(rates[:, 0], [1.0, 2.0, 3.0])


def test_integer_epoch_timestamps_preserve_nanosecond_intervals(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "epoch.csv"
    origin = 1_660_100_288_344_525_469
    write_csv(
        csv_path,
        [
            "timestamp_ns,gx",
            f"{origin},0.0",
            f"{origin + 2_500_000},0.1",
            f"{origin + 5_000_000},0.2",
            f"{origin + 7_500_000},0.3",
        ],
    )
    timestamps, _, statistics = load_imu_rate_csv(
        csv_path,
        timestamp_column="timestamp_ns",
        value_columns=("gx",),
        timestamp_scale_to_seconds=1e-9,
    )

    np.testing.assert_allclose(timestamps, [0.0, 0.0025, 0.005, 0.0075])
    assert statistics["sample_rate_hz"] == pytest.approx(400.0)


def test_load_feather_columns_and_units(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    feather = pytest.importorskip("pyarrow.feather")
    table = pyarrow.table({
        "timestamp_ns": np.array(
            [1_000_000_000, 1_010_000_000, 1_020_000_000, 1_030_000_000],
            dtype=np.int64,
        ),
        "gx": np.array([1.0, 2.0, 3.0, 4.0]),
        "gy": np.array([-1.0, -2.0, -3.0, -4.0]),
    })
    path = tmp_path / "imu.feather"
    feather.write_feather(table, path)

    timestamps, rates, statistics = load_imu_rate_feather(
        path,
        timestamp_column="timestamp_ns",
        value_columns=("gx", "gy"),
        timestamp_scale_to_seconds=1e-9,
        value_scale_to_si=0.5,
    )

    np.testing.assert_allclose(timestamps, [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(rates[0], [0.5, -0.5])
    assert statistics["sample_rate_hz"] == pytest.approx(100.0)


def test_extract_allan_parameters_uses_documented_conversions() -> None:
    taus = np.logspace(-2, 2, 100)
    white_coefficient = 0.03
    deviations = white_coefficient / np.sqrt(taus)
    parameters = extract_allan_parameters(
        taus,
        deviations,
        white_fit_range=(0.01, 1.0),
    )

    white = parameters["white_noise"]
    assert white["slope"] == pytest.approx(-0.5)
    assert white["coefficient"] == pytest.approx(white_coefficient)
    assert white["continuous_psd"] == pytest.approx(
        white_coefficient**2
    )
    assert white["is_valid"]
    assert parameters["rate_random_walk"] is None


def test_load_csv_rejects_missing_columns_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bad.csv"
    write_csv(
        csv_path,
        [
            "time_s,gx",
            "0.00,0.1",
            "0.01,nan",
            "0.02,0.2",
        ],
    )
    with pytest.raises(ValueError, match="missing columns"):
        load_imu_rate_csv(
            csv_path,
            timestamp_column="time_s",
            value_columns=("gy",),
        )
    with pytest.raises(ValueError, match="NaN"):
        load_imu_rate_csv(
            csv_path,
            timestamp_column="time_s",
            value_columns=("gx",),
        )


def test_real_csv_cli_writes_auditable_outputs(tmp_path: Path) -> None:
    csv_path = tmp_path / "stationary_gyro.csv"
    rows = ["timestamp_ms,gx,gy,gz"]
    rng = np.random.default_rng(21)
    for index, values in enumerate(rng.normal(0.0, 0.01, size=(2000, 3))):
        rows.append(
            f"{index * 10},{values[0]},{values[1]},{values[2]}"
        )
    write_csv(csv_path, rows)

    output_dir = tmp_path / "outputs"
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "python")
    subprocess.run(
        [
            sys.executable,
            str(repository_root / "python/examples/analyze_imu_allan.py"),
            str(csv_path),
            "--timestamp-column",
            "timestamp_ms",
            "--timestamp-unit",
            "ms",
            "--value-columns",
            "gx,gy,gz",
            "--tau-count",
            "15",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert (output_dir / "allan_deviation.csv").is_file()
    assert (output_dir / "allan_deviation.png").is_file()
    assert (output_dir / "allan_parameter_summary.png").is_file()
    assert (output_dir / "allan_difference_pairs.png").is_file()
    assert (output_dir / "allan_parameters.csv").is_file()
    assert (output_dir / "allan_parameter_report.zh-CN.md").is_file()
    metadata = json.loads(
        (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["maturity"] == "verified MVP; not deployment-ready"
    assert metadata["sample_count"] == 2000
    assert metadata["sampling"]["sample_rate_hz"] == pytest.approx(100.0)
