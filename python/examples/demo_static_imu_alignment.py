"""Estimate initial attitude from dataset1's pre-truth static IMU window."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from gnss_imu import (  # noqa: E402
    StaticAlignmentConfig,
    TimedIMUIncrement,
    initialize_from_static_imu,
)


def main() -> None:
    imu_path = REPO_ROOT / "data" / "dataset1" / "Leador-A15.txt"
    table = np.loadtxt(imu_path, max_rows=10000)
    dt = np.diff(table[:, 0])
    samples = [
        TimedIMUIncrement(
            table[index, 0],
            table[index, 1:4],
            table[index, 4:7],
            dt[index - 1],
        )
        for index in range(1, table.shape[0])
    ]
    result = initialize_from_static_imu(
        samples,
        latitude_rad=np.deg2rad(30.4447873710),
        longitude_rad=np.deg2rad(114.4718631927),
        height_m=20.904,
        use_gyrocompass=True,
        config=StaticAlignmentConfig(min_samples=5000, min_duration_s=20.0),
    )
    diagnostic = result.diagnostics
    truth_rpy_deg = np.array([0.85266, -2.03401, 185.67273])
    estimate_rpy_deg = np.array(
        [diagnostic.roll_deg, diagnostic.pitch_deg, diagnostic.yaw_deg]
    )
    error_deg = estimate_rpy_deg - truth_rpy_deg
    error_deg[2] = (error_deg[2] + 180.0) % 360.0 - 180.0

    print("Dataset1 static IMU initialization")
    print(f"Samples: {diagnostic.sample_count}")
    print(f"Duration: {diagnostic.duration_s:.3f} s")
    print(f"Yaw source: {diagnostic.yaw_source}")
    print(f"Estimated RPY [deg]: {estimate_rpy_deg}")
    print(f"Truth RPY [deg]:     {truth_rpy_deg}")
    print(f"RPY error [deg]:     {error_deg}")
    print(f"Estimated gyro bias [rad/s]: {result.state.gyro_bias_rps}")
    print(f"Gyro std [rad/s]: {diagnostic.gyro_std_rps}")
    print(f"Accel std [m/s^2]: {diagnostic.accel_std_mps2}")
    print(
        "Note: gyrocompassing is intended for sufficiently stable high-grade "
        "gyros; MEMS generally needs an external yaw source."
    )


if __name__ == "__main__":
    main()
