"""Run the production-oriented GNSS/IMU loose ESKF baseline on dataset1."""

from __future__ import annotations

import argparse
import csv
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
    ESKFConfig,
    ESKFState,
    GNSSIntegrityConfig,
    GNSSIntegrityManager,
    GNSSIntegrityState,
    GNSSPositionMeasurement,
    GNSSUpdateResult,
    IMUNoiseModel,
    LooselyCoupledESKF,
    FixedLagGNSSFusion,
    TimedIMUIncrement,
    StaticAlignmentConfig,
    apply_ned_position_delta,
    default_initial_covariance,
    euler_zyx_to_quat,
    geodetic_to_ned,
    load_dataset1,
    initialize_from_static_imu,
    quat_to_dcm,
    rpy_deg_to_body_to_ned,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay dataset1 through a 15-state GNSS/IMU loose ESKF."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "dataset1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dataset1_eskf",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Replay duration from truth start; <=0 processes all available data.",
    )
    parser.add_argument(
        "--imu-profile",
        choices=("mems", "navigation-grade"),
        default="navigation-grade",
    )
    parser.add_argument(
        "--initialization",
        choices=("truth", "gyrocompass", "external-yaw"),
        default="gyrocompass",
        help="Initialization source. 'truth' is evaluation-only.",
    )
    parser.add_argument(
        "--initial-yaw-deg",
        type=float,
        default=None,
        help="Required by --initialization external-yaw.",
    )
    parser.add_argument(
        "--alignment-duration-s",
        type=float,
        default=30.0,
        help="Static IMU window before truth evaluation start.",
    )
    parser.add_argument(
        "--lever-arm-b-m",
        type=float,
        nargs=3,
        metavar=("FORWARD", "RIGHT", "DOWN"),
        default=(0.14722696, -0.29821683, -0.18079014),
        help="Calibrated GNSS antenna minus IMU lever arm in body FRD [m].",
    )
    parser.add_argument(
        "--gnss-update-mode",
        choices=("delayed-replay", "nearest"),
        default="delayed-replay",
        help="Exact fixed-lag replay or the legacy nearest-epoch update.",
    )
    parser.add_argument(
        "--gnss-time-offset-s",
        type=float,
        default=0.0,
        help="Timestamp convention: effective_time = reported_time + offset.",
    )
    parser.add_argument(
        "--fixed-lag-s",
        type=float,
        default=2.0,
        help="Retained IMU history for delayed GNSS replay [s].",
    )
    parser.add_argument(
        "--gnss-outage",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("START_S", "END_S"),
        help="Drop GNSS epochs in an elapsed-time interval; may be repeated.",
    )
    parser.add_argument(
        "--gnss-integrity-mode",
        choices=("off", "recovery"),
        default="recovery",
        help="Enable outage detection and cautious GNSS reacquisition.",
    )
    parser.add_argument("--gnss-outage-timeout-s", type=float, default=2.0)
    parser.add_argument("--gnss-recovery-accepts", type=int, default=3)
    parser.add_argument(
        "--gnss-recovery-initial-std-scale", type=float, default=10.0
    )
    parser.add_argument("--gnss-recovery-scale-decay", type=float, default=0.5)
    return parser.parse_args()


def _validated_outage_intervals(
    intervals: object,
) -> tuple[tuple[float, float], ...]:
    """Validate non-overlapping GNSS outage intervals in elapsed seconds."""
    validated = []
    for interval in intervals or []:
        if len(interval) != 2:
            raise ValueError("each gnss_outage interval needs start and end")
        start_s, end_s = (float(interval[0]), float(interval[1]))
        if not np.isfinite(start_s) or not np.isfinite(end_s):
            raise ValueError("gnss_outage bounds must be finite")
        if start_s < 0.0 or end_s <= start_s:
            raise ValueError("gnss_outage requires 0 <= start < end")
        validated.append((start_s, end_s))
    validated.sort()
    for previous, current in zip(validated, validated[1:]):
        if current[0] < previous[1]:
            raise ValueError("gnss_outage intervals must not overlap")
    return tuple(validated)


def _interpolate_truth(data, query_time_s: np.ndarray) -> dict[str, np.ndarray]:
    truth = data.truth
    return {
        "latitude_deg": np.interp(query_time_s, truth.time_s, truth.latitude_deg),
        "longitude_deg": np.interp(query_time_s, truth.time_s, truth.longitude_deg),
        "height_m": np.interp(query_time_s, truth.time_s, truth.height_m),
        "velocity_ned_mps": np.column_stack(
            [
                np.interp(query_time_s, truth.time_s, truth.velocity_ned_mps[:, i])
                for i in range(3)
            ]
        ),
        "attitude_rpy_deg": np.column_stack(
            [
                np.interp(query_time_s, truth.time_s, truth.attitude_rpy_deg[:, i])
                for i in range(3)
            ]
        ),
    }


def _build_initial_state(
    data,
    args: argparse.Namespace,
    config: ESKFConfig,
    evaluation_start_s: float,
) -> tuple[ESKFState, dict[str, object], int | None]:
    """Create the initial state and return the GNSS row consumed by it."""
    mode = getattr(args, "initialization", "truth")
    truth = data.truth
    if mode == "truth":
        state = ESKFState(
            time_s=evaluation_start_s,
            latitude_rad=np.deg2rad(truth.latitude_deg[0]),
            longitude_rad=np.deg2rad(truth.longitude_deg[0]),
            height_m=float(truth.height_m[0]),
            velocity_ned_mps=truth.velocity_ned_mps[0],
            q_bn=euler_zyx_to_quat(*truth.attitude_rpy_deg[0], degrees=True),
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
        return (
            state,
            {
                "mode": "truth",
                "note": "evaluation-only truth-assisted initialization",
            },
            None,
        )

    if mode == "gyrocompass" and args.imu_profile == "mems":
        raise ValueError(
            "MEMS profile must use external-yaw; static gyrocompass is not "
            "declared observable for this profile"
        )
    if mode == "external-yaw" and getattr(args, "initial_yaw_deg", None) is None:
        raise ValueError("external-yaw initialization requires --initial-yaw-deg")
    alignment_duration = float(getattr(args, "alignment_duration_s", 30.0))
    if not np.isfinite(alignment_duration) or alignment_duration < 5.0:
        raise ValueError("alignment_duration_s must be finite and at least 5 s")

    imu_indices = np.flatnonzero(
        (data.imu.time_s < evaluation_start_s)
        & (data.imu.time_s >= evaluation_start_s - alignment_duration)
    )
    imu_indices = imu_indices[imu_indices > 0]
    if imu_indices.size < 2:
        raise ValueError("no usable pre-start IMU alignment window")
    alignment_samples = [
        TimedIMUIncrement(
            float(data.imu.time_s[index]),
            data.imu.delta_angle_rad[index],
            data.imu.delta_velocity_mps[index],
            float(data.imu.time_s[index] - data.imu.time_s[index - 1]),
        )
        for index in imu_indices
    ]

    gnss_index = int(
        np.searchsorted(data.rtk.time_s, evaluation_start_s, side="right") - 1
    )
    if gnss_index < 0:
        raise ValueError("no GNSS position is available for initialization")
    position_std = data.rtk.std_ned_m[gnss_index]
    covariance = default_initial_covariance(
        position_std_m=float(np.max(position_std)),
        velocity_std_mps=0.05,
        attitude_std_deg=(2.0 if mode == "gyrocompass" else 1.0),
        accel_bias_std_mps2=0.03,
        gyro_bias_std_deg_s=0.01,
    )
    covariance[0:3, 0:3] = np.diag(position_std**2)
    alignment = initialize_from_static_imu(
        alignment_samples,
        latitude_rad=np.deg2rad(data.rtk.latitude_deg[gnss_index]),
        longitude_rad=np.deg2rad(data.rtk.longitude_deg[gnss_index]),
        height_m=float(data.rtk.height_m[gnss_index]),
        yaw_rad=(
            None
            if mode == "gyrocompass"
            else np.deg2rad(float(args.initial_yaw_deg))
        ),
        use_gyrocompass=(mode == "gyrocompass"),
        covariance=covariance,
        config=StaticAlignmentConfig(
            min_samples=400,
            min_duration_s=max(5.0, 0.8 * alignment_duration),
        ),
    )
    state = alignment.state
    # GNSS is at the antenna; convert the initialized position to the IMU point.
    lever_n = quat_to_dcm(state.q_bn) @ config.gnss_lever_arm_b_m
    lat, lon, height = apply_ned_position_delta(
        state.latitude_rad,
        state.longitude_rad,
        state.height_m,
        -lever_n,
    )
    state.latitude_rad = lat
    state.longitude_rad = lon
    state.height_m = height
    diagnostic = alignment.diagnostics
    metadata = {
        "mode": mode,
        "gnss_time_s": float(data.rtk.time_s[gnss_index]),
        "alignment_start_s": float(alignment_samples[0].time_s),
        "alignment_end_s": float(alignment_samples[-1].time_s),
        "sample_count": diagnostic.sample_count,
        "duration_s": diagnostic.duration_s,
        "yaw_source": diagnostic.yaw_source,
        "estimated_rpy_deg": [
            diagnostic.roll_deg,
            diagnostic.pitch_deg,
            diagnostic.yaw_deg,
        ],
        "gyro_std_rps": diagnostic.gyro_std_rps.tolist(),
        "accel_std_mps2": diagnostic.accel_std_mps2.tolist(),
        "gravity_norm_error_mps2": diagnostic.gravity_norm_error_mps2,
    }
    return state, metadata, gnss_index


def _write_outputs(
    output_dir: Path,
    history: dict[str, list],
    data,
    eskf: LooselyCoupledESKF,
    skipped_gnss: int,
    profile_name: str,
    initialization_metadata: dict[str, object],
    timing_metadata: dict[str, object],
    integrity_metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(history["time_s"])
    latitude_deg = np.rad2deg(np.asarray(history["latitude_rad"]))
    longitude_deg = np.rad2deg(np.asarray(history["longitude_rad"]))
    height_m = np.asarray(history["height_m"])
    velocity = np.asarray(history["velocity_ned_mps"])
    quaternion = np.asarray(history["q_bn"])
    accel_bias = np.asarray(history["accel_bias_mps2"])
    gyro_bias = np.asarray(history["gyro_bias_rps"])
    covariance_std = np.asarray(history["covariance_std"])

    truth = _interpolate_truth(data, times)
    reference_llh = (
        float(truth["latitude_deg"][0]),
        float(truth["longitude_deg"][0]),
        float(truth["height_m"][0]),
    )
    estimate_ned = geodetic_to_ned(
        latitude_deg,
        longitude_deg,
        height_m,
        reference_llh,
    )
    truth_ned = geodetic_to_ned(
        truth["latitude_deg"],
        truth["longitude_deg"],
        truth["height_m"],
        reference_llh,
    )
    position_error = estimate_ned - truth_ned
    velocity_error = velocity - truth["velocity_ned_mps"]

    estimate_dcm = np.stack([quat_to_dcm(q) for q in quaternion])
    truth_dcm = rpy_deg_to_body_to_ned(
        truth["attitude_rpy_deg"][:, 0],
        truth["attitude_rpy_deg"][:, 1],
        truth["attitude_rpy_deg"][:, 2],
    )
    relative_dcm = np.einsum("nij,njk->nik", np.swapaxes(truth_dcm, 1, 2), estimate_dcm)
    cosine_angle = np.clip((np.trace(relative_dcm, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    attitude_error_deg = np.rad2deg(np.arccos(cosine_angle))

    output_table = np.column_stack(
        [
            times,
            latitude_deg,
            longitude_deg,
            height_m,
            velocity,
            quaternion,
            accel_bias,
            gyro_bias,
            covariance_std,
            position_error,
            velocity_error,
            attitude_error_deg,
        ]
    )
    header = ",".join(
        [
            "time_s", "latitude_deg", "longitude_deg", "height_m",
            "vn_mps", "ve_mps", "vd_mps", "qw", "qx", "qy", "qz",
            "ba_x_mps2", "ba_y_mps2", "ba_z_mps2",
            "bg_x_rps", "bg_y_rps", "bg_z_rps",
            *[f"state_std_{i}" for i in range(15)],
            "error_n_m", "error_e_m", "error_d_m",
            "error_vn_mps", "error_ve_mps", "error_vd_mps",
            "attitude_error_deg",
        ]
    )
    np.savetxt(
        output_dir / "eskf_solution.csv",
        output_table,
        delimiter=",",
        header=header,
        comments="",
    )

    position_rms = np.sqrt(np.mean(position_error**2, axis=0))
    velocity_rms = np.sqrt(np.mean(velocity_error**2, axis=0))
    summary = {
        "maturity": "verified MVP / production-oriented baseline",
        "initialization": initialization_metadata,
        "gnss_timing": timing_metadata,
        "gnss_integrity": integrity_metadata,
        "imu_profile": profile_name,
        "duration_s": float(times[-1] - times[0]),
        "navigation_epochs": int(times.size),
        "accepted_gnss_updates": eskf.accepted_gnss_updates,
        "rejected_gnss_updates": eskf.rejected_gnss_updates,
        "skipped_unsynchronized_gnss": skipped_gnss,
        "position_rms_ned_m": position_rms.tolist(),
        "position_rms_3d_m": float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))),
        "velocity_rms_ned_mps": velocity_rms.tolist(),
        "attitude_error_rms_deg": float(np.sqrt(np.mean(attitude_error_deg**2))),
        "limitations": [
            "Static alignment is window-based; online window search and fine alignment are not implemented.",
            "Known calibration matrices are supported; online scale-factor, temperature, and vibration-rectification models are not.",
            "The configured GNSS time offset is constant; clock drift and per-epoch latency jitter are not estimated online.",
            "Noise profiles are starting points and must be replaced by calibrated device parameters.",
            "Python implementation is for offline verification, not certified hard real-time deployment.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    elapsed = times - times[0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, label in enumerate(("North", "East", "Down")):
        axes[axis].plot(elapsed, position_error[:, axis])
        axes[axis].set_ylabel(f"{label} [m]")
        axes[axis].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Elapsed time [s]")
    fig.suptitle("ESKF Position Error")
    fig.tight_layout()
    fig.savefig(output_dir / "position_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, label in enumerate(("VN", "VE", "VD")):
        axes[axis].plot(elapsed, velocity_error[:, axis])
        axes[axis].set_ylabel(f"{label} [m/s]")
        axes[axis].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Elapsed time [s]")
    fig.suptitle("ESKF Velocity Error")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_error.png", dpi=180)
    plt.close(fig)


def run_replay(args: argparse.Namespace) -> dict[str, object]:
    data = load_dataset1(args.dataset_dir)
    truth = data.truth
    start_time = float(truth.time_s[0])
    end_time = float(truth.time_s[-1])
    if args.duration_s > 0.0:
        end_time = min(end_time, start_time + args.duration_s)

    noise = (
        IMUNoiseModel.mems_default()
        if args.imu_profile == "mems"
        else IMUNoiseModel.navigation_grade_default()
    )
    config = ESKFConfig(
        imu_noise=noise,
        gnss_lever_arm_b_m=np.asarray(args.lever_arm_b_m, dtype=float),
    )
    state, initialization_metadata, consumed_gnss_index = _build_initial_state(
        data,
        args,
        config,
        start_time,
    )
    eskf = LooselyCoupledESKF(state, config)
    update_mode = getattr(args, "gnss_update_mode", "delayed-replay")
    gnss_time_offset_s = float(getattr(args, "gnss_time_offset_s", 0.0))
    fixed_lag_s = float(getattr(args, "fixed_lag_s", 2.0))
    if update_mode not in ("delayed-replay", "nearest"):
        raise ValueError("gnss_update_mode must be 'delayed-replay' or 'nearest'")
    if not np.isfinite(gnss_time_offset_s):
        raise ValueError("gnss_time_offset_s must be finite")
    integrity_mode = getattr(args, "gnss_integrity_mode", "recovery")
    if integrity_mode not in ("off", "recovery"):
        raise ValueError("gnss_integrity_mode must be 'off' or 'recovery'")
    outage_intervals = _validated_outage_intervals(
        getattr(args, "gnss_outage", [])
    )
    integrity_manager = None
    if integrity_mode == "recovery":
        integrity_config = GNSSIntegrityConfig(
            outage_timeout_s=float(
                getattr(args, "gnss_outage_timeout_s", 2.0)
            ),
            recovery_required_accepts=int(
                getattr(args, "gnss_recovery_accepts", 3)
            ),
            recovery_initial_std_scale=float(
                getattr(args, "gnss_recovery_initial_std_scale", 10.0)
            ),
            recovery_scale_decay=float(
                getattr(args, "gnss_recovery_scale_decay", 0.5)
            ),
        )
        integrity_manager = GNSSIntegrityManager(
            integrity_config,
            initial_measurement_time_s=float(
                initialization_metadata.get("gnss_time_s", eskf.state.time_s)
            ),
        )

    history: dict[str, list] = {
        "time_s": [], "latitude_rad": [], "longitude_rad": [], "height_m": [],
        "velocity_ned_mps": [], "q_bn": [], "accel_bias_mps2": [],
        "gyro_bias_rps": [], "covariance_std": [],
    }

    def record_state() -> None:
        history["time_s"].append(eskf.state.time_s)
        history["latitude_rad"].append(eskf.state.latitude_rad)
        history["longitude_rad"].append(eskf.state.longitude_rad)
        history["height_m"].append(eskf.state.height_m)
        history["velocity_ned_mps"].append(eskf.state.velocity_ned_mps.copy())
        history["q_bn"].append(eskf.state.q_bn.copy())
        history["accel_bias_mps2"].append(eskf.state.accel_bias_mps2.copy())
        history["gyro_bias_rps"].append(eskf.state.gyro_bias_rps.copy())
        history["covariance_std"].append(np.sqrt(np.diag(eskf.state.covariance)))

    imu_start = int(
        np.searchsorted(data.imu.time_s, eskf.state.time_s, side="left")
    )
    # A row timestamp marks the end of its increment.  Skip any row whose
    # integration interval starts before the initialized state epoch, even if
    # floating-point timestamp text differs by only a few nanoseconds.
    while imu_start < data.imu.time_s.size:
        if imu_start == 0:
            imu_start += 1
            continue
        candidate_dt = (
            data.imu.time_s[imu_start] - data.imu.time_s[imu_start - 1]
        )
        candidate_start = data.imu.time_s[imu_start] - candidate_dt
        if candidate_start >= eskf.state.time_s - 1e-6:
            break
        imu_start += 1
    gnss_index = (
        consumed_gnss_index + 1
        if consumed_gnss_index is not None
        else int(np.searchsorted(data.rtk.time_s, start_time, side="right"))
    )
    skipped_gnss = 0
    skipped_outage_gnss = 0
    delayed_results = []

    def is_in_outage(time_s: float) -> bool:
        elapsed_s = time_s - start_time
        return any(start <= elapsed_s < end for start, end in outage_intervals)

    def prepare_gnss(
        measurement: GNSSPositionMeasurement,
    ) -> tuple[GNSSPositionMeasurement, float, GNSSIntegrityState | None]:
        if integrity_manager is None:
            return measurement, 1.0, None
        return integrity_manager.prepare_measurement(measurement)

    def record_gnss_result(
        measurement: GNSSPositionMeasurement,
        scale: float,
        state_before: GNSSIntegrityState | None,
        update: GNSSUpdateResult,
    ) -> None:
        if integrity_manager is None:
            return
        if state_before is None:
            raise RuntimeError("integrity state is required when recovery is enabled")
        integrity_manager.record_update(
            measurement.time_s,
            accepted=update.accepted,
            nis=update.nis,
            measurement_std_scale=scale,
            state_before_update=state_before,
        )

    imu_index = imu_start
    fusion = (
        FixedLagGNSSFusion(
            eskf,
            lag_s=fixed_lag_s,
            gnss_time_offset_s=gnss_time_offset_s,
        )
        if update_mode == "delayed-replay"
        else None
    )

    def make_imu(index: int) -> TimedIMUIncrement:
        time_s = float(data.imu.time_s[index])
        dt_s = time_s - float(data.imu.time_s[index - 1])
        return TimedIMUIncrement(
            time_s,
            data.imu.delta_angle_rad[index],
            data.imu.delta_velocity_mps[index],
            dt_s,
        )

    def make_gnss(index: int) -> GNSSPositionMeasurement:
        return GNSSPositionMeasurement(
            time_s=float(data.rtk.time_s[index]),
            latitude_rad=np.deg2rad(data.rtk.latitude_deg[index]),
            longitude_rad=np.deg2rad(data.rtk.longitude_deg[index]),
            height_m=float(data.rtk.height_m[index]),
            std_ned_m=data.rtk.std_ned_m[index],
        )

    if update_mode == "delayed-replay":
        assert fusion is not None
        if eskf.state.time_s >= start_time - 1e-6:
            record_state()
        samples_since_record = 0
        while imu_index < data.imu.time_s.size:
            imu = make_imu(imu_index)
            if imu.time_s > end_time:
                break
            fusion.process_imu(imu)
            if integrity_manager is not None:
                integrity_manager.advance_time(eskf.state.time_s)
            while gnss_index < data.rtk.time_s.size:
                measurement = make_gnss(gnss_index)
                delivery_time = max(
                    measurement.time_s,
                    measurement.time_s + gnss_time_offset_s,
                )
                if delivery_time > eskf.state.time_s + 1e-9:
                    break
                if is_in_outage(measurement.time_s):
                    skipped_outage_gnss += 1
                    if integrity_manager is not None:
                        integrity_manager.mark_measurement_missing(
                            measurement.time_s
                        )
                    gnss_index += 1
                    continue
                adjusted, scale, state_before = prepare_gnss(measurement)
                delayed_result = fusion.process_gnss(
                    adjusted,
                    arrival_time_s=eskf.state.time_s,
                )
                delayed_results.append(delayed_result)
                record_gnss_result(
                    measurement,
                    scale,
                    state_before,
                    delayed_result.update,
                )
                gnss_index += 1
            if eskf.state.time_s >= start_time - 1e-6:
                samples_since_record += 1
                # 与原双子样主流程保持约 100 Hz 的解文件记录频率；
                # 滤波内部仍逐个处理并保存约 200 Hz 的原始 IMU 增量。
                if not history["time_s"] or samples_since_record >= 2:
                    record_state()
                    samples_since_record = 0
            imu_index += 1
    else:
        if eskf.state.time_s < start_time - 1e-6:
            eskf.predict_single_sample(make_imu(imu_index))
            imu_index += 1
            if eskf.state.time_s < start_time - 1e-6:
                raise ValueError("one boundary IMU sample did not reach the evaluation start")
        record_state()
        while imu_index + 1 < data.imu.time_s.size:
            imu1 = make_imu(imu_index)
            imu2 = make_imu(imu_index + 1)
            if imu2.time_s > end_time:
                break
            eskf.predict_two_sample(imu1, imu2)
            if integrity_manager is not None:
                integrity_manager.advance_time(eskf.state.time_s)
            while (
                gnss_index < data.rtk.time_s.size
                and data.rtk.time_s[gnss_index] <= eskf.state.time_s
            ):
                measurement = make_gnss(gnss_index)
                if is_in_outage(measurement.time_s):
                    skipped_outage_gnss += 1
                    if integrity_manager is not None:
                        integrity_manager.mark_measurement_missing(
                            measurement.time_s
                        )
                elif (
                    abs(eskf.state.time_s - measurement.time_s)
                    <= config.max_gnss_time_error_s
                ):
                    adjusted, scale, state_before = prepare_gnss(measurement)
                    update = eskf.update_gnss_position(adjusted)
                    record_gnss_result(
                        measurement, scale, state_before, update
                    )
                else:
                    skipped_gnss += 1
                gnss_index += 1
            record_state()
            imu_index += 2

    replay_counts = np.asarray(
        [result.replayed_imu_samples for result in delayed_results],
        dtype=float,
    )
    timing_metadata = {
        "update_mode": update_mode,
        "time_offset_convention": "effective_time = reported_time + offset",
        "gnss_time_offset_s": gnss_time_offset_s,
        "fixed_lag_s": fixed_lag_s if update_mode == "delayed-replay" else None,
        "delayed_update_count": len(delayed_results),
        "mean_replayed_imu_samples": (
            float(np.mean(replay_counts)) if replay_counts.size else 0.0
        ),
        "max_replayed_imu_samples": (
            int(np.max(replay_counts)) if replay_counts.size else 0
        ),
    }
    integrity_metadata = {
        "mode": integrity_mode,
        "outage_intervals_elapsed_s": [list(item) for item in outage_intervals],
        "skipped_outage_gnss": skipped_outage_gnss,
    }
    if integrity_manager is not None:
        integrity_metadata.update(
            {
                "config": {
                    "outage_timeout_s": integrity_manager.config.outage_timeout_s,
                    "recovery_required_accepts": (
                        integrity_manager.config.recovery_required_accepts
                    ),
                    "recovery_initial_std_scale": (
                        integrity_manager.config.recovery_initial_std_scale
                    ),
                    "recovery_scale_decay": (
                        integrity_manager.config.recovery_scale_decay
                    ),
                },
                **integrity_manager.summary(),
            }
        )

    _write_outputs(
        args.output_dir,
        history,
        data,
        eskf,
        skipped_gnss,
        args.imu_profile,
        initialization_metadata,
        timing_metadata,
        integrity_metadata,
    )
    if integrity_manager is not None:
        event_rows = [
            [
                event.time_s,
                event.state_before,
                event.state_after,
                event.event,
                "" if event.accepted is None else int(event.accepted),
                "" if event.nis is None else event.nis,
                event.measurement_std_scale,
            ]
            for event in integrity_manager.events
        ]
        event_path = args.output_dir / "gnss_integrity_events.csv"
        with event_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "time_s",
                    "state_before",
                    "state_after",
                    "event",
                    "accepted",
                    "nis",
                    "measurement_std_scale",
                ]
            )
            writer.writerows(event_rows)
    return {
        "eskf": eskf,
        "history": history,
        "skipped_gnss": skipped_gnss,
        "initialization": initialization_metadata,
        "gnss_timing": timing_metadata,
        "gnss_integrity": integrity_metadata,
    }


def main() -> None:
    args = parse_args()
    result = run_replay(args)
    eskf = result["eskf"]
    print(f"Wrote ESKF replay artifacts to {args.output_dir}")
    print(f"Accepted GNSS updates: {eskf.accepted_gnss_updates}")
    print(f"Rejected GNSS updates: {eskf.rejected_gnss_updates}")
    print(f"Skipped unsynchronized GNSS: {result['skipped_gnss']}")


if __name__ == "__main__":
    main()
