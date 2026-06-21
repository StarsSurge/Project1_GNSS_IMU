"""Fixed-lag delayed GNSS updates and constant time-offset calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gnss_imu.loosely_coupled_eskf import (
    ESKFConfig,
    ESKFState,
    GNSSPositionMeasurement,
    GNSSUpdateResult,
    LooselyCoupledESKF,
    TimedIMUIncrement,
)


def clone_eskf_state(state: ESKFState) -> ESKFState:
    """深复制 ESKF 状态，防止回放修改历史快照中的数组。"""
    return ESKFState(
        time_s=state.time_s,
        latitude_rad=state.latitude_rad,
        longitude_rad=state.longitude_rad,
        height_m=state.height_m,
        velocity_ned_mps=state.velocity_ned_mps.copy(),
        q_bn=state.q_bn.copy(),
        accel_bias_mps2=state.accel_bias_mps2.copy(),
        gyro_bias_rps=state.gyro_bias_rps.copy(),
        covariance=state.covariance.copy(),
    )


@dataclass(frozen=True)
class DelayedGNSSUpdateResult:
    update: GNSSUpdateResult
    reported_time_s: float
    effective_time_s: float
    arrival_time_s: float
    rewind_s: float
    replayed_imu_samples: int


@dataclass(frozen=True)
class TimeOffsetCandidateScore:
    offset_s: float
    robust_mean_nis: float
    median_nis: float
    measurement_count: int
    accepted_count: int
    rejected_count: int


@dataclass(frozen=True)
class TimeOffsetCalibrationResult:
    best_offset_s: float
    scores: tuple[TimeOffsetCandidateScore, ...]
    peak_speed_mps: float


@dataclass(frozen=True)
class _FilterSnapshot:
    state: ESKFState
    accepted_updates: int
    rejected_updates: int


@dataclass(frozen=True)
class _GNSSRecord:
    identifier: int
    measurement: GNSSPositionMeasurement
    effective_time_s: float
    arrival_time_s: float


class FixedLagGNSSFusion:
    """Fuse out-of-sequence GNSS by rewinding and replaying raw IMU increments.

    IMU increments are split proportionally when a GNSS effective timestamp
    falls inside an interval.  This is exact under the piecewise-constant-rate
    assumption used for that raw increment.
    """

    def __init__(
        self,
        eskf: LooselyCoupledESKF,
        *,
        lag_s: float = 2.0,
        gnss_time_offset_s: float = 0.0,
        time_tolerance_s: float = 1e-9,
    ) -> None:
        if not np.isfinite(lag_s) or lag_s <= 0.0:
            raise ValueError("lag_s must be positive and finite")
        if not np.isfinite(gnss_time_offset_s):
            raise ValueError("gnss_time_offset_s must be finite")
        if not np.isfinite(time_tolerance_s) or time_tolerance_s <= 0.0:
            raise ValueError("time_tolerance_s must be positive and finite")
        self.eskf = eskf
        self.lag_s = float(lag_s)
        self.gnss_time_offset_s = float(gnss_time_offset_s)
        self.time_tolerance_s = float(time_tolerance_s)
        self._anchor = self._snapshot_filter(eskf)
        self._imu_history: list[TimedIMUIncrement] = []
        self._gnss_history: list[_GNSSRecord] = []
        self._boundary_snapshots: list[_FilterSnapshot] = []
        self._next_identifier = 0

    def process_imu(self, imu: TimedIMUIncrement) -> ESKFState:
        """传播一个原始 IMU 增量，并将增量和边界状态保存在固定滞后窗口中。"""
        self.eskf.predict_single_sample(imu)
        self._imu_history.append(imu)
        self._boundary_snapshots.append(self._snapshot_filter(self.eskf))
        self._prune_history()
        return self.eskf.state

    def process_gnss(
        self,
        measurement: GNSSPositionMeasurement,
        *,
        arrival_time_s: float | None = None,
    ) -> DelayedGNSSUpdateResult:
        """插入可能延迟到达的 GNSS 观测，并重放 IMU 至当前时刻。"""
        arrival = self.eskf.state.time_s if arrival_time_s is None else float(arrival_time_s)
        if not np.isfinite(arrival):
            raise ValueError("arrival_time_s must be finite")
        if arrival + self.time_tolerance_s < measurement.time_s:
            raise ValueError("GNSS arrival time cannot precede its reported timestamp")
        current_time = self.eskf.state.time_s
        if arrival > current_time + self.time_tolerance_s:
            raise ValueError("GNSS arrival time cannot exceed the current filter time")
        # 本仓库的符号约定：有效时刻 = 报告时刻 + 常量时间偏差。
        effective_time = measurement.time_s + self.gnss_time_offset_s
        if effective_time > current_time + self.time_tolerance_s:
            raise ValueError("GNSS effective timestamp is in the filter future")
        if effective_time < self._anchor.state.time_s - self.time_tolerance_s:
            raise ValueError("GNSS measurement is older than the fixed-lag history")

        identifier = self._next_identifier
        self._next_identifier += 1
        record = _GNSSRecord(identifier, measurement, effective_time, arrival)
        self._gnss_history.append(record)
        self._gnss_history.sort(key=lambda item: (item.effective_time_s, item.identifier))
        results = self._replay_history()
        update = results[identifier]
        replayed_count = sum(
            imu.time_s > effective_time + self.time_tolerance_s
            for imu in self._imu_history
        )
        return DelayedGNSSUpdateResult(
            update=update,
            reported_time_s=measurement.time_s,
            effective_time_s=effective_time,
            arrival_time_s=arrival,
            rewind_s=max(0.0, current_time - effective_time),
            replayed_imu_samples=int(replayed_count),
        )

    def _replay_history(self) -> dict[int, GNSSUpdateResult]:
        replay = LooselyCoupledESKF(
            clone_eskf_state(self._anchor.state),
            self.eskf.config,
        )
        replay.accepted_gnss_updates = self._anchor.accepted_updates
        replay.rejected_gnss_updates = self._anchor.rejected_updates
        records = list(self._gnss_history)
        record_index = 0
        results: dict[int, GNSSUpdateResult] = {}
        snapshots: list[_FilterSnapshot] = []

        def apply_record(record: _GNSSRecord) -> None:
            adjusted = GNSSPositionMeasurement(
                time_s=record.effective_time_s,
                latitude_rad=record.measurement.latitude_rad,
                longitude_rad=record.measurement.longitude_rad,
                height_m=record.measurement.height_m,
                std_ned_m=record.measurement.std_ned_m,
            )
            results[record.identifier] = replay.update_gnss_position(adjusted)

        for imu in self._imu_history:
            interval_start = imu.time_s - imu.dt_s
            interval_end = imu.time_s
            while (
                record_index < len(records)
                and records[record_index].effective_time_s
                <= interval_start + self.time_tolerance_s
            ):
                apply_record(records[record_index])
                record_index += 1

            segment_start = interval_start
            while (
                record_index < len(records)
                and records[record_index].effective_time_s
                < interval_end - self.time_tolerance_s
            ):
                event_time = records[record_index].effective_time_s
                segment_dt = event_time - segment_start
                if segment_dt > self.time_tolerance_s:
                    # 分段常值假设下按时间比例切分增量：
                    # alpha = segment_dt / imu.dt_s
                    # dtheta_segment = alpha * dtheta, dvel_segment = alpha * dvel
                    # alpha 无量纲，所以两个分段量的单位仍分别为 rad 和 m/s。
                    ratio = segment_dt / imu.dt_s
                    replay.predict_single_sample(
                        TimedIMUIncrement(
                            event_time,
                            imu.dtheta_rad * ratio,
                            imu.dvel_mps * ratio,
                            segment_dt,
                        )
                    )
                apply_record(records[record_index])
                segment_start = event_time
                record_index += 1

            remaining_dt = interval_end - segment_start
            if remaining_dt > self.time_tolerance_s:
                # GNSS 更新后必须用原始增量的剩余部分重新传播，不能沿用旧轨迹。
                ratio = remaining_dt / imu.dt_s
                replay.predict_single_sample(
                    TimedIMUIncrement(
                        interval_end,
                        imu.dtheta_rad * ratio,
                        imu.dvel_mps * ratio,
                        remaining_dt,
                    )
                )
            while (
                record_index < len(records)
                and abs(records[record_index].effective_time_s - interval_end)
                <= self.time_tolerance_s
            ):
                apply_record(records[record_index])
                record_index += 1
            snapshots.append(self._snapshot_filter(replay))

        while record_index < len(records):
            if abs(records[record_index].effective_time_s - replay.state.time_s) > self.time_tolerance_s:
                raise RuntimeError("GNSS event is not covered by retained IMU history")
            apply_record(records[record_index])
            record_index += 1

        self.eskf.state = clone_eskf_state(replay.state)
        self.eskf.accepted_gnss_updates = replay.accepted_gnss_updates
        self.eskf.rejected_gnss_updates = replay.rejected_gnss_updates
        self.eskf.last_specific_force_b_mps2 = replay.last_specific_force_b_mps2.copy()
        self.eskf.last_angular_rate_b_rps = replay.last_angular_rate_b_rps.copy()
        self._boundary_snapshots = snapshots
        return results

    def _prune_history(self) -> None:
        cutoff = self.eskf.state.time_s - self.lag_s
        prune_count = 0
        for imu in self._imu_history:
            if imu.time_s <= cutoff + self.time_tolerance_s:
                prune_count += 1
            else:
                break
        if prune_count == 0:
            return
        self._anchor = self._boundary_snapshots[prune_count - 1]
        self._imu_history = self._imu_history[prune_count:]
        self._boundary_snapshots = self._boundary_snapshots[prune_count:]
        self._gnss_history = [
            record
            for record in self._gnss_history
            if record.effective_time_s > self._anchor.state.time_s + self.time_tolerance_s
        ]

    @staticmethod
    def _snapshot_filter(eskf: LooselyCoupledESKF) -> _FilterSnapshot:
        return _FilterSnapshot(
            clone_eskf_state(eskf.state),
            eskf.accepted_gnss_updates,
            eskf.rejected_gnss_updates,
        )


def calibrate_constant_gnss_time_offset(
    initial_state: ESKFState,
    eskf_config: ESKFConfig,
    imu_samples: Sequence[TimedIMUIncrement],
    gnss_measurements: Sequence[GNSSPositionMeasurement],
    candidate_offsets_s: Sequence[float],
    *,
    lag_s: float = 2.0,
    min_peak_speed_mps: float = 2.0,
) -> TimeOffsetCalibrationResult:
    """Select a constant GNSS timestamp offset by robust innovation scoring.

    Offset convention: ``effective_time = reported_time + offset``.
    Calibration is rejected without enough motion because time offset is not
    observable from a static position sequence.
    """
    candidates = np.asarray(candidate_offsets_s, dtype=float)
    if candidates.ndim != 1 or candidates.size < 2 or not np.all(np.isfinite(candidates)):
        raise ValueError("candidate_offsets_s must contain at least two finite values")
    if np.unique(candidates).size != candidates.size:
        raise ValueError("candidate_offsets_s must not contain duplicates")
    if not imu_samples or not gnss_measurements:
        raise ValueError("time-offset calibration requires IMU and GNSS data")
    if not np.isfinite(min_peak_speed_mps) or min_peak_speed_mps < 0.0:
        raise ValueError("min_peak_speed_mps must be finite and nonnegative")

    # 可观性检查只看纯 IMU 参考轨迹，不能让错误候选的 GNSS 更新“制造”运动。
    excitation_filter = LooselyCoupledESKF(
        clone_eskf_state(initial_state),
        eskf_config,
    )
    peak_speed_all = float(np.linalg.norm(initial_state.velocity_ned_mps))
    for imu in imu_samples:
        excitation_filter.predict_single_sample(imu)
        peak_speed_all = max(
            peak_speed_all,
            float(np.linalg.norm(excitation_filter.state.velocity_ned_mps)),
        )
    if peak_speed_all < min_peak_speed_mps:
        raise ValueError(
            "GNSS time offset is not observable: motion excitation is insufficient"
        )

    scores: list[TimeOffsetCandidateScore] = []
    for offset in candidates:
        fusion = FixedLagGNSSFusion(
            LooselyCoupledESKF(clone_eskf_state(initial_state), eskf_config),
            lag_s=max(lag_s, abs(float(offset)) + 0.1),
            gnss_time_offset_s=float(offset),
        )
        measurement_index = 0
        nis_values: list[float] = []
        for imu in imu_samples:
            fusion.process_imu(imu)
            while measurement_index < len(gnss_measurements):
                measurement = gnss_measurements[measurement_index]
                effective_time = measurement.time_s + float(offset)
                delivery_time = max(measurement.time_s, effective_time)
                if delivery_time > fusion.eskf.state.time_s + fusion.time_tolerance_s:
                    break
                delayed = fusion.process_gnss(
                    measurement,
                    arrival_time_s=fusion.eskf.state.time_s,
                )
                nis_values.append(delayed.update.nis)
                measurement_index += 1
        if len(nis_values) < 3:
            raise ValueError("each time-offset candidate needs at least three GNSS updates")
        nis_array = np.asarray(nis_values)
        # score(tau) = mean(min(NIS_k(tau), 100))。
        # 截断只降低少量跳点的支配作用，不替代 GNSS 异常检测。
        scores.append(
            TimeOffsetCandidateScore(
                offset_s=float(offset),
                robust_mean_nis=float(np.mean(np.minimum(nis_array, 100.0))),
                median_nis=float(np.median(nis_array)),
                measurement_count=int(nis_array.size),
                accepted_count=fusion.eskf.accepted_gnss_updates,
                rejected_count=fusion.eskf.rejected_gnss_updates,
            )
        )
    best = min(scores, key=lambda item: (item.robust_mean_nis, item.median_nis))
    return TimeOffsetCalibrationResult(
        best_offset_s=best.offset_s,
        scores=tuple(scores),
        peak_speed_mps=peak_speed_all,
    )
