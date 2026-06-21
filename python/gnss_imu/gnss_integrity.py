"""GNSS outage detection, cautious recovery, and integrity state tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from gnss_imu.loosely_coupled_eskf import GNSSPositionMeasurement


class GNSSIntegrityState(str, Enum):
    TRACKING = "tracking"
    OUTAGE = "outage"
    RECOVERY = "recovery"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class GNSSIntegrityConfig:
    outage_timeout_s: float = 2.0
    recovery_required_accepts: int = 3
    recovery_initial_std_scale: float = 10.0
    recovery_scale_decay: float = 0.5
    tracking_rejections_to_degraded: int = 3
    recovery_rejections_to_degraded: int = 3

    def __post_init__(self) -> None:
        for name in (
            "outage_timeout_s",
            "recovery_initial_std_scale",
            "recovery_scale_decay",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.recovery_initial_std_scale < 1.0:
            raise ValueError("recovery_initial_std_scale must be at least one")
        if self.recovery_scale_decay > 1.0:
            raise ValueError("recovery_scale_decay must not exceed one")
        for name in (
            "recovery_required_accepts",
            "tracking_rejections_to_degraded",
            "recovery_rejections_to_degraded",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least one")


@dataclass(frozen=True)
class GNSSIntegrityEvent:
    time_s: float
    state_before: str
    state_after: str
    event: str
    accepted: bool | None
    nis: float | None
    measurement_std_scale: float


class GNSSIntegrityManager:
    """Stateful policy around GNSS updates; it does not own ESKF mathematics."""

    def __init__(
        self,
        config: GNSSIntegrityConfig | None = None,
        *,
        initial_measurement_time_s: float | None = None,
    ) -> None:
        self.config = GNSSIntegrityConfig() if config is None else config
        self.state = GNSSIntegrityState.TRACKING
        self.last_measurement_time_s = (
            None
            if initial_measurement_time_s is None
            else self._finite_time(initial_measurement_time_s)
        )
        self.last_accepted_time_s: float | None = None
        self.consecutive_accepts = 0
        self.consecutive_rejections = 0
        self.events: list[GNSSIntegrityEvent] = []

    def advance_time(self, time_s: float) -> GNSSIntegrityEvent | None:
        """Detect a GNSS outage from elapsed time since the last measurement."""
        current_time = self._finite_time(time_s)
        if self.last_measurement_time_s is None:
            return None
        if current_time + 1e-9 < self.last_measurement_time_s:
            raise ValueError("integrity time cannot move backwards")
        if (
            self.state != GNSSIntegrityState.OUTAGE
            and current_time - self.last_measurement_time_s
            >= self.config.outage_timeout_s
        ):
            return self._transition(
                current_time,
                GNSSIntegrityState.OUTAGE,
                "outage-detected",
                accepted=None,
                nis=None,
                std_scale=1.0,
            )
        return None

    def mark_measurement_missing(self, measurement_time_s: float) -> None:
        """Record an intentionally dropped or unavailable GNSS epoch."""
        self.advance_time(measurement_time_s)

    def prepare_measurement(
        self,
        measurement: GNSSPositionMeasurement,
    ) -> tuple[GNSSPositionMeasurement, float, GNSSIntegrityState]:
        """Apply recovery covariance inflation before the ESKF update."""
        time_s = self._finite_time(measurement.time_s)
        self.advance_time(time_s)
        if self.state == GNSSIntegrityState.OUTAGE:
            self._transition(
                time_s,
                GNSSIntegrityState.RECOVERY,
                "measurement-reacquired",
                accepted=None,
                nis=None,
                std_scale=self.config.recovery_initial_std_scale,
            )
            self.consecutive_accepts = 0
            self.consecutive_rejections = 0
        elif self.state == GNSSIntegrityState.DEGRADED:
            self.state = GNSSIntegrityState.RECOVERY
            self.consecutive_accepts = 0

        state_before_update = self.state
        scale = self.measurement_std_scale
        adjusted = GNSSPositionMeasurement(
            measurement.time_s,
            measurement.latitude_rad,
            measurement.longitude_rad,
            measurement.height_m,
            measurement.std_ned_m * scale,
        )
        self.last_measurement_time_s = time_s
        return adjusted, scale, state_before_update

    def record_update(
        self,
        time_s: float,
        *,
        accepted: bool,
        nis: float,
        measurement_std_scale: float,
        state_before_update: GNSSIntegrityState,
    ) -> GNSSIntegrityEvent:
        """Advance integrity state after observing the ESKF update result."""
        update_time = self._finite_time(time_s)
        if not np.isfinite(nis) or nis < 0.0:
            raise ValueError("nis must be finite and nonnegative")
        if not np.isfinite(measurement_std_scale) or measurement_std_scale < 1.0:
            raise ValueError("measurement_std_scale must be finite and at least one")
        self.last_measurement_time_s = update_time
        event_name = "update-accepted" if accepted else "update-rejected"
        if accepted:
            self.last_accepted_time_s = update_time
            self.consecutive_accepts += 1
            self.consecutive_rejections = 0
            if (
                self.state == GNSSIntegrityState.RECOVERY
                and self.consecutive_accepts >= self.config.recovery_required_accepts
            ):
                self.state = GNSSIntegrityState.TRACKING
                event_name = "recovery-complete"
                self.consecutive_accepts = 0
        else:
            self.consecutive_accepts = 0
            self.consecutive_rejections += 1
            rejection_limit = (
                self.config.tracking_rejections_to_degraded
                if self.state == GNSSIntegrityState.TRACKING
                else self.config.recovery_rejections_to_degraded
            )
            if self.consecutive_rejections >= rejection_limit:
                self.state = GNSSIntegrityState.DEGRADED
                event_name = "integrity-degraded"
        event = GNSSIntegrityEvent(
            time_s=update_time,
            state_before=state_before_update.value,
            state_after=self.state.value,
            event=event_name,
            accepted=bool(accepted),
            nis=float(nis),
            measurement_std_scale=float(measurement_std_scale),
        )
        self.events.append(event)
        return event

    @property
    def measurement_std_scale(self) -> float:
        if self.state != GNSSIntegrityState.RECOVERY:
            return 1.0
        return max(
            1.0,
            self.config.recovery_initial_std_scale
            * self.config.recovery_scale_decay**self.consecutive_accepts,
        )

    def summary(self) -> dict[str, object]:
        return {
            "final_state": self.state.value,
            "event_count": len(self.events),
            "outage_detection_count": sum(
                event.event == "outage-detected" for event in self.events
            ),
            "reacquisition_count": sum(
                event.event == "measurement-reacquired" for event in self.events
            ),
            "recovery_completion_count": sum(
                event.event == "recovery-complete" for event in self.events
            ),
            "degraded_count": sum(
                event.event == "integrity-degraded" for event in self.events
            ),
        }

    def _transition(
        self,
        time_s: float,
        new_state: GNSSIntegrityState,
        event_name: str,
        *,
        accepted: bool | None,
        nis: float | None,
        std_scale: float,
    ) -> GNSSIntegrityEvent:
        old_state = self.state
        self.state = new_state
        event = GNSSIntegrityEvent(
            time_s=float(time_s),
            state_before=old_state.value,
            state_after=new_state.value,
            event=event_name,
            accepted=accepted,
            nis=nis,
            measurement_std_scale=float(std_scale),
        )
        self.events.append(event)
        return event

    @staticmethod
    def _finite_time(value: float) -> float:
        time_s = float(value)
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        return time_s
