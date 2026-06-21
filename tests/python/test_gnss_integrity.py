"""Deterministic tests for GNSS outage and recovery policy."""

from __future__ import annotations

import numpy as np
import pytest

from gnss_imu import (
    GNSSIntegrityConfig,
    GNSSIntegrityManager,
    GNSSIntegrityState,
    GNSSPositionMeasurement,
)


def measurement(time_s: float) -> GNSSPositionMeasurement:
    return GNSSPositionMeasurement(
        time_s,
        np.deg2rad(30.0),
        np.deg2rad(114.0),
        20.0,
        np.ones(3),
    )


def accept(manager: GNSSIntegrityManager, time_s: float) -> float:
    _, scale, state_before = manager.prepare_measurement(measurement(time_s))
    manager.record_update(
        time_s,
        accepted=True,
        nis=1.0,
        measurement_std_scale=scale,
        state_before_update=state_before,
    )
    return scale


def test_outage_recovery_requires_consecutive_cautious_updates() -> None:
    manager = GNSSIntegrityManager(
        GNSSIntegrityConfig(
            outage_timeout_s=2.0,
            recovery_required_accepts=3,
            recovery_initial_std_scale=10.0,
            recovery_scale_decay=0.5,
        ),
        initial_measurement_time_s=0.0,
    )

    outage = manager.advance_time(2.0)
    assert outage is not None
    assert manager.state == GNSSIntegrityState.OUTAGE
    assert accept(manager, 3.0) == pytest.approx(10.0)
    assert manager.state == GNSSIntegrityState.RECOVERY
    assert accept(manager, 4.0) == pytest.approx(5.0)
    assert manager.state == GNSSIntegrityState.RECOVERY
    assert accept(manager, 5.0) == pytest.approx(2.5)
    assert manager.state == GNSSIntegrityState.TRACKING
    assert manager.measurement_std_scale == pytest.approx(1.0)
    assert manager.summary()["recovery_completion_count"] == 1


def test_consecutive_rejections_enter_degraded_state() -> None:
    manager = GNSSIntegrityManager(
        GNSSIntegrityConfig(tracking_rejections_to_degraded=3),
        initial_measurement_time_s=0.0,
    )

    for time_s in (1.0, 2.0, 3.0):
        _, scale, state_before = manager.prepare_measurement(measurement(time_s))
        manager.record_update(
            time_s,
            accepted=False,
            nis=100.0,
            measurement_std_scale=scale,
            state_before_update=state_before,
        )

    assert manager.state == GNSSIntegrityState.DEGRADED
    assert manager.summary()["degraded_count"] == 1
    adjusted, scale, state_before = manager.prepare_measurement(measurement(4.0))
    assert state_before == GNSSIntegrityState.RECOVERY
    assert scale == pytest.approx(10.0)
    np.testing.assert_allclose(adjusted.std_ned_m, 10.0)


def test_integrity_time_cannot_move_backwards() -> None:
    manager = GNSSIntegrityManager(initial_measurement_time_s=10.0)

    with pytest.raises(ValueError, match="cannot move backwards"):
        manager.advance_time(9.0)
