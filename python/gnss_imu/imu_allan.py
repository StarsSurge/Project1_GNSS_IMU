"""Allan-deviation utilities for uniformly sampled IMU rate data.

The module is intentionally independent of plotting. It validates timestamps,
loads CSV or Feather tables, converts increment logs to rate data, and exposes
both overlapping and non-overlapping Allan estimators for use by scripts and
tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

Array = np.ndarray


def _validated_signal(signal: Array, sample_rate_hz: float) -> Array:
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 3:
        raise ValueError("signal must contain at least three samples")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    return signal


def _cluster_sizes(
    taus: Array,
    sample_rate_hz: float,
    signal_size: int,
    overlapping: bool,
) -> Array:
    taus = np.asarray(taus, dtype=float).reshape(-1)
    if taus.size == 0 or not np.all(np.isfinite(taus)):
        raise ValueError("taus must contain finite positive values")
    if np.any(taus <= 0.0):
        raise ValueError("taus must contain finite positive values")

    sizes = np.unique(np.rint(taus * sample_rate_hz).astype(int))
    sizes = sizes[sizes >= 1]
    if overlapping:
        sizes = sizes[2 * sizes < signal_size]
    else:
        sizes = sizes[signal_size // sizes >= 3]
    return sizes


def allan_deviation(
    signal: Array,
    sample_rate_hz: float,
    taus: Array,
) -> tuple[Array, Array]:
    """Return non-overlapping Allan deviation of uniformly sampled rate data."""
    signal = _validated_signal(signal, sample_rate_hz)
    sizes = _cluster_sizes(
        taus, sample_rate_hz, signal.size, overlapping=False
    )
    deviations = []
    for cluster_size in sizes:
        cluster_count = signal.size // cluster_size
        means = signal[: cluster_count * cluster_size].reshape(
            cluster_count, cluster_size
        ).mean(axis=1)
        deviations.append(np.sqrt(0.5 * np.mean(np.diff(means) ** 2)))
    return sizes / sample_rate_hz, np.asarray(deviations)


def overlapping_allan_deviation(
    signal: Array,
    sample_rate_hz: float,
    taus: Array,
) -> tuple[Array, Array, Array]:
    """Return overlapping Allan deviation and raw difference-pair counts."""
    signal = _validated_signal(signal, sample_rate_hz)
    sizes = _cluster_sizes(
        taus, sample_rate_hz, signal.size, overlapping=True
    )
    cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
    deviations = []
    pair_counts = []

    for cluster_size in sizes:
        moving_means = (
            cumulative[cluster_size:] - cumulative[:-cluster_size]
        ) / cluster_size
        differences = (
            moving_means[cluster_size:] - moving_means[:-cluster_size]
        )
        deviations.append(np.sqrt(0.5 * np.mean(differences**2)))
        pair_counts.append(differences.size)

    return (
        sizes / sample_rate_hz,
        np.asarray(deviations),
        np.asarray(pair_counts, dtype=int),
    )


def sampling_interval_statistics(timestamps: Array) -> dict[str, float]:
    """Summarize timestamp regularity before an Allan analysis."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if timestamps.size < 3:
        raise ValueError("timestamps must contain at least three samples")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps must contain only finite values")

    intervals = np.diff(timestamps)
    if np.any(intervals <= 0.0):
        raise ValueError("timestamps must be strictly increasing")

    median_dt = float(np.median(intervals))
    return {
        "median_dt": median_dt,
        "sample_rate_hz": 1.0 / median_dt,
        "min_dt": float(np.min(intervals)),
        "max_dt": float(np.max(intervals)),
        "relative_jitter_rms": float(
            np.sqrt(np.mean((intervals - median_dt) ** 2)) / median_dt
        ),
        "max_relative_gap": float(np.max(intervals) / median_dt),
    }


def validate_uniform_sampling(
    timestamps: Array,
    *,
    max_relative_jitter_rms: float = 0.02,
    max_relative_gap: float = 1.5,
) -> dict[str, float]:
    """Validate the equal-sampling assumption required by this implementation."""
    if max_relative_jitter_rms < 0.0 or max_relative_gap < 1.0:
        raise ValueError("invalid timestamp-quality thresholds")
    statistics = sampling_interval_statistics(timestamps)
    if statistics["relative_jitter_rms"] > max_relative_jitter_rms:
        raise ValueError(
            "timestamp jitter exceeds the configured limit: "
            f"{statistics['relative_jitter_rms']:.6g} > "
            f"{max_relative_jitter_rms:.6g}"
        )
    if statistics["max_relative_gap"] > max_relative_gap:
        raise ValueError(
            "timestamp gap exceeds the configured limit: "
            f"{statistics['max_relative_gap']:.6g} > "
            f"{max_relative_gap:.6g}"
        )
    return statistics


def _relative_seconds(raw_timestamps: Array, scale_to_seconds: float) -> Array:
    """Convert timestamps to relative seconds without losing epoch precision."""
    raw = np.asarray(raw_timestamps).reshape(-1)
    if raw.size < 3:
        raise ValueError("timestamps must contain at least three samples")
    if scale_to_seconds <= 0.0 or not np.isfinite(scale_to_seconds):
        raise ValueError("timestamp_scale_to_seconds must be positive")

    if np.issubdtype(raw.dtype, np.integer):
        relative = raw - raw[0]
        timestamps = relative.astype(float) * scale_to_seconds
    else:
        raw = raw.astype(float)
        if not np.all(np.isfinite(raw)):
            raise ValueError("timestamps must contain only finite values")
        timestamps = (raw - raw[0]) * scale_to_seconds
    return timestamps


def _finalize_rate_data(
    raw_timestamps: Array,
    values: Array,
    *,
    timestamp_scale_to_seconds: float,
    value_scale_to_si: float,
    input_kind: str,
    max_relative_jitter_rms: float,
    max_relative_gap: float,
) -> tuple[Array, Array, dict[str, float]]:
    if input_kind not in {"rate", "increment"}:
        raise ValueError("input_kind must be 'rate' or 'increment'")
    if value_scale_to_si <= 0.0 or not np.isfinite(value_scale_to_si):
        raise ValueError("value_scale_to_si must be positive")

    timestamps = _relative_seconds(
        raw_timestamps, timestamp_scale_to_seconds
    )
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if timestamps.size != values.shape[0]:
        raise ValueError("timestamp and value columns have different lengths")
    if not np.all(np.isfinite(values)):
        raise ValueError("IMU value columns contain NaN or infinity")
    values = values * value_scale_to_si

    statistics = validate_uniform_sampling(
        timestamps,
        max_relative_jitter_rms=max_relative_jitter_rms,
        max_relative_gap=max_relative_gap,
    )
    if input_kind == "increment":
        intervals = np.diff(timestamps)
        values = values[1:] / intervals[:, None]
        timestamps = timestamps[1:]
        statistics = sampling_interval_statistics(timestamps)

    return timestamps, values, statistics


def load_imu_rate_csv(
    path: str | Path,
    *,
    timestamp_column: str,
    value_columns: tuple[str, ...],
    timestamp_scale_to_seconds: float = 1.0,
    value_scale_to_si: float = 1.0,
    input_kind: str = "rate",
    max_relative_jitter_rms: float = 0.02,
    max_relative_gap: float = 1.5,
) -> tuple[Array, Array, dict[str, float]]:
    """Load named CSV columns and return timestamps and SI rate data.

    ``input_kind`` is ``"rate"`` for angular-rate/specific-force logs and
    ``"increment"`` for delta-angle/delta-velocity logs. Increment sample i is
    interpreted as accumulated over ``timestamp[i-1]`` to ``timestamp[i]``;
    therefore the first row is dropped during conversion.
    """
    if not value_columns:
        raise ValueError("value_columns must not be empty")

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    table = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8-sig",
    )
    if table.dtype.names is None:
        raise ValueError("CSV must contain a header row")
    missing = [
        name
        for name in (timestamp_column, *value_columns)
        if name not in table.dtype.names
    ]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")

    raw_timestamps = np.atleast_1d(table[timestamp_column])
    values = np.column_stack(
        [np.atleast_1d(table[name]).astype(float) for name in value_columns]
    )
    return _finalize_rate_data(
        raw_timestamps,
        values,
        timestamp_scale_to_seconds=timestamp_scale_to_seconds,
        value_scale_to_si=value_scale_to_si,
        input_kind=input_kind,
        max_relative_jitter_rms=max_relative_jitter_rms,
        max_relative_gap=max_relative_gap,
    )


def load_imu_rate_feather(
    path: str | Path,
    *,
    timestamp_column: str,
    value_columns: tuple[str, ...],
    timestamp_scale_to_seconds: float = 1.0,
    value_scale_to_si: float = 1.0,
    input_kind: str = "rate",
    max_relative_jitter_rms: float = 0.02,
    max_relative_gap: float = 1.5,
) -> tuple[Array, Array, dict[str, float]]:
    """Load named Feather columns and return relative timestamps and SI rates."""
    if not value_columns:
        raise ValueError("value_columns must not be empty")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        import pyarrow.feather as feather
    except ImportError as error:
        raise ImportError(
            "Feather input requires pyarrow; install project requirements"
        ) from error

    requested_columns = [timestamp_column, *value_columns]
    try:
        table = feather.read_table(
            path, columns=requested_columns, memory_map=True
        )
    except Exception as error:
        message = str(error)
        if "Field named" in message or "does not exist" in message:
            raise ValueError(
                f"Feather is missing one or more columns: {requested_columns}"
            ) from error
        raise

    raw_timestamps = table[timestamp_column].combine_chunks().to_numpy()
    values = np.column_stack(
        [
            table[column].combine_chunks().to_numpy(zero_copy_only=False)
            for column in value_columns
        ]
    )
    return _finalize_rate_data(
        raw_timestamps,
        values,
        timestamp_scale_to_seconds=timestamp_scale_to_seconds,
        value_scale_to_si=value_scale_to_si,
        input_kind=input_kind,
        max_relative_jitter_rms=max_relative_jitter_rms,
        max_relative_gap=max_relative_gap,
    )


def load_imu_rate_table(
    path: str | Path,
    **kwargs: object,
) -> tuple[Array, Array, dict[str, float]]:
    """Load a CSV or Feather IMU table selected by its file suffix."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_imu_rate_csv(path, **kwargs)
    if suffix in {".feather", ".arrow"}:
        return load_imu_rate_feather(path, **kwargs)
    raise ValueError("input file must use .csv, .feather, or .arrow")


def fit_allan_log_slope(
    taus: Array,
    deviations: Array,
    tau_min: float,
    tau_max: float,
) -> dict[str, float]:
    """Fit log10(deviation) = slope * log10(tau) + intercept."""
    taus = np.asarray(taus, dtype=float).reshape(-1)
    deviations = np.asarray(deviations, dtype=float).reshape(-1)
    if taus.shape != deviations.shape:
        raise ValueError("taus and deviations must have the same shape")
    if tau_min <= 0.0 or tau_max <= tau_min:
        raise ValueError("tau bounds must satisfy 0 < tau_min < tau_max")

    mask = (
        (taus >= tau_min)
        & (taus <= tau_max)
        & np.isfinite(taus)
        & np.isfinite(deviations)
        & (deviations > 0.0)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("the fit interval must contain at least three points")

    slope, intercept = np.polyfit(
        np.log10(taus[mask]), np.log10(deviations[mask]), 1
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "value_at_tau_1": float(10.0**intercept),
        "point_count": int(np.count_nonzero(mask)),
    }


def extract_allan_parameters(
    taus: Array,
    deviations: Array,
    *,
    white_fit_range: tuple[float, float] | None = None,
    random_walk_fit_range: tuple[float, float] | None = None,
    slope_tolerance: float = 0.12,
    bias_instability_factor: float = 0.664,
) -> dict[str, object]:
    """Extract explicitly fitted Allan coefficients with validity flags.

    White-noise and random-walk parameters are only returned when their fit
    ranges are supplied. The bias-instability value is always a preliminary
    candidate based on the Allan minimum, not a deployment-ready calibration.
    """
    taus = np.asarray(taus, dtype=float).reshape(-1)
    deviations = np.asarray(deviations, dtype=float).reshape(-1)
    if taus.shape != deviations.shape or taus.size < 3:
        raise ValueError("taus and deviations must have the same usable shape")
    if slope_tolerance <= 0.0:
        raise ValueError("slope_tolerance must be positive")
    if bias_instability_factor <= 0.0:
        raise ValueError("bias_instability_factor must be positive")

    log_slopes = np.gradient(np.log10(deviations), np.log10(taus))
    minimum_index = int(np.argmin(deviations))
    edge_margin = max(1, int(np.ceil(0.1 * taus.size)))
    bias_candidate = {
        "tau_s": float(taus[minimum_index]),
        "allan_deviation": float(deviations[minimum_index]),
        "coefficient": float(
            deviations[minimum_index] / bias_instability_factor
        ),
        "local_slope": float(log_slopes[minimum_index]),
        "is_interior_minimum": bool(
            edge_margin <= minimum_index < taus.size - edge_margin
        ),
        "is_near_plateau": bool(
            abs(log_slopes[minimum_index]) <= slope_tolerance
        ),
        "status": "candidate_only",
    }

    result: dict[str, object] = {
        "bias_instability_candidate": bias_candidate,
        "white_noise": None,
        "rate_random_walk": None,
    }
    if white_fit_range is not None:
        fit = fit_allan_log_slope(
            taus, deviations, white_fit_range[0], white_fit_range[1]
        )
        coefficient = fit["value_at_tau_1"]
        result["white_noise"] = {
            **fit,
            "tau_min_s": float(white_fit_range[0]),
            "tau_max_s": float(white_fit_range[1]),
            "coefficient": coefficient,
            "continuous_psd": coefficient**2,
            "target_slope": -0.5,
            "is_valid": bool(
                abs(fit["slope"] + 0.5) <= slope_tolerance
            ),
        }

    if random_walk_fit_range is not None:
        fit = fit_allan_log_slope(
            taus,
            deviations,
            random_walk_fit_range[0],
            random_walk_fit_range[1],
        )
        coefficient = fit["value_at_tau_1"]
        result["rate_random_walk"] = {
            **fit,
            "tau_min_s": float(random_walk_fit_range[0]),
            "tau_max_s": float(random_walk_fit_range[1]),
            "coefficient": coefficient,
            "bias_drive_psd": 3.0 * coefficient**2,
            "bias_drive_amplitude": np.sqrt(3.0) * coefficient,
            "target_slope": 0.5,
            "is_valid": bool(
                abs(fit["slope"] - 0.5) <= slope_tolerance
            ),
        }
    return result
