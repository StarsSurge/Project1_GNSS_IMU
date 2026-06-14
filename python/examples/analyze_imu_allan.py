"""Analyze real IMU CSV rate or increment columns with overlapping Allan deviation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnss_imu.imu_allan import (
    load_imu_rate_csv,
    overlapping_allan_deviation,
)

TIME_SCALES = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run overlapping Allan-deviation analysis on uniformly sampled "
            "IMU CSV columns. This tool rejects timestamp gaps by default."
        )
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--timestamp-column", required=True)
    parser.add_argument(
        "--value-columns",
        required=True,
        help="Comma-separated columns, for example gyro_x,gyro_y,gyro_z",
    )
    parser.add_argument(
        "--timestamp-unit", choices=TIME_SCALES, default="s"
    )
    parser.add_argument(
        "--input-kind", choices=("rate", "increment"), default="rate"
    )
    parser.add_argument(
        "--value-scale-to-si",
        type=float,
        default=1.0,
        help="Multiply input values by this factor before analysis.",
    )
    parser.add_argument("--tau-min", type=float)
    parser.add_argument("--tau-max", type=float)
    parser.add_argument("--tau-count", type=int, default=60)
    parser.add_argument("--max-relative-jitter-rms", type=float, default=0.02)
    parser.add_argument("--max-relative-gap", type=float, default=1.5)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/imu_allan")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = tuple(
        column.strip()
        for column in args.value_columns.split(",")
        if column.strip()
    )
    timestamps, rates, sampling = load_imu_rate_csv(
        args.csv_path,
        timestamp_column=args.timestamp_column,
        value_columns=columns,
        timestamp_scale_to_seconds=TIME_SCALES[args.timestamp_unit],
        value_scale_to_si=args.value_scale_to_si,
        input_kind=args.input_kind,
        max_relative_jitter_rms=args.max_relative_jitter_rms,
        max_relative_gap=args.max_relative_gap,
    )

    duration = float(timestamps[-1] - timestamps[0])
    tau_min = args.tau_min or sampling["median_dt"]
    tau_max = args.tau_max or duration / 10.0
    if tau_min <= 0.0 or tau_max <= tau_min:
        raise ValueError("tau range must satisfy 0 < tau_min < tau_max")
    if args.tau_count < 3:
        raise ValueError("tau-count must be at least three")

    requested_taus = np.logspace(
        np.log10(tau_min), np.log10(tau_max), args.tau_count
    )
    axis_results = []
    for axis, column in enumerate(columns):
        taus, deviations, pairs = overlapping_allan_deviation(
            rates[:, axis], sampling["sample_rate_hz"], requested_taus
        )
        axis_results.append((column, taus, deviations, pairs))

    common_taus = axis_results[0][1]
    if any(not np.array_equal(result[1], common_taus) for result in axis_results):
        raise RuntimeError("Allan tau grids differ unexpectedly between axes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = args.output_dir / "allan_deviation.csv"
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["tau_s", "difference_pairs"]
        header.extend(f"{column}_allan_deviation_si" for column in columns)
        writer.writerow(header)
        for index, tau in enumerate(common_taus):
            writer.writerow(
                [tau, axis_results[0][3][index]]
                + [result[2][index] for result in axis_results]
            )

    metadata = {
        "maturity": "verified MVP; not deployment-ready",
        "source_csv": str(args.csv_path),
        "timestamp_column": args.timestamp_column,
        "value_columns": columns,
        "timestamp_unit": args.timestamp_unit,
        "input_kind": args.input_kind,
        "value_scale_to_si": args.value_scale_to_si,
        "sample_count": int(rates.shape[0]),
        "duration_s": duration,
        "sampling": sampling,
        "tau_min_s": float(common_taus[0]),
        "tau_max_s": float(common_taus[-1]),
        "min_pairs_warning_threshold": args.min_pairs,
        "limitations": [
            "input must be stationary and uniformly sampled",
            "no automatic motion, saturation, temperature, or outlier rejection",
            "difference-pair count is not equivalent degrees of freedom",
            "noise coefficients require documented slope-region fitting",
        ],
    }
    metadata_output = args.output_dir / "analysis_metadata.json"
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, (ax, ax_pairs) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [3.5, 1.0]},
        constrained_layout=True,
    )
    for column, taus, deviations, _ in axis_results:
        ax.loglog(taus, deviations, label=column)
    ax.set_title("Overlapping Allan deviation of IMU rate data")
    ax.set_ylabel("Allan deviation [SI rate unit]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    pair_counts = axis_results[0][3]
    ax_pairs.loglog(common_taus, pair_counts, color="black")
    ax_pairs.axhline(
        args.min_pairs,
        color="tab:red",
        linestyle="--",
        label=f"warning threshold: {args.min_pairs} pairs",
    )
    ax_pairs.set_xlabel("Cluster time tau [s]")
    ax_pairs.set_ylabel("Difference pairs")
    ax_pairs.grid(True, which="both", alpha=0.3)
    ax_pairs.legend()
    figure_output = args.output_dir / "allan_deviation.png"
    fig.savefig(figure_output, dpi=160)
    plt.close(fig)

    print(f"Samples: {rates.shape[0]}, duration: {duration:.3f} s")
    print(
        "Sampling: "
        f"{sampling['sample_rate_hz']:.6f} Hz, "
        f"jitter RMS={sampling['relative_jitter_rms']:.3e}, "
        f"max gap ratio={sampling['max_relative_gap']:.3f}"
    )
    print(f"Saved: {csv_output}")
    print(f"Saved: {metadata_output}")
    print(f"Saved: {figure_output}")


if __name__ == "__main__":
    main()
