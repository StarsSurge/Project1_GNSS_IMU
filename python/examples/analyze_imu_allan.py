"""Analyze real IMU CSV or Feather data with overlapping Allan deviation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from gnss_imu.imu_allan import (
    extract_allan_parameters,
    fit_allan_log_slope,
    load_imu_rate_table,
    overlapping_allan_deviation,
)

TIME_SCALES = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}
DIAGNOSTIC_BANDS = (
    (0.01, 0.1),
    (0.1, 1.0),
    (1.0, 10.0),
    (10.0, 100.0),
    (100.0, 500.0),
)


def configure_chinese_font() -> None:
    """Select an installed Chinese font for generated report figures."""
    candidates = (
        "Microsoft YaHei",
        "DengXian",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next(
        (candidate for candidate in candidates if candidate in installed),
        None,
    )
    if selected is not None:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run overlapping Allan-deviation analysis on uniformly sampled "
            "IMU CSV or Feather columns. Timestamp gaps are rejected."
        )
    )
    parser.add_argument("input_path", type=Path)
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
    parser.add_argument("--tau-count", type=int, default=80)
    parser.add_argument("--white-fit-min", type=float)
    parser.add_argument("--white-fit-max", type=float)
    parser.add_argument("--random-walk-fit-min", type=float)
    parser.add_argument("--random-walk-fit-max", type=float)
    parser.add_argument("--slope-tolerance", type=float, default=0.12)
    parser.add_argument("--max-relative-jitter-rms", type=float, default=0.02)
    parser.add_argument("--max-relative-gap", type=float, default=1.5)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/imu_allan")
    )
    return parser.parse_args()


def optional_range(
    minimum: float | None,
    maximum: float | None,
    name: str,
) -> tuple[float, float] | None:
    if minimum is None and maximum is None:
        return None
    if minimum is None or maximum is None:
        raise ValueError(f"{name} requires both minimum and maximum")
    if minimum <= 0.0 or maximum <= minimum:
        raise ValueError(f"{name} must satisfy 0 < min < max")
    return minimum, maximum


def infer_unit(column: str) -> str:
    match = re.search(r"\[([^\]]+)\]\s*$", column)
    return match.group(1) if match else "SI rate unit"


def coefficient_unit(rate_unit: str) -> str:
    if rate_unit == "rad/s":
        return "rad/sqrt(s)"
    if rate_unit in {"m/s^2", "m/s²"}:
        return "m/s/sqrt(s)"
    return f"({rate_unit})*sqrt(s)"


def common_white_value(
    coefficient: float, rate_unit: str
) -> tuple[float | None, str]:
    if rate_unit == "rad/s":
        return coefficient * 180.0 / np.pi * 60.0, "deg/sqrt(h)"
    if rate_unit in {"m/s^2", "m/s²"}:
        return coefficient * 60.0, "m/s/sqrt(h)"
    return None, ""


def common_bias_value(
    coefficient: float, rate_unit: str
) -> tuple[float | None, str]:
    if rate_unit == "rad/s":
        return coefficient * 180.0 / np.pi * 3600.0, "deg/h"
    if rate_unit in {"m/s^2", "m/s²"}:
        return coefficient / 9.80665 * 1000.0, "mg"
    return None, ""


def estimate_quantization_step(values: np.ndarray) -> float:
    unique_values = np.unique(values)
    if unique_values.size < 2:
        return 0.0
    positive_steps = np.diff(unique_values)
    positive_steps = positive_steps[positive_steps > 0.0]
    return float(np.median(positive_steps)) if positive_steps.size else 0.0


def diagnostic_slopes(
    taus: np.ndarray, deviations: np.ndarray
) -> list[dict[str, float]]:
    fits = []
    for lower, upper in DIAGNOSTIC_BANDS:
        try:
            fit = fit_allan_log_slope(
                taus, deviations, lower, upper
            )
        except ValueError:
            continue
        fits.append(
            {
                "tau_min_s": lower,
                "tau_max_s": upper,
                **fit,
            }
        )
    return fits


def axis_statistics(
    timestamps: np.ndarray, values: np.ndarray
) -> dict[str, float]:
    duration = float(timestamps[-1] - timestamps[0])
    window = min(600.0, max(duration * 0.1, 1.0))
    first_mask = timestamps <= timestamps[0] + window
    last_mask = timestamps >= timestamps[-1] - window
    first_mean = float(np.mean(values[first_mask]))
    last_mean = float(np.mean(values[last_mask]))
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "quantization_step_estimate": estimate_quantization_step(values),
        "drift_window_s": window,
        "first_window_mean": first_mean,
        "last_window_mean": last_mean,
        "first_to_last_window_drift": last_mean - first_mean,
    }


def write_parameter_csv(
    path: Path, axis_reports: list[dict[str, object]]
) -> None:
    fieldnames = [
        "axis",
        "rate_unit",
        "white_fit_tau_min_s",
        "white_fit_tau_max_s",
        "white_fit_slope",
        "white_fit_valid",
        "white_coefficient_si",
        "white_coefficient_common",
        "white_coefficient_common_unit",
        "white_continuous_psd_si",
        "bias_candidate_tau_s",
        "bias_candidate_local_slope",
        "bias_candidate_interior",
        "bias_candidate_quality",
        "bias_instability_candidate_si",
        "bias_instability_candidate_common",
        "bias_instability_candidate_common_unit",
        "random_walk_fitted",
        "random_walk_fit_slope",
        "bias_drive_amplitude_si",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for report in axis_reports:
            white = report["parameters"]["white_noise"]
            bias = report["parameters"]["bias_instability_candidate"]
            random_walk = report["parameters"]["rate_random_walk"]
            writer.writerow(
                {
                    "axis": report["column"],
                    "rate_unit": report["rate_unit"],
                    "white_fit_tau_min_s": (
                        white["tau_min_s"] if white else ""
                    ),
                    "white_fit_tau_max_s": (
                        white["tau_max_s"] if white else ""
                    ),
                    "white_fit_slope": white["slope"] if white else "",
                    "white_fit_valid": white["is_valid"] if white else "",
                    "white_coefficient_si": (
                        white["coefficient"] if white else ""
                    ),
                    "white_coefficient_common": report[
                        "white_common_value"
                    ],
                    "white_coefficient_common_unit": report[
                        "white_common_unit"
                    ],
                    "white_continuous_psd_si": (
                        white["continuous_psd"] if white else ""
                    ),
                    "bias_candidate_tau_s": bias["tau_s"],
                    "bias_candidate_local_slope": bias["local_slope"],
                    "bias_candidate_interior": bias[
                        "is_interior_minimum"
                    ],
                    "bias_candidate_quality": (
                        "preliminary"
                        if bias["is_interior_minimum"]
                        and bias["is_near_plateau"]
                        else "boundary_or_no_plateau"
                    ),
                    "bias_instability_candidate_si": bias["coefficient"],
                    "bias_instability_candidate_common": report[
                        "bias_common_value"
                    ],
                    "bias_instability_candidate_common_unit": report[
                        "bias_common_unit"
                    ],
                    "random_walk_fitted": random_walk is not None,
                    "random_walk_fit_slope": (
                        random_walk["slope"] if random_walk else ""
                    ),
                    "bias_drive_amplitude_si": (
                        random_walk["bias_drive_amplitude"]
                        if random_walk
                        else ""
                    ),
                }
            )


def markdown_number(value: object, precision: int = 6) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.{precision}g}"


def write_report(
    path: Path,
    *,
    input_path: Path,
    sample_count: int,
    duration: float,
    sampling: dict[str, float],
    white_fit_range: tuple[float, float] | None,
    random_walk_fit_range: tuple[float, float] | None,
    axis_reports: list[dict[str, object]],
) -> None:
    lines = [
        "# IMU Allan 方差参数报告",
        "",
        "## 1. 数据与分析条件",
        "",
        f"- 输入文件：`{input_path.as_posix()}`",
        f"- 样本数：`{sample_count}`",
        f"- 记录时长：`{duration:.3f} s`（约 `{duration / 3600:.3f} h`）",
        f"- 采样率：`{sampling['sample_rate_hz']:.9f} Hz`",
        f"- 中位采样周期：`{sampling['median_dt']:.12g} s`",
        f"- 相对时间抖动 RMS：`{sampling['relative_jitter_rms']:.3e}`",
        f"- 最大采样间隔比：`{sampling['max_relative_gap']:.6f}`",
        "- 估计方法：重叠 Allan deviation",
        "",
        "时间戳在转换为秒之前先减去首个整数时间戳，避免纳秒 Unix "
        "epoch 直接转浮点数造成间隔精度损失。",
        "",
        "## 2. 白噪声参数",
        "",
    ]
    if white_fit_range is None:
        lines.append("本次没有指定白噪声拟合区间，因此未输出白噪声参数。")
    else:
        lines.extend(
            [
                "拟合模型："
                r"\(\sigma_A(\tau)=C_{-1/2}\tau^{-1/2}\)，"
                r"\(q=C_{-1/2}^2\)。",
                "",
                f"指定拟合区间：`{white_fit_range[0]:g}～"
                f"{white_fit_range[1]:g} s`；斜率与 `-0.5` 的差不超过"
                "配置容差时标记为有效。",
                "",
                "| 轴 | 拟合斜率 | SI 系数 | 常用单位 | 有效 |",
                "|---|---:|---:|---:|:---:|",
            ]
        )
        for report in axis_reports:
            white = report["parameters"]["white_noise"]
            common = (
                f"{markdown_number(report['white_common_value'])} "
                f"{report['white_common_unit']}"
                if report["white_common_value"] is not None
                else "-"
            )
            lines.append(
                f"| `{report['column']}` | {white['slope']:+.4f} | "
                f"{white['coefficient']:.6g} "
                f"{report['white_coefficient_unit']} | {common} | "
                f"{'是' if white['is_valid'] else '否'} |"
            )

    lines.extend(
        [
            "",
            "## 3. Bias instability 候选值",
            "",
            "候选值按常见 IEEE 约定 "
            r"\(B=\sigma_{A,\min}/0.664\) 计算。它只是曲线最低点对应的"
            "初步读数；没有温度记录、重复批次和可信平台区拟合时，不能直接"
            "作为部署级标定参数或 ESKF 过程噪声。",
            "",
            "| 轴 | 最低点 tau | 局部斜率 | 候选值 SI | 常用单位 | 质量标记 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for report in axis_reports:
        bias = report["parameters"]["bias_instability_candidate"]
        common = (
            f"{markdown_number(report['bias_common_value'])} "
            f"{report['bias_common_unit']}"
            if report["bias_common_value"] is not None
            else "-"
        )
        lines.append(
            f"| `{report['column']}` | {bias['tau_s']:.6g} s | "
            f"{bias['local_slope']:+.4f} | {bias['coefficient']:.6g} "
            f"{report['rate_unit']} | {common} | "
            f"{'初步候选' if bias['is_interior_minimum'] and bias['is_near_plateau'] else '边界或非平台，不可靠'} |"
        )

    lines.extend(["", "## 4. Rate random walk / bias 驱动", ""])
    if random_walk_fit_range is None:
        lines.append(
            "本次未指定 `+1/2` 拟合区间，因此没有输出 bias 随机游走驱动"
            "参数。当前曲线应先确认存在持续的 `+1/2` 区域，再使用 "
            r"\(q_b=3C_{+1/2}^2\) 换算。"
        )
    else:
        lines.extend(
            [
                f"指定拟合区间：`{random_walk_fit_range[0]:g}～"
                f"{random_walk_fit_range[1]:g} s`。",
                "",
                "| 轴 | 拟合斜率 | 驱动幅度 SI | 有效 |",
                "|---|---:|---:|:---:|",
            ]
        )
        for report in axis_reports:
            random_walk = report["parameters"]["rate_random_walk"]
            lines.append(
                f"| `{report['column']}` | {random_walk['slope']:+.4f} | "
                f"{random_walk['bias_drive_amplitude']:.6g} "
                f"{report['rate_unit']}/sqrt(s) | "
                f"{'是' if random_walk['is_valid'] else '否'} |"
            )

    lines.extend(
        [
            "",
            "## 5. 数据稳定性提示",
            "",
            "| 轴 | 均值 | 标准差 | 量化步进估计 | 首末窗口均值变化 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for report in axis_reports:
        statistics = report["statistics"]
        lines.append(
            f"| `{report['column']}` | {statistics['mean']:.6g} | "
            f"{statistics['standard_deviation']:.6g} | "
            f"{statistics['quantization_step_estimate']:.6g} | "
            f"{statistics['first_to_last_window_drift']:+.6g} |"
        )

    valid_white_axes = [
        report["column"]
        for report in axis_reports
        if report["parameters"]["white_noise"]
        and report["parameters"]["white_noise"]["is_valid"]
    ]
    boundary_bias_axes = [
        report["column"]
        for report in axis_reports
        if not report["parameters"]["bias_instability_candidate"][
            "is_interior_minimum"
        ]
    ]
    drift_axes = [
        report["column"]
        for report in axis_reports
        if abs(
            report["statistics"]["first_to_last_window_drift"]
        )
        > 0.25 * report["statistics"]["standard_deviation"]
    ]
    lines.extend(
        [
            "",
            "## 6. 结论与边界",
            "",
            f"- 白噪声拟合通过的轴：`{', '.join(valid_white_axes)}`。",
            (
                "- 最低点位于分析区间长时边缘、bias 候选不可靠的轴："
                f"`{', '.join(boundary_bias_axes)}`。"
                if boundary_bias_axes
                else "- 所有最低点均位于内部，但仍需重复实验确认平台。"
            ),
            (
                "- 首末 10 分钟均值变化超过单样本标准差 25% 的轴："
                f"`{', '.join(drift_axes)}`；长时间端可能受到温漂或"
                "安装缓慢变化影响。"
                if drift_axes
                else "- 未发现明显的首末窗口均值变化。"
            ),
            "- 白噪声参数只在拟合斜率通过检查时采用。",
            "- 曲线最低点不是自动等价于真实 bias instability 平台。",
            "- 未观察到稳定 `+1/2` 区域时，不报告 rate random walk。",
            "- 当前文件没有温度、设备型号、带宽、预热和安装环境元数据；"
            "长时间端可能混入温漂、安装缓慢变化或环境扰动。",
            "- 差分对数量不等于等效自由度，当前报告没有给出严格置信区间。",
            "- 参数进入 ESKF 前仍需按连续 PSD、状态模型和离散化约定换算，"
            "并通过真实融合回放验证。",
            "",
            "配套文件：",
            "",
            "- `allan_deviation.png`：逐轴 Allan 曲线",
            "- `allan_parameter_summary.png`：六轴关键参数汇总图",
            "- `allan_difference_pairs.png`：重叠原始差分对数量与"
            "非重叠基线数量",
            "- `allan_deviation.csv`：完整曲线数据",
            "- `allan_parameters.csv`：参数表",
            "- `analysis_metadata.json`：机器可读分析条件与结果",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_chinese_font()
    args = parse_args()
    columns = tuple(
        column.strip()
        for column in args.value_columns.split(",")
        if column.strip()
    )
    white_fit_range = optional_range(
        args.white_fit_min, args.white_fit_max, "white fit range"
    )
    random_walk_fit_range = optional_range(
        args.random_walk_fit_min,
        args.random_walk_fit_max,
        "random-walk fit range",
    )
    timestamps, rates, sampling = load_imu_rate_table(
        args.input_path,
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
    axis_reports: list[dict[str, object]] = []
    for axis, column in enumerate(columns):
        taus, deviations, pairs = overlapping_allan_deviation(
            rates[:, axis], sampling["sample_rate_hz"], requested_taus
        )
        parameters = extract_allan_parameters(
            taus,
            deviations,
            white_fit_range=white_fit_range,
            random_walk_fit_range=random_walk_fit_range,
            slope_tolerance=args.slope_tolerance,
        )
        rate_unit = infer_unit(column)
        white = parameters["white_noise"]
        bias = parameters["bias_instability_candidate"]
        white_common_value, white_common_unit = (
            common_white_value(white["coefficient"], rate_unit)
            if white
            else (None, "")
        )
        bias_common_value, bias_common_unit = common_bias_value(
            bias["coefficient"], rate_unit
        )
        axis_results.append((column, rate_unit, taus, deviations, pairs))
        axis_reports.append(
            {
                "column": column,
                "rate_unit": rate_unit,
                "white_coefficient_unit": coefficient_unit(rate_unit),
                "white_common_value": white_common_value,
                "white_common_unit": white_common_unit,
                "bias_common_value": bias_common_value,
                "bias_common_unit": bias_common_unit,
                "statistics": axis_statistics(timestamps, rates[:, axis]),
                "parameters": parameters,
                "diagnostic_slopes": diagnostic_slopes(taus, deviations),
            }
        )

    common_taus = axis_results[0][2]
    if any(
        not np.array_equal(result[2], common_taus)
        for result in axis_results
    ):
        raise RuntimeError("Allan tau grids differ unexpectedly between axes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = args.output_dir / "allan_deviation.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = [
            "tau_s",
            "overlapping_difference_pairs",
            "nonoverlapping_difference_pairs",
        ]
        header.extend(f"{column}_allan_deviation" for column in columns)
        writer.writerow(header)
        for index, tau in enumerate(common_taus):
            cluster_size = max(
                1, int(round(tau * sampling["sample_rate_hz"]))
            )
            nonoverlap_pairs = max(
                0, rates.shape[0] // cluster_size - 1
            )
            writer.writerow(
                [tau, axis_results[0][4][index], nonoverlap_pairs]
                + [result[3][index] for result in axis_results]
            )

    parameter_csv = args.output_dir / "allan_parameters.csv"
    write_parameter_csv(parameter_csv, axis_reports)

    metadata = {
        "maturity": "verified MVP; not deployment-ready",
        "source_file": str(args.input_path),
        "input_format": args.input_path.suffix.lower().lstrip("."),
        "timestamp_column": args.timestamp_column,
        "value_columns": columns,
        "timestamp_unit": args.timestamp_unit,
        "timestamps_are_relative_internally": True,
        "input_kind": args.input_kind,
        "value_scale_to_si": args.value_scale_to_si,
        "sample_count": int(rates.shape[0]),
        "duration_s": duration,
        "sampling": sampling,
        "tau_min_s": float(common_taus[0]),
        "tau_max_s": float(common_taus[-1]),
        "white_fit_range_s": white_fit_range,
        "random_walk_fit_range_s": random_walk_fit_range,
        "slope_tolerance": args.slope_tolerance,
        "min_pairs_warning_threshold": args.min_pairs,
        "axes": axis_reports,
        "limitations": [
            "input must be stationary and uniformly sampled",
            "no automatic motion, saturation, temperature, or outlier rejection",
            "difference-pair count is not equivalent degrees of freedom",
            "bias-instability minimum is a candidate, not a final calibration",
            "random-walk parameters require an explicit valid +1/2 fit region",
        ],
    }
    metadata_output = args.output_dir / "analysis_metadata.json"
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    column_count = min(3, len(axis_results))
    row_count = math.ceil(len(axis_results) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 3.8 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    for plot_axis, result, report in zip(
        axes.flat, axis_results, axis_reports
    ):
        column, rate_unit, taus, deviations, _ = result
        plot_axis.loglog(taus, deviations, color="black", linewidth=1.5)
        white = report["parameters"]["white_noise"]
        if white is not None:
            fit_mask = (
                (taus >= white["tau_min_s"])
                & (taus <= white["tau_max_s"])
            )
            fitted = (
                10.0 ** white["intercept"]
                * taus[fit_mask] ** white["slope"]
            )
            plot_axis.loglog(
                taus[fit_mask],
                fitted,
                linestyle="--",
                color="tab:blue",
                label=f"white fit, slope={white['slope']:+.3f}",
            )
        bias = report["parameters"]["bias_instability_candidate"]
        plot_axis.scatter(
            [bias["tau_s"]],
            [bias["allan_deviation"]],
            color="tab:orange",
            zorder=3,
            label="minimum / bias candidate",
        )
        white_text = (
            f"White: {report['white_common_value']:.4g} "
            f"{report['white_common_unit']}\n"
            f"slope: {white['slope']:+.3f}"
            if white is not None
            else "White: not fitted"
        )
        bias_quality = (
            "preliminary"
            if bias["is_interior_minimum"] and bias["is_near_plateau"]
            else "boundary / unreliable"
        )
        if report["bias_common_value"] is not None:
            bias_value_text = (
                f"{report['bias_common_value']:.4g} "
                f"{report['bias_common_unit']}"
            )
        else:
            bias_value_text = (
                f"{bias['coefficient']:.4g} {report['rate_unit']}"
            )
        bias_text = (
            f"Bias candidate: {bias_value_text}\n"
            f"tau={bias['tau_s']:.3g} s, {bias_quality}"
        )
        plot_axis.text(
            0.03,
            0.04,
            f"{white_text}\n{bias_text}",
            transform=plot_axis.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.88,
            },
        )
        plot_axis.set_title(column)
        plot_axis.set_xlabel("Cluster time tau [s]")
        plot_axis.set_ylabel(f"Allan deviation [{rate_unit}]")
        plot_axis.grid(True, which="both", alpha=0.3)
        plot_axis.legend(fontsize=8)
    for unused_axis in axes.flat[len(axis_results):]:
        unused_axis.set_visible(False)
    fig.suptitle("Overlapping Allan deviation of stationary IMU data")
    figure_output = args.output_dir / "allan_deviation.png"
    fig.savefig(figure_output, dpi=180)
    plt.close(fig)

    summary_figure, summary_axis = plt.subplots(
        figsize=(14, 5.6), constrained_layout=True
    )
    summary_axis.axis("off")
    summary_rows = []
    for report in axis_reports:
        white = report["parameters"]["white_noise"]
        bias = report["parameters"]["bias_instability_candidate"]
        bias_quality = (
            "初步候选"
            if bias["is_interior_minimum"] and bias["is_near_plateau"]
            else "边界不可靠"
        )
        summary_rows.append(
            [
                report["column"],
                f"{white['slope']:+.3f}" if white else "-",
                (
                    f"{report['white_common_value']:.6g} "
                    f"{report['white_common_unit']}"
                    if report["white_common_value"] is not None
                    else (
                        f"{white['coefficient']:.6g} "
                        f"{report['white_coefficient_unit']}"
                        if white
                        else "not fitted"
                    )
                ),
                (
                    f"{report['bias_common_value']:.6g} "
                    f"{report['bias_common_unit']}"
                    if report["bias_common_value"] is not None
                    else (
                        f"{bias['coefficient']:.6g} "
                        f"{report['rate_unit']}"
                    )
                ),
                f"{bias['tau_s']:.4g} s",
                bias_quality,
                (
                    f"{report['statistics']['quantization_step_estimate']:.6g}"
                ),
                (
                    f"{report['statistics']['first_to_last_window_drift']:+.6g}"
                ),
            ]
        )
    summary_table = summary_axis.table(
        cellText=summary_rows,
        colLabels=[
            "轴",
            "白噪声斜率",
            "ARW / VRW",
            "零偏不稳定性候选值",
            "最低点 τ",
            "候选值质量",
            "量化步进估计",
            "首末窗口漂移",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.09, 0.16, 0.15, 0.10, 0.15, 0.12, 0.12],
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)
    summary_table.scale(1.0, 2.0)
    for column_index in range(8):
        summary_table[(0, column_index)].set_facecolor("#dce6f1")
        summary_table[(0, column_index)].set_text_props(weight="bold")
    for row_index, report in enumerate(axis_reports, start=1):
        white = report["parameters"]["white_noise"]
        bias = report["parameters"]["bias_instability_candidate"]
        if white is not None and white["is_valid"]:
            summary_table[(row_index, 2)].set_facecolor("#e2f0d9")
        if not (
            bias["is_interior_minimum"] and bias["is_near_plateau"]
        ):
            summary_table[(row_index, 3)].set_facecolor("#fce4d6")
            summary_table[(row_index, 5)].set_facecolor("#fce4d6")
    summary_axis.set_title(
        "IMU Allan 方差参数汇总\n"
        "绿色：白噪声拟合有效；橙色：零偏候选值可靠性不足",
        fontsize=15,
        pad=14,
    )
    summary_axis.text(
        0.01,
        0.02,
        "速率随机游走：当前没有经过确认的 +1/2 斜率区间，因此未辨识。"
        "零偏数值来自曲线最低点，只是候选值，不能直接作为 ESKF 过程噪声。",
        transform=summary_axis.transAxes,
        fontsize=9,
    )
    summary_figure_output = (
        args.output_dir / "allan_parameter_summary.png"
    )
    summary_figure.savefig(summary_figure_output, dpi=180)
    plt.close(summary_figure)

    pair_counts = axis_results[0][4]
    cluster_sizes = np.maximum(
        1, np.rint(common_taus * sampling["sample_rate_hz"]).astype(int)
    )
    nonoverlap_pairs = np.maximum(
        0, rates.shape[0] // cluster_sizes - 1
    )
    fig_pairs, ax_pairs = plt.subplots(
        figsize=(10, 4), constrained_layout=True
    )
    ax_pairs.loglog(
        common_taus,
        pair_counts,
        color="black",
        label="raw overlapping pairs (correlated)",
    )
    ax_pairs.loglog(
        common_taus,
        nonoverlap_pairs,
        color="tab:blue",
        linestyle="--",
        label="non-overlapping baseline pairs",
    )
    ax_pairs.axhline(
        args.min_pairs,
        color="tab:red",
        linestyle="--",
        label=f"warning threshold: {args.min_pairs} pairs",
    )
    ax_pairs.set_title("Allan difference-pair count diagnostics")
    ax_pairs.set_xlabel("Cluster time tau [s]")
    ax_pairs.set_ylabel("Difference pairs")
    ax_pairs.grid(True, which="both", alpha=0.3)
    ax_pairs.legend()
    pair_figure_output = args.output_dir / "allan_difference_pairs.png"
    fig_pairs.savefig(pair_figure_output, dpi=180)
    plt.close(fig_pairs)

    report_output = args.output_dir / "allan_parameter_report.zh-CN.md"
    write_report(
        report_output,
        input_path=args.input_path,
        sample_count=int(rates.shape[0]),
        duration=duration,
        sampling=sampling,
        white_fit_range=white_fit_range,
        random_walk_fit_range=random_walk_fit_range,
        axis_reports=axis_reports,
    )

    print(f"Samples: {rates.shape[0]}, duration: {duration:.3f} s")
    print(
        "Sampling: "
        f"{sampling['sample_rate_hz']:.9f} Hz, "
        f"jitter RMS={sampling['relative_jitter_rms']:.3e}, "
        f"max gap ratio={sampling['max_relative_gap']:.6f}"
    )
    for output in (
        curve_csv,
        parameter_csv,
        metadata_output,
        figure_output,
        summary_figure_output,
        pair_figure_output,
        report_output,
    ):
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
