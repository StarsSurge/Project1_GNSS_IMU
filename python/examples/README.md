# Python Examples

This directory contains small, runnable examples for each learning step.

Available examples:

- `demo_1d_kalman_filter.py`: one predict-update cycle with printed matrices
- `demo_kalman_filter.py`: 2D position tracking with a general linear KF
- `demo_extended_kalman_filter.py`: 2D range-bearing tracking with an EKF
- `analyze_imu_allan.py`: overlapping Allan analysis for named-column IMU
  CSV/Feather data, with parameter tables and an auditable report
- `visualize_dataset1.py`: first-pass RTK, IMU increment, and truth
  visualization for `data/dataset1`
- `demo_imu_state_update_mvp.py`: two-sample IMU nominal state update
  prototype with static and time-aligned measured-data cases
- `run_dataset1_eskf.py`: timestamp-driven 15-state GNSS/IMU loose ESKF replay
  with WGS-84 mechanization, lever arm, NIS gating, CSV/JSON, and error plots
- `demo_static_imu_alignment.py`: static detection, leveling, and
  navigation-grade gyrocompass initialization on dataset1

Run from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\python"
python python\examples\demo_kalman_filter.py
```

Example for a real gyro-rate CSV already converted to `rad/s`:

```powershell
$env:PYTHONPATH = "$PWD\python"
python python\examples\analyze_imu_allan.py data\raw\imu.csv `
  --timestamp-column timestamp_ns `
  --timestamp-unit ns `
  --value-columns gyro_x,gyro_y,gyro_z `
  --input-kind rate `
  --output-dir results\imu_allan
```

Feather example with an explicit, reviewed white-noise fit interval:

```powershell
$env:PYTHONPATH = "$PWD\python"
python python\examples\analyze_imu_allan.py data\allan\imu0.feather `
  --timestamp-column "Timestamp[nanosec]" `
  --timestamp-unit ns `
  --value-columns "gx[rad/s],gy[rad/s],gz[rad/s],ax[m/s^2],ay[m/s^2],az[m/s^2]" `
  --white-fit-min 0.01 `
  --white-fit-max 0.1 `
  --output-dir results\imu_allan\imu0
```

For degree-per-second input, add
`--value-scale-to-si 0.017453292519943295`. For delta-angle input, use
`--input-kind increment`; increment row `i` is divided by
`timestamp[i] - timestamp[i-1]`, and the first row is dropped.

The tool rejects excessive timestamp jitter and gaps by default. Integer epoch
timestamps are converted to relative time before floating-point scaling. It
does not automatically detect motion, saturation, temperature transients, or
outliers; these must be checked before interpreting the result.

GNSS/IMU mechanization and ESKF demos are the next planned examples.

IMU state-update MVP prototype:

```powershell
$env:PYTHONPATH = "$PWD\python"
.\.venv\Scripts\python.exe python\examples\demo_imu_state_update_mvp.py
```

The prototype verifies a known static case, then propagates two measured IMU
increments that occur after the selected `truth.nav` initialization epoch.

Dataset1 visualization:

```powershell
$env:PYTHONPATH = "$PWD\python"
python python\examples\visualize_dataset1.py
```

This writes trajectory, RTK uncertainty, RTK-minus-truth residual,
lever-arm diagnostic residual, IMU increment/rate, truth velocity/attitude
plots, and `summary.json` to `results\dataset1_visualization`.

GNSS/IMU ESKF 60-second replay:

```powershell
.\.venv\Scripts\python.exe python\examples\run_dataset1_eskf.py `
  --duration-s 60 `
  --imu-profile navigation-grade `
  --output-dir results\dataset1_eskf
```
