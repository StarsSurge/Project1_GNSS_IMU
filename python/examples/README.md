# Python Examples

This directory contains small, runnable examples for each learning step.

Available examples:

- `demo_1d_kalman_filter.py`: one predict-update cycle with printed matrices
- `demo_kalman_filter.py`: 2D position tracking with a general linear KF
- `demo_extended_kalman_filter.py`: 2D range-bearing tracking with an EKF
- `analyze_imu_allan.py`: overlapping Allan analysis for a named-column IMU CSV

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

For degree-per-second input, add
`--value-scale-to-si 0.017453292519943295`. For delta-angle input, use
`--input-kind increment`; increment row `i` is divided by
`timestamp[i] - timestamp[i-1]`, and the first row is dropped.

The tool rejects excessive timestamp jitter and gaps by default. It does not
automatically detect motion, saturation, temperature transients, or outliers;
these must be checked before interpreting the result.

GNSS/IMU mechanization and ESKF demos are the next planned examples.
