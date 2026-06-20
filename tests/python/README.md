# Python Tests

This directory contains lightweight verification tests for the Python prototypes.

Current coverage includes:

- KF and EKF matrix dimensions and invalid-input handling
- closed-form one-step KF reference values
- analytic versus finite-difference EKF Jacobians
- range-bearing angle wrapping across the `±pi` branch
- Joseph covariance update, symmetry, and positive semidefiniteness
- deterministic synthetic convergence checks
- overlapping and non-overlapping Allan deviation checks
- IMU timestamp gap diagnostics
- IMU mechanization input validation and two-sample interval assumptions
- static, free-fall, constant-yaw, and measured two-frame propagation checks
- roll-pitch-yaw and quaternion body-to-NED convention consistency
- WGS-84, stationary Earth-rate compensation, covariance PSD, GNSS lever arm,
  innovation gating, known IMU calibration, timestamp failures, and dataset1
  ESKF replay integration
- static-window validation, analytic leveling, yaw observability rejection,
  gyro bias initialization, and dataset1 navigation-grade gyrocompassing

Run from the repository root:

```powershell
python -m pytest tests
```
