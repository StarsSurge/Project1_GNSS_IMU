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
- exact mid-interval delayed GNSS replay, fixed-lag expiry, known constant
  time-offset recovery, and static time-offset observability rejection
- profile-NIS interval interpolation and boundary detection, plus weighted
  constant-offset versus linear clock-drift model selection
- sub-grid quadratic refinement with a grid-resolution uncertainty floor
- robust joint lever/time calibration, observability rejection, MAP lever
  priors, and fixed-independent-lever time calibration
- antenna rotational velocity cross-product direction and nonlinear joint
  lever/time Gauss-Newton verification
- frozen-trajectory time-offset sign recovery and residual autocorrelation
  effective-sample-size diagnostics
- reproducible circular moving-block bootstrap with grid-resolution and
  candidate-boundary diagnostics
- GNSS integrity-state transitions, recovery covariance inflation, degradation
  after consecutive rejects, and dataset1 outage/reacquisition integration

Run from the repository root:

```powershell
python -m pytest tests
```
