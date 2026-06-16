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

Run from the repository root:

```powershell
python -m pytest tests
```
