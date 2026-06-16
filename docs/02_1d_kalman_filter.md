# 1D Kalman Filter: Position and Velocity Observation Fusion

## Problem Definition

This section implements a minimal but complete 1D Kalman Filter for estimating the state of a target moving along a straight line:

```text
x = [position, velocity]^T
```

A sensor provides noisy observations of both position and velocity:

```text
z = [z_position, z_velocity]^T
```

This example mirrors a simplified problem in GNSS/robotics localization: an external sensor such as GNSS provides position, while a wheel odometer, Doppler velocity, or another module provides velocity. The filter performs weighted fusion based on the motion model and observation noise.

## State Variables and Dimensions

State vector:

```text
x = [p, v]^T, dim(x) = 2 x 1
```

Covariance and model matrices:

```text
P: 2 x 2, state estimation error covariance
F: 2 x 2, state transition matrix
H: 2 x 2, observation matrix
Q: 2 x 2, process noise covariance
R: 2 x 2, observation noise covariance
K: 2 x 2, Kalman gain
```

## Mathematical Formulas

The constant velocity motion model is used:

```text
p_k = p_{k-1} + v_{k-1} dt
v_k = v_{k-1}
```

In matrix form:

```text
x_k = F x_{k-1}

F = [1 dt
     0  1]
```

**Prediction:**

```text
x_pred = F x
P_pred = F P F^T + Q
```

**Update:**

```text
residual = z - H x_pred
S = H P_pred H^T + R
K = P_pred H^T S^{-1}
x_upd = x_pred + K residual
P_upd = (I - K H) P_pred       (simplified covariance update)
```

The implementation uses the numerically safer Joseph form
``A P_pred Aᵀ + K R Kᵀ``, where ``A = I - K H``. With the exact optimal
gain, both forms are algebraically equivalent.

For numerical stability, the code computes `K` using `np.linalg.solve` rather than explicitly constructing `S^{-1}`.

## Derivation Outline

1. Predict the next position and velocity using the constant velocity model.
2. `Q` represents the mismatch between real motion and the constant velocity assumption.
3. `H` maps the state into observation space. In this example both position and velocity are directly observable, so `H = I`.
4. The **residual** is the difference between actual and predicted observations.
5. `S` represents the uncertainty (innovation covariance) of the residual.
6. `K` automatically determines whether to trust the model or the observation more in this update.
7. Update `x` and `P`.

## Implementation Details

Core implementation:

```text
python/gnss_imu/kalman_filter_1d.py
```

Example script:

```text
python/examples/demo_1d_kalman_filter.py
```

Tests:

```text
tests/python/test_kalman_filter_1d.py
```

The default observation used in the demo is:

```text
z = [1.2, 0.9]^T
```

After one predict-update cycle the state is approximately:

```text
x_upd = [1.02387807, 0.96858594]^T
```

This shows the position observation `1.2 m` slightly pulls the predicted position upward, while the velocity observation `0.9 m/s` pulls the predicted velocity downward. Since the default position observation noise `R[0,0] = 4.0` is larger than the velocity observation noise `R[1,1] = 1.0`, the position update is more conservative (it trusts the model more than the observation for position).

## Common Interview Questions

- What do `P`, `Q`, and `R` physically represent?
- Why does a larger `R` cause the filter to trust observations less?
- Why is `solve` preferred over `inv` in engineering implementations?
- What do the residual and innovation covariance `S` represent?
- If the real target has acceleration but the model assumes constant velocity, should you increase `Q` or `R`?
- Does `P` always decrease over time? Under what circumstances can it increase?

## Common Mistakes

- Shaping the state vector as `(2,)`, which causes ambiguous matrix multiplication dimensions.
- Misunderstanding `Q` as an observation weight — it actually represents process noise.
- Misunderstanding `R` as the observation value itself — it actually represents observation noise covariance.
- Forgetting the transpose in `P_pred = F P F^T + Q`.
- Copying formulas directly and writing incorrect `solve(A, b)` argument order.
- Checking only final numerical values in tests without verifying matrix dimensions.
