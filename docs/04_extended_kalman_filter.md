# Extended Kalman Filter (EKF)

## Problem Definition

The linear Kalman filter assumes that both the motion model ``f`` and the
observation model ``h`` are linear transformations of the state.  Many
real-world sensors — GNSS pseudorange, range-bearing radar, monocular
camera — produce measurements that are *non-linear* functions of the
state:

```text
x_k = f(x_{k-1}) + w       (motion)
z_k = h(x_k) + v            (observation)
```

The Extended Kalman Filter handles these non-linearities by linearising
about the current state estimate at each step.

Our motivating example is **2D range-bearing tracking**: a sensor at a
known location measures the range ``r`` and bearing ``θ`` to a moving
target.  Both are non-linear functions of the target's Cartesian
position.

## State Variables and Dimensions

The 2D range-bearing EKF state:

```text
x = [px, py, vx, vy]ᵀ,  dim(x) = 4 × 1
```

Measurement:

```text
z = [r, θ]ᵀ,  dim(z) = 2 × 1
```

Matrices:

```text
P: 4 × 4,  state error covariance
Q: 4 × 4,  process noise covariance
R: 2 × 2,  measurement noise covariance
K: 4 × 2,  Kalman gain
F_jac(x): 4 × 4  (linearised motion model Jacobian)
H_jac(x): 2 × 4  (linearised observation Jacobian)
```

## Mathematical Formulas

### Linearisation

At each time step the EKF linearises using first-order Taylor expansions
around the current estimate ``x̂``:

```text
F = ∂f/∂x |_{x̂}        state transition Jacobian
H = ∂h/∂x |_{x̂}        observation Jacobian
```

### Prediction

```text
x_pred = f(x)                   (non-linear propagation)
P_pred = F P Fᵀ + Q             (linearised covariance propagation)
```

### Update

```text
z_pred = h(x_pred)              (predicted measurement)
residual = z - z_pred           (innovation)
S = H P_pred Hᵀ + R             (innovation covariance)
K = P_pred Hᵀ S⁻¹               (Kalman gain)
x_upd = x_pred + K residual     (state correction)
P_upd = (I - K H) P_pred        (Joseph-form covariance)
```

## Range-Bearing Observation Model

Sensor at ``(sx, sy)``, target at ``(px, py)``:

```text
r   = √((px - sx)² + (py - sy)²)
θ   = atan2(py - sy, px - sx)
```

The analytic Jacobian of ``h = [r, θ]ᵀ`` with respect to ``x``:

```text
H = ┌                              ┐
    │  dx/r   dy/r   0   0 │
    │ -dy/r²  dx/r²  0   0 │
    └                              ┘

where dx = px - sx, dy = py - sy, r = √(dx² + dy²)
```

## Derivation Outline

1. Start with the non-linear system equations.
2. Linearise around the current estimate using first-order Taylor
   expansion.
3. Apply the standard linear Kalman filter prediction with the
   linearised ``F`` and non-linear ``f``.
4. Apply the standard Kalman update with the linearised ``H`` and
   non-linear ``h`` to compute the innovation.
5. The Kalman gain ``K`` is computed from the linearised ``H`` and the
   current ``P``.

## Implementation Details

Core class in:

```text
python/gnss_imu/extended_kalman_filter.py
```

The EKF requires four user-supplied callables:

| Callable | Signature | Purpose |
|----------|-----------|---------|
| ``f(x)`` | ``(n,1) → (n,1)`` | Non-linear state transition |
| ``F_jac(x)`` | ``(n,1) → (n,n)`` | Jacobian of ``f`` |
| ``h(x)`` | ``(n,1) → (m,1)`` | Non-linear observation |
| ``H_jac(x)`` | ``(n,1) → (m,n)`` | Jacobian of ``h`` |

Factory function: ``create_range_bearing_ekf()`` — builds an EKF for
2D range-bearing tracking with sensible defaults.

Demo in:

```text
python/examples/demo_extended_kalman_filter.py
```

Tests in:

```text
tests/python/test_extended_kalman_filter.py
```

## Limitations of the EKF

1. **First-order approximation**: the EKF only keeps the linear term of
   the Taylor expansion.  Strong non-linearities can cause divergence.
2. **Jacobian maintenance**: analytic Jacobians are error-prone;
   numerical Jacobians are slower.  Both must be correct for the filter
   to work.
3. **Consistency**: the EKF can become over-confident when linearisation
   errors are large (the covariance ``P`` underestimates true error).
4. **Initialisation sensitivity**: a poor initial guess means the filter
   linearises around a bad point, which can lead to divergence.
5. **Not invariant to coordinate choice**: linearisation in Cartesian
   vs. polar coordinates can give different results.

## Relationship to ESKF (Error-State Kalman Filter)

The EKF linearises the full state directly.  The Error-State Kalman
Filter (ESKF) — the next step in this project — separates the nominal
state (propagated non-linearly) from a small error state (propagated
linearly).  This is the standard approach for GNSS/IMU fusion because:

- The error state is always small → linearisation is more accurate.
- Attitude is parameterised as a quaternion in the nominal state but a
  3-DOF small-angle vector in the error state.
- The error state can be reset to zero after each correction.

## Common Interview Questions

- Why linearise?  What would happen if we used the non-linear ``h``
  directly in the Kalman update?
- What is the difference between analytic and numerical Jacobians?
  When would you use each?
- Under what conditions does the EKF diverge?
- How does the EKF compare to the Unscented Kalman Filter (UKF)?
  What about particle filters?
- Why is the Joseph form ``(I - KH) P`` preferred for the covariance
  update?
- In the range-bearing example, why is bearing noise more damaging at
  long range?
- What happens to the Jacobian when the target is exactly at the
  sensor?  How do you handle it?

## Common Mistakes

- Forgetting to linearise ``F`` at the *current* state before prediction
  (using an old Jacobian).
- Using a constant Jacobian throughout — it must be recomputed at each
  step.
- Not checking that ``h(x)`` and ``z`` have the same dimension.
- Using ``arctan2`` without checking the quadrant convention.
- Forgetting the chain rule when computing ``F_jac`` for a composition
  of transformations.
- Setting ``R`` too small — the EKF trusts bad linearisation-based
  predictions when measurement noise is underestimated.
- Choosing initial ``P`` too small — the filter rejects new measurements
  and converges slowly or not at all.
