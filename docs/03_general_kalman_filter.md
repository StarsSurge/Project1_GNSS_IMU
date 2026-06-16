# General N-Dimensional Kalman Filter

## Problem Definition

The 1D Kalman filter estimates a 2-element state ``[position, velocity]ᵀ``
with a fixed motion model and fixed observation matrix.  In real
localisation problems the state can have many more dimensions — 2D or 3D
position and velocity, sensor biases, clock errors — and the
measurements may observe only a subset of the state.

This section generalises the linear Kalman filter to arbitrary state
dimension *n* and measurement dimension *m* (with *m* ≤ *n*):

```text
x: (n, 1) — state vector
z: (m, 1) — measurement vector
```

## State Variables and Dimensions

The general Kalman filter matrix dimensions are:

```text
x: (n, 1)    state vector
P: (n, n)    error covariance
F: (n, n)    state transition matrix
H: (m, n)    observation matrix
Q: (n, n)    process noise covariance
R: (m, m)    measurement noise covariance
K: (n, m)    Kalman gain
```

For an *N*-dimensional constant-velocity model the state packs position
then velocity: ``[p₁ … p_d, v₁ … v_d]ᵀ``, giving *n* = 2·*d*.  The ``H``
matrix typically observes only position (``H = [I_d, 0_d]``), so *m* = *d*.

## Mathematical Formulas

The equations are the same as the 1D case, only the matrix sizes differ:

```text
x_pred = F x
P_pred = F P Fᵀ + Q

residual = z - H x_pred
S = H P_pred Hᵀ + R
K = P_pred Hᵀ S⁻¹
x_upd = x_pred + K residual
A = I - K H
P_upd = A P_pred Aᵀ + K R Kᵀ
```

The final line is the Joseph covariance update. With the exact optimal
Kalman gain it is algebraically equivalent to ``(I - K H) P_pred``, but
its symmetric structure is more robust to floating-point round-off.

The code uses ``np.linalg.solve`` for the Kalman gain:

```python
K = np.linalg.solve(S.T, (P @ H.T).T).T
```

## Block-Matrix Forms for Constant-Velocity Models

For *d* spatial dimensions with time step *dt*:

```text
    ┌         ┐
F = │ I_d  dt·I_d │
    │ 0_d    I_d  │
    └         ┘

    ┌         ┐
H = │ I_d  0_d │   (position only)
    └         ┘
```

If both position and velocity are observed (as in ``KalmanFilter1D``),
``H = I_{2d}`` and *m* = *n*.

## Derivation Outline

1. Start from the 1D Kalman filter equations.
2. Replace hard-coded ``(2, 1)`` and ``(2, 2)`` with dimension variables.
3. Infer *n* from ``x.shape[0]`` and *m* from ``H.shape[0]`` at construction.
4. Use the same predict / update logic — linear algebra works for any compatible dimensions.

## Relationship to KalmanFilter1D

``KalmanFilter1D`` is a concrete special case: *n* = 2, *m* = 2, both
position and velocity observed.  The general ``KalmanFilter`` reproduces
its numerical output exactly when initialised with the same matrices
(verified by cross-validation in the test suite).

Keep ``KalmanFilter1D`` as a learning stepping stone — its small,
hard-coded shapes make the equations easy to follow.  Move to
``KalmanFilter`` when the problem needs more than one spatial dimension
or partial-state observations.

## Implementation Details

Core class in:

```text
python/gnss_imu/kalman_filter.py
```

Factory functions:

- ``create_constant_velocity_filter_1d()`` — 1D, full-state observation,
  numerically identical to ``KalmanFilter1D``.
- ``create_constant_velocity_filter_nd(dim, dt, ...)`` — *d*-dimensional,
  position-only observation.

Demo in:

```text
python/examples/demo_kalman_filter.py
```

Tests in:

```text
tests/python/test_kalman_filter.py
```

## Common Interview Questions

- How does measurement dimension *m* relate to state dimension *n*?
  What happens if *m* < *n*?
- If ``H`` is not full column rank, is the system observable?
- Why is ``Q`` typically block-diagonal (position / velocity)?
- How would you modify ``F`` for a constant-acceleration model?
- What does the trace of ``P`` tell you about filter confidence?
- How would you handle a sensor that gives position in polar
  coordinates?  (This motivates the EKF.)

## Common Mistakes

- Forgetting that ``H`` has shape ``(m, n)``, not ``(n, n)`` — R must
  match *m*, not *n*.
- Passing a state vector as ``(n,)`` instead of ``(n, 1)``.
- Setting ``Q`` dimension based on measurement size instead of state
  size.
- Using the same noise parameters for 1D as for 2D/3D without scaling
  (Q and R are per-dimension).
- Expecting the filter to work well when *m* ≪ *n* and the unobserved
  states are not excited by the dynamics.
