# Error-State Kalman Filter (ESKF) — GNSS/IMU Loose Coupling

> The MVP implementation follows the frozen right-multiplicative,
> body-frame attitude-error convention in
> [06_eskf_mvp_spec.md](06_eskf_mvp_spec.md).

## Problem Definition

In GNSS/IMU integration we have two complementary sensors:

- **IMU**: high-rate (200 Hz) angular and velocity increments.  Smooth
  short-term but drifts without bound due to integration of bias and noise.
- **GNSS**: low-rate (1 Hz) absolute position fixes.  No drift but noisy.

The goal is a filter that predicts with IMU at every step and corrects with
GNSS when a position fix arrives, outputting a continuous, smooth estimate of
position, velocity, and attitude.

**Why estimate the *error* rather than the full state?**  The full state
(position in kilometres, velocity in m/s) involves large numerics.
The error (centimetres of position error, cm/s of velocity error, fractions
of a degree of attitude error) is always small.  Linearising around a small
value produces a far more accurate approximation than linearising around a
large one:

```
ESKF philosophy:
    Nominal state → non-linear propagation (accumulates large values safely)
    Error state   → linear propagation of covariance (always small, linear OK)
    GNSS update   → Kalman correction on the error state
    Inject + reset → fold correction into nominal, zero the error, keep P
```

---

## Dual State Architecture

### Nominal State — 16 DOF

The "best guess" of the true state, propagated with the full non-linear
INS mechanisation:

```
x_nom = [p, v, q, b_a, b_g]

p   : position (NED frame)           3×1  [m]
v   : velocity (NED frame)           3×1  [m/s]
q   : attitude quaternion (body→nav) 4×1  [unitless]
b_a : accelerometer bias             3×1  [m/s²]
b_g : gyroscope bias                 3×1  [rad/s]
```

### Error State — 15 DOF

The small difference between true state and nominal state — always near zero:

```
δx = [δp, δv, δθ, δb_a, δb_g]

δp   : position error           3×1  [m]
δv   : velocity error           3×1  [m/s]
δθ   : attitude error           3×1  [rad]  ← minimal 3-DOF representation
δb_a : accel bias error         3×1  [m/s²]
δb_g : gyro bias error          3×1  [rad/s]
```

**Key insight**: attitude is a 4-D quaternion in the nominal state but a
3-D small-angle vector in the error state.  This keeps the covariance
matrix minimal (15×15 instead of 16×16) and avoids quaternion norm
constraint issues in the EKF.

---

## IMU Error Model — Complete Taxonomy

### A. Systematic Errors (deterministic, can be calibrated)

| Type | Symbol | Unit | Source | Estimable? |
|------|--------|------|--------|:--:|
| **Bias** | b_a, b_g | m/s², rad/s | Sensor offset, slowly drifts after power-on | Online (ESKF state) |
| **Scale Factor** | s_a, s_g | ppm | Linear deviation between output and truth | Online (ESKF state) |
| **Misalignment** | m_xy, ... | rad | Non-orthogonal axes causing cross-leakage | Factory cal only |
| **Non-linearity** | — | % | Non-linear response at large inputs | Negligible in rated range |
| **g-sensitivity** | — | °/h/g | Acceleration affecting gyro output | May be significant for MEMS |

### B. Stochastic Errors (unpredictable, described statistically)

Five noise types identified by Allan variance:

| Noise Type | Allan Slope | Physical Meaning | Variance in Q_d |
|-----------|:--:|------|------|
| **ARW / VRW** (white noise) | −1/2 | High-frequency random jitter, integrated | continuous PSD σ²; first-order Qd multiplies by dt |
| **Bias Instability** | 0 | Slow bias drift magnitude over time | Modelled via σ_ba²·I·dt |
| **Rate Random Walk** | +1/2 | Random walk driving the bias itself | σ²·I·dt |
| **Rate Ramp** | +1 | Ultra-slow deterministic drift (ageing) | Usually ignored |
| **Quantisation** | −1 | A/D discretisation error | Usually ignored |

### C. Measurement Equations (with scale factor & misalignment)

```
Gyro:      Δθ_meas = (I + S_g + M_g)·Δθ_true + b_g·dt + n_g·√dt
Accel:     Δv_meas  = (I + S_a + M_a)·Δv_true  + b_a·dt + n_a·√dt
```

Where S = diag(s_x, s_y, s_z) is scale factor and M is the misalignment matrix.

### D. Bias Evolution Model

Standard model is 1st-order Gauss-Markov:

```
ḃ(t) = −(1/τ)·b(t) + w(t)

τ  : correlation time
w(t): driving white noise, σ² = 2·σ_b²/τ

τ → ∞ : pure random walk (current ESKF simplification)
τ → 0 : pure white noise
```

### E. Noise Parameters for Leador-A15 (tactical-grade MEMS)

```
Gyro:
  ARW       : 0.05 °/√h  ≈ 2.5e-4 rad/s/√Hz
  Bias Inst : 0.5 °/h    ≈ 2.4e-6 rad/s

Accel:
  VRW       : 0.05 m/s/√h ≈ 8.3e-4 m/s²/√Hz
  Bias Inst : 50 μg       ≈ 4.9e-4 m/s²
```

---

## INS Precise Mechanisation — Two-Subsample Coning/Sculling Compensation

The fundamental challenge: **IMU measures increments in a rotating body frame,
but we need navigation state in the fixed nav frame.** If the body rotates
during dt, the direction of velocity increments changes mid-interval.

### Three Incommutability Errors

**1. Coning —** two orthogonal axes oscillating at the same frequency
produce a net rotation about the third axis (even with zero net angular rate).
Analogy: a cone rolling on a table; the cone gains a net spin about its own axis.

**2. Sculling —** angular oscillation × linear oscillation = fictitious
constant velocity. Analogy: rowing a boat — hand wiggles (angular) and the
blade pushes water (linear) → boat moves at steady speed.

**3. Rotation Effect —** during dt the body frame rotates; the velocity
increment measured early in the interval is in a *different* body frame
than one measured late in the interval.

### Two-Subsample Compensation

Split the navigation interval into two halves `[t, t+T/2]` and `[t+T/2, t+T]`.
Let Δθ₁,Δθ₂ and Δv₁,Δv₂ be the increments.

**Coning compensation:**
```
Δθ_coning = (2/3)·(Δθ₁ × Δθ₂)       ← cross-product correction
Δθ_corrected = Δθ + Δθ_coning
```

**Sculling compensation:**
```
Δv_scul = (2/3)·(Δθ₁ × Δv₂ + Δv₁ × Δθ₂)
```

**Rotation effect compensation:**
```
Δv_rot = (1/2)·(Δθ × Δv)
```

**Total velocity correction:**
```
Δv_corrected = Δv + Δv_scul + Δv_rot
```

### Complete Two-Subsample Mechanisation

```
① Total increments:  Δθ = Δθ₁+Δθ₂,  Δv = Δv₁+Δv₂
② Coning:            Δθ += (2/3)·(Δθ₁ × Δθ₂)
③ Sculling:          Δv_scul = (2/3)·(Δθ₁ × Δv₂ + Δv₁ × Δθ₂)
④ Rotation:          Δv_rot = (1/2)·(Δθ × Δv)
⑤ Total Δv:          Δv += Δv_scul + Δv_rot
⑥ Attitude:          q_new = q_old ⊗ axis_angle_to_quat(Δθ)
⑦ Velocity:          C = quat_to_dcm(q_old); v_new = v_old + C@Δv + g·dt
⑧ Position:          p_new = p_old + (v_old + v_new)·dt/2
```

### Accuracy Comparison (vehicle, 200 Hz, ~10°/s, ~0.5 g vib)

| Error | Uncompensated | Two-Subsample |
|-------|:---:|:---:|
| Yaw drift (coning) | ~0.1 °/h | ~0.001 °/h |
| Velocity bias (sculling) | ~0.5 cm/s | ~0.001 cm/s |
| Position drift (100 s) | ~0.5 m | ~1 mm |

**For MVP**: 200 Hz single-subsample is sufficient to verify the logic.
For down-sampled processing (e.g. 100 Hz), two-subsample is required.

---

## Error-State Dynamics — Precise P Propagation

### Continuous-Time Error-State Equations

```
δṗ  = δv
δv̇  = −C_b^n·[f_b×]·δθ − C_b^n·δb_a − C_b^n·n_a
δθ̇  = −[ω_b×]·δθ − δb_g − n_g
δḃ_a = n_ba
δḃ_g = n_bg
```

### F Matrix — 15×15 Block Structure

```
     ┌                            ┐
     │ 0     I₃    0     0     0 │  ← δṗ = δv
F =  │ 0     0   −C[f×]   −C     0 │
     │ 0     0   −[ω×]     0    −I │
     │ 0     0     0     0     0 │  ← δḃ_a = 0
     │ 0     0     0     0     0 │  ← δḃ_g = 0
     └                            ┘
```

Block (1,2) `−C[f_b×]` is the dominant term: attitude error → wrong gravity
compensation direction → velocity drifts linearly in time → position drifts quadratically.

### G Matrix — 15×12 Block Structure

```
     ┌                      ┐
     │ 0     0    0    0  │
G =  │−C     0    0    0  │  ← accel noise → velocity
     │ 0    −I    0    0  │  ← gyro noise → body-frame attitude
     │ 0     0    I₃   0  │  ← accel bias random walk
     │ 0     0    0    I₃ │  ← gyro bias random walk
     └                     ┘
```

### Discretisation Methods

**(A) 1st-order (MVP):** `Φ ≈ I + F·dt` — sufficient at 200 Hz (dt = 5 ms).

**(B) Matrix exponential:** `Φ = exp(F·dt) = I + F·dt + (F·dt)²/2! + ...`
Upper-triangular F means terms beyond 3rd order are mostly zero.

**(C) van Loan (production):** Construct augmented matrix, compute matrix
exponential once to obtain both Φ and Q_d simultaneously.

### Discrete Process Noise

```
Q_c = diag(σ_a²·I₃, σ_g²·I₃, σ_ba²·I₃, σ_bg²·I₃)    (12×12)

Q_d ≈ Q_c · dt                                         (1st-order discrete)

P_pred = Φ @ P @ Φᵀ + G @ Q_d @ Gᵀ
```

### Continuous-time error-state equations

```
δṗ  = δv
δv̇  = -C_b^n [f_b×] δθ - C_b^n δb_a - C_b^n n_a
δθ̇  = -[ω_b×]δθ - δb_g - n_g
δḃ_a = n_ba
δḃ_g = n_bg
```

In matrix form:  **δẋ = F·δx + G·n_imu**

### F matrix — 15×15 block structure

```
     ┌                            ┐
     │ 0     I     0     0     0 │  ← δṗ
     │ 0     0  -C[f×]   -C     0 │  ← δv̇
F =  │ 0     0   -[ω×]    0    -I │  ← δθ̇
     │ 0     0     0     0     0 │  ← δḃ_a
     │ 0     0     0     0     0 │  ← δḃ_g
     └                            ┘

Each block is 3×3:

Block (0,1) = I₃          — position error grows with velocity error
Block (1,2) = -C[f_b×]     — specific-force coupling
Block (1,3) = -C           — velocity error grows with accel bias
Block (2,2) = -[ω_b×]      — body-frame attitude-error rotation
Block (2,4) = -I           — attitude error grows with gyro bias
```

### G matrix — 15×12 block structure

```
     ┌                      ┐
     │ 0    0    0    0  │
     │-C    0    0    0  │   ← accel noise enters velocity channel
G =  │ 0   -I    0    0  │   ← gyro noise enters attitude channel
     │ 0    0    I    0  │   ← bias random walk for accel
     │ 0    0    0    I  │   ← bias random walk for gyro
     └                    ┘
```

### Discretisation

```
Φ = I + F·dt                (1st-order approximation, 15×15)

P_pred = Φ @ P @ Φ^T + G @ Q_d @ G^T
```

Where `Q_d` (12×12) contains the discrete-time process noise:

```
Q_c = diag(σ_a²·I₃, σ_g²·I₃, σ_ba²·I₃, σ_bg²·I₃)
Q_d = Q_c·dt
```

---

## GNSS Loosely-Coupled Update

### Observation equation (for the error state)

```
z_gnss = p_ned_true + n_gnss       — observed position

residual = z_gnss - p_nom          — observed minus nominal = position error observation

H = [I₃, 0₃, 0₃, 0₃, 0₃]         — observes position only  (3×15)
```

### Standard Kalman update

```
S = H @ P @ H^T + R_gnss
K = P @ H^T @ S^{-1}               (15×3)

δx = K @ residual
A = I - K @ H
P = A @ P @ A.T + K @ R @ K.T     (Joseph form)
```

---

## Error Injection & Reset — The ESKF-Specific Step

### Inject

Fold the estimated error into the nominal state:

```
p_nom += δp
v_nom += δv
q_nom ← q_nom ⊗ δq      where δq ≈ [1, δθ/2] (small-angle quaternion)
b_a   += δb_a
b_g   += δb_g
```

### Reset

After injection the error state is zero by definition.  The covariance
undergoes a gauge transformation, but for small errors J ≈ I₁₅:

```
δx = 0
P  = J @ P @ J^T  ≈ P   (simplified: keep P unchanged)
```

---

## Complete ESKF Pipeline

```
Initialisation:
    p, v, q ← from first few seconds of truth
    b_a, b_g = 0
    P = diag(initial variances)      (15×15)

━━━ Main loop (200 Hz) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each IMU epoch (dθ, dv):
    dt = t_curr - t_prev

    ┌─ predict_imu() ──────────────────────────┐
    │ ① Mechanisation: update p, v, q           │
    │ ② C = quat_to_dcm(q)                      │
    │ ③ Build F (15×15), G (15×12)              │
    │    Φ = I + F·dt                           │
    │ ④ P = Φ @ P @ Φ^T + G @ Q_d @ G^T         │
    └───────────────────────────────────────────┘

    if GNSS epoch:
    ┌─ update_gnss() + inject_reset() ──────────┐
    │ ⑤ z = lla_to_ned(rtk)                     │
    │    residual = z - p_nom                   │
    │ ⑥ H = [I₃, 0₃, 0₃, 0₃, 0₃]              │
    │    K = P @ H^T @ S^{-1}                   │
    │ ⑦ δx = K @ residual                      │
    │    A = I - K @ H                          │
    │    P = A @ P @ A.T + K @ R @ K.T         │
    │ ⑧ inject error into nominal; reset δx = 0 │
    └───────────────────────────────────────────┘

    Log estimated state → compare with truth.nav
```

---

## ESKF vs Ordinary EKF

| | EKF (range-bearing) | ESKF (GNSS/IMU) |
|---|---|---|
| State | Filtered directly on [px,py,vx,vy] | Nominal + error state separation |
| Nominal propagation | f(x) — constant velocity | INS mechanisation with IMU increments |
| Error propagation | — | F·δx, linear in error space |
| Covariance dimension | 4×4 | 15×15 (on the 15-D error state) |
| Linearisation point | Current nominal state | δx = 0 (always the optimal point!) |
| Key advantage | Simplicity | Better linearisation accuracy; attitude in quaternion avoids gimbal lock |

---

## Common Interview Questions

- Why is ESKF better than directly EKF-ing the full state?
- The nominal state is 16-D but the error state is 15-D.  Where did the extra DOF go?
- Why must the error state be reset to zero after injection?  What happens if you don't?
- What is the physical meaning of the `-C[f_b×]` block in the F matrix?
- If GNSS is lost for an extended period (e.g. tunnel), how does P evolve?
  What is the drift characteristic of pure INS?
- Why are b_a and b_g modelled as random walks (ḃ = n) rather than
  first-order Gauss-Markov (ḃ = -βb + n)?
- What advantage does quaternion attitude parameterisation have over
  Euler angles?

## Common Mistakes

- Reusing a stale DCM C instead of recomputing it each IMU step
- Adding the attitude error δθ to the quaternion instead of quaternion-multiplying
- Mixing left/nav-frame and right/body-frame attitude-error equations
- Using the wrong sign for the bias blocks in F
- Using Δv directly as acceleration and forgetting to account for dt
- Subtracting NED position from WGS84 lat/lon/alt without coordinate conversion
- Getting the gravity sign wrong: +g (down) in NED vs −g (up) in ENU
