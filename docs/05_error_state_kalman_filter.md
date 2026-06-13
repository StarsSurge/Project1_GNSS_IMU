# Error-State Kalman Filter (ESKF) — GNSS/IMU Loose Coupling

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

## IMU Measurement Model

Our Leador-A15 outputs **increments** — pre-integrated quantities — not
instantaneous rates:

```
Δθ = ∫ ω dt    angle increment [rad]
Δv = ∫ a dt    velocity increment [m/s]
```

Measurement equations (simplified):

```
Δθ = ω_true·dt + b_g·dt + n_g·√dt
Δv  = a_true·dt + b_a·dt + n_a·√dt
```

Biases are modelled as **random walks** driven by white noise:

```
ḃ_a = n_ba          ḃ_g = n_bg
```

The complete IMU noise vector is 12-dimensional:

```
n_imu = [n_a(3), n_g(3), n_ba(3), n_bg(3)]^T   (12×1)
```

---

## INS Mechanisation — Propagating the Nominal State

Mechanisation converts IMU angle and velocity increments into changes in
position, velocity, and attitude.

### Frame conventions (our dataset)

```
IMU frame:  Forward-Right-Down  (X-front, Y-right, Z-down)
Nav frame:  North-East-Down     (NED)
Gravity:    g_n = [0, 0, +9.78]^T  (positive-down in NED)
Rate:       200 Hz, dt = 0.005 s
```

### Mechanisation equations (single-subsample, 1st-order)

**(1) Attitude update** — construct incremental quaternion, right-multiply:

```
Δθ  = √(dθ_x² + dθ_y² + dθ_z²)
u   = [dθ_x, dθ_y, dθ_z] / Δθ
δq  = [cos(Δθ/2), u·sin(Δθ/2)]^T

q_b^n (new) = q_b^n (old) ⊗ δq    (Hamilton product)
```

**(2) Velocity update** — rotate body-frame increment to nav-frame, add gravity:

```
Δv_n = C_b^n @ Δv
v_n (new) = v_n (old) + Δv_n + g_n · dt
```

Where `C_b^n = quat_to_dcm(q_b^n)`.

**(3) Position update** — trapezoidal integration using mean velocity:

```
p_n (new) = p_n (old) + v_n (old)·dt + 0.5·(Δv_n + g_n·dt)·dt
```

---

## Error-State Dynamics — Propagating P

### Continuous-time error-state equations

```
δṗ  = δv
δv̇  = -C_b^n [Δv×] δθ + C_b^n δb_a + C_b^n n_a
δθ̇  = -C_b^n δb_g - C_b^n n_g
δḃ_a = n_ba
δḃ_g = n_bg
```

In matrix form:  **δẋ = F·δx + G·n_imu**

### F matrix — 15×15 block structure

```
     ┌                            ┐
     │ 0     I     0     0     0 │  ← δṗ
     │ 0     0  -C[Δv×]  C     0 │  ← δv̇
F =  │ 0     0     0     0    -C │  ← δθ̇
     │ 0     0     0     0     0 │  ← δḃ_a
     │ 0     0     0     0     0 │  ← δḃ_g
     └                            ┘

Each block is 3×3:

Block (0,1) = I₃          — position error grows with velocity error
Block (1,2) = -C[Δv×]     — velocity error couples to attitude error ("specific force coupling")
Block (1,3) = C            — velocity error grows with accel bias
Block (2,4) = -C           — attitude error grows with gyro bias
```

### G matrix — 15×12 block structure

```
     ┌                      ┐
     │ 0    0    0    0  │
     │ C    0    0    0  │   ← accel noise enters velocity channel
G =  │ 0   -C    0    0  │   ← gyro noise enters attitude channel
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
Q_d = diag(
    σ_a²·I₃·dt²,       — accelerometer white noise
    σ_g²·I₃·dt²,       — gyroscope white noise
    σ_ba²·I₃·dt,       — accel bias random walk
    σ_bg²·I₃·dt        — gyro bias random walk
)
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
P = (I - KH) @ P                   (Joseph form)
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
    │    P = (I - KH) @ P                       │
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
- What is the physical meaning of the `-C[Δv×]` block in the F matrix?
- If GNSS is lost for an extended period (e.g. tunnel), how does P evolve?
  What is the drift characteristic of pure INS?
- Why are b_a and b_g modelled as random walks (ḃ = n) rather than
  first-order Gauss-Markov (ḃ = -βb + n)?
- What advantage does quaternion attitude parameterisation have over
  Euler angles?

## Common Mistakes

- Reusing a stale DCM C instead of recomputing it each IMU step
- Adding the attitude error δθ to the quaternion instead of quaternion-multiplying
- Forgetting the minus sign on `-C` in the F matrix
- Using Δv directly as acceleration and forgetting to account for dt
- Subtracting NED position from WGS84 lat/lon/alt without coordinate conversion
- Getting the gravity sign wrong: +g (down) in NED vs −g (up) in ENU
