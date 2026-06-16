# GNSS/IMU Fusion Basics

## Problem Definition

GNSS/IMU integration estimates a moving platform's navigation state by combining:

- IMU measurements: high-rate acceleration and angular velocity
- GNSS measurements: lower-rate absolute position, velocity, or pseudorange-derived observations

The IMU provides smooth short-term motion propagation but drifts over time because of integration error and sensor bias. GNSS provides global observability but can be noisy, low-rate, delayed, or unavailable. Fusion uses the complementary properties of both sensors.

For robotics localization, this same idea appears in mobile robots, UAVs, autonomous driving, and SLAM systems. IMU propagation gives local motion continuity, while GNSS or other exteroceptive sensors constrain long-term drift.

## Common State Variables

A typical 3D navigation state is:

```text
x = [p, v, q, b_a, b_g]
```

where:

- `p`: position, 3 x 1
- `v`: velocity, 3 x 1
- `q`: attitude quaternion, 4 x 1 nominal representation
- `b_a`: accelerometer bias, 3 x 1
- `b_g`: gyroscope bias, 3 x 1

The nominal state has dimension 16 if the quaternion has 4 parameters:

```text
dim(x_nominal) = 3 + 3 + 4 + 3 + 3 = 16
```

In an error-state Kalman filter, the attitude error is usually represented with a minimal 3D small-angle vector:

```text
delta_x = [delta_p, delta_v, delta_theta, delta_b_a, delta_b_g]
```

The error state dimension is:

```text
dim(delta_x) = 3 + 3 + 3 + 3 + 3 = 15
```

This distinction is a frequent interview topic: the nominal state may use a quaternion, while the filter covariance is maintained on a minimal local error state.

## IMU Measurement Model

The accelerometer and gyroscope measurements can be modeled as:

```text
a_m = a_true + b_a + n_a
w_m = w_true + b_g + n_g
```

where:

- `a_m`: measured specific force
- `w_m`: measured angular velocity
- `b_a`, `b_g`: slowly varying biases
- `n_a`, `n_g`: white measurement noise

A simple bias random-walk model is:

```text
b_a(k+1) = b_a(k) + n_ba
b_g(k+1) = b_g(k) + n_bg
```

## GNSS Measurement Model

For a first educational loosely coupled model, assume GNSS directly provides position:

```text
z_gnss = p + n_gnss
```

The corresponding measurement matrix for an error-state filter is:

```text
H = [I_3, 0_3, 0_3, 0_3, 0_3]
```

This is a simple starting point. Later, tighter GNSS integration can use pseudorange, Doppler, carrier phase, clock bias, and clock drift.

## Fusion Intuition

The prediction step is driven by IMU integration:

```text
x_hat(k+1) = f(x_hat(k), u_imu(k), dt)
P(k+1) = F P(k) F^T + G Q G^T
```

The update step is driven by GNSS:

```text
y = z - h(x_hat)
S = H P H^T + R
K = P H^T S^-1
delta_x = K y
P = (I - K H) P       (simplified covariance update)
```

Engineering implementations often use the Joseph form
``A P Aᵀ + K R Kᵀ``, ``A = I - K H``, for better numerical robustness.

For ESKF, `delta_x` is injected into the nominal state, then the error state is reset to zero.

## Derivation Outline

1. Define navigation frames and sensor frames.
2. Write continuous-time IMU-driven nominal dynamics.
3. Define small error variables around the nominal state.
4. Linearize the error dynamics to obtain `F` and `G`.
5. Discretize the error dynamics for implementation.
6. Define the GNSS measurement residual.
7. Use the Kalman update to correct the error state.
8. Inject the correction into position, velocity, attitude, and biases.

## Implementation Details to Track

- Units: meters, seconds, radians.
- Gravity convention: document whether `z` is up or down.
- Frame convention: ENU, NED, ECEF, or local tangent frame.
- Quaternion convention: scalar-first or scalar-last.
- Sensor timestamps: IMU is high-rate, GNSS is low-rate.
- Noise parameters: measurement noise versus bias random walk.
- Reproducibility: synthetic data should use a fixed random seed.

## Common Interview Questions

- Why does pure IMU dead reckoning drift quickly?
- Why is GNSS useful even if it is noisy and low-rate?
- What is the difference between EKF and ESKF?
- Why is attitude error often represented as a 3D vector instead of a 4D quaternion?
- What does covariance represent physically?
- How do accelerometer and gyroscope bias affect position error?
- What happens during a GNSS outage?
- How would you check whether a filter is consistent?

## Common Mistakes

- Mixing degrees and radians.
- Mixing ENU and NED frame conventions.
- Forgetting gravity compensation.
- Treating accelerometer output as world-frame acceleration.
- Updating quaternion attitude without normalization.
- Using measurement noise where process noise is required.
- Ignoring timestamp alignment between GNSS and IMU.
- Reporting only final error without plotting drift and correction behavior.
