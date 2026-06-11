# Project Overview

## Goal

This project builds a small but interview-ready GNSS/IMU integration portfolio. The emphasis is on understanding the math, implementing the core ideas clearly, and verifying behavior with synthetic experiments.

The intended direction is robotics localization rather than only traditional navigation. That means each topic should eventually connect to robot state estimation, multi-sensor fusion, SLAM front-end/back-end ideas, and ROS-style visualization.

## Phase Plan

### Phase 1: Project Structure and Foundations

Create the repository skeleton, document the learning path, and prepare a lightweight Python environment.

Expected outputs:

- `README.md`
- `requirements.txt`
- initial notes under `docs/`
- Python package skeleton under `python/`
- test and result directories

### Phase 2: Synthetic Motion and Sensor Data

Create a simple 2D or 3D synthetic trajectory and generate ideal/noisy GNSS and IMU measurements.

Key learning points:

- position, velocity, attitude, bias, and noise
- sampling rate mismatch between GNSS and IMU
- reproducible random seeds
- plots for trajectory and sensor measurements

### Phase 3: Basic Kalman Filter

Implement a small linear Kalman filter before moving to nonlinear navigation.

Key learning points:

- prediction and update equations
- covariance propagation
- process noise and measurement noise
- innovation and consistency checks

### Phase 4: Loosely Coupled GNSS/IMU Fusion

Fuse GNSS position measurements with IMU propagation in an educational EKF or ESKF.

Key learning points:

- navigation state definition
- error-state representation
- discrete-time propagation
- GNSS measurement update
- bias estimation

### Phase 5: C++ and Robotics Integration

Port selected modules to C++17 and connect the results to robotics tooling.

Key learning points:

- Eigen matrix implementation
- CMake project layout
- ROS messages, bags, and RViz visualization
- comparison between Python prototype and C++ implementation

## Suggested Repository Structure

```text
Project1_GNSS_IMU/
|-- docs/
|-- python/
|   |-- gnss_imu/
|   `-- examples/
|-- cpp/
|-- tests/
|-- data/
`-- results/
```

The current first phase creates only the directories needed immediately. C++ and ROS folders can be added when their environment is ready.

## Engineering Principles

- Keep the first implementation small and readable.
- Prefer synthetic data until the estimator behavior is understood.
- Verify dimensions and simple limiting cases with tests.
- Plot intermediate results; do not rely only on final RMSE.
- Document assumptions such as frame convention, units, gravity direction, and noise model.

## Interview Orientation

Each algorithm step should answer these questions:

- What is the state?
- What is the process model?
- What is the measurement model?
- What are the noise assumptions?
- Which quantities are directly observed and which are inferred?
- What failure modes appear when GNSS is weak or IMU bias is large?
