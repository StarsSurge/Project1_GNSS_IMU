# AGENTS.md

## Project Goal

This repository is part of my robotics / embodied AI job-preparation learning series.

The goal is not only to make the code run, but to build interview-ready algorithm projects with:

1. Clear mathematical understanding
2. Clean implementation from scratch when appropriate
3. Unit tests or reproducible verification
4. Visualizations and experiment results
5. Well-written README and learning notes
6. Code quality suitable for GitHub portfolio

## My Background

I am a GNSS algorithm engineer transitioning toward robotics localization, multi-sensor fusion, SLAM, and embodied AI.

My target roles are:

* Robotics localization algorithm engineer
* Multi-sensor fusion engineer
* SLAM / perception engineer
* Embodied AI / robot learning engineer

Therefore, prioritize explanations and implementations that connect to:

* GNSS / IMU integration
* Kalman filter, EKF, ESKF
* LiDAR / camera / IMU fusion
* ROS / RViz / rosbag
* Robotics learning and manipulation

## Working Style

For every task, follow this workflow:

1. First inspect the existing repository structure.
2. Summarize what already exists.
3. Propose a small implementation plan.
4. Implement only the current requested step.
5. Add minimal tests, demos, or verification scripts.
6. Run the relevant tests or commands if possible.
7. Summarize:

   * What changed
   * How to run it
   * What I should study next
   * Any limitations or TODOs

## Learning Requirement

Do not only generate code.

For every important algorithm, also create or update learning notes under `docs/`.

The notes should include:

* Problem definition
* State variables and dimensions
* Mathematical formulas
* Derivation outline
* Implementation details
* Common interview questions
* Common mistakes

Use clear explanations suitable for someone with a GNSS / robotics background.

## Code Requirements

Prefer simple, readable, educational code over overly abstract code.

Use:

* Python for algorithm prototypes
* NumPy for from-scratch ML / filtering algorithms
* Matplotlib for visualization
* C++17, Eigen, Sophus, Ceres, PCL, OpenCV, and ROS when requested

Do not introduce heavy dependencies unless necessary.

When adding dependencies, update the README or environment file.

## Repository Quality Rules

Keep the repository organized.

Suggested structure:

* `docs/` for learning notes and derivations
* `python/` for Python prototypes
* `cpp/` for C++ implementations
* `ros/` for ROS packages, launch files, and RViz configs
* `tests/` for verification scripts
* `data/` for small sample data or download instructions
* `results/` for generated plots, logs, or figures

Do not place everything in one large script.

## Safety and Scope Rules

Do not delete existing files unless explicitly asked.

Do not perform large refactors unless explicitly requested.

Do not hide failed tests.

Do not fake results.

If data is missing, create a small synthetic demo dataset and clearly mark it as synthetic.

If there are multiple possible implementations, choose the simplest educational version first.

## Output Style

At the end of each task, provide:

1. Summary of changes
2. Commands to run
3. Expected output
4. What I should understand from this step
5. Recommended next task
