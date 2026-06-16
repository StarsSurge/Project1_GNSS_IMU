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

## Algorithm Teaching Standard

Algorithm notes must build a bridge from intuitive understanding to
mathematical behavior. Do not write documentation as a list of definitions,
formulas, slope values, implementation steps, or interview answers that the
reader is expected to memorize.

For every core algorithm concept, use the following teaching sequence whenever
it is applicable:

1. **Motivating question**
   State the concrete question the concept answers and why it appears in a
   robotics or sensor-processing system.
2. **Physical or everyday intuition**
   Introduce a simple physical experiment, geometric picture, or everyday
   analogy. State where the analogy stops being accurate.
3. **Minimal scalar or low-dimensional example**
   Use the smallest numerical or one-dimensional example that exposes the
   mechanism before introducing matrix notation.
4. **Definitions and conventions**
   Define every physical quantity, state, unit, frame, timestamp meaning,
   sign convention, and stochastic assumption before using it in a formula.
5. **Step-by-step derivation**
   Explain where each important term, sign, coefficient, exponent, and matrix
   block comes from. Do not present a standard formula without showing the
   reasoning needed to reconstruct it.
6. **Order-of-magnitude and scaling behavior**
   Explain how the result changes with time, sample rate, noise level, motion
   amplitude, or averaging length. Derive important power laws and limiting
   cases instead of only stating them.
7. **Geometric or physical interpretation**
   Translate the final equation back into a physical statement. Explain what
   each term does and what observable behavior it produces.
8. **Code mapping**
   Map the mathematical objects and assumptions to specific functions,
   variables, array dimensions, and coordinate transformations in the code.
9. **Verification that can fail**
   Provide a numerical example, unit test, reference solution, limiting case,
   or visualization that would expose an incorrect sign, unit, frame, or
   implementation.
10. **Real-world boundary**
    State the assumptions, failure modes, omitted effects, data requirements,
    and the gap between the educational model and deployment use.
11. **Self-check questions**
    Include questions that require the reader to explain or re-derive the
    result, not merely repeat a definition.

Important teaching rules:

* Explain **why** a recognizable result occurs. For example, do not only say
  that white noise has an Allan-deviation slope of `-1/2`; derive it from the
  variance of an average decreasing as `1/m`.
* Distinguish concepts that use similar words but represent different physical
  objects, such as gravity versus gravitation, acceleration versus specific
  force, reference frame versus coordinate representation, rate versus
  increment, noise density versus discrete standard deviation, and
  correlation time versus Allan cluster time.
* Introduce matrix and stochastic notation only after the reader understands
  the scalar mechanism it generalizes.
* Use equations as part of an argument. Every important equation should have
  a sentence before it explaining why it is introduced and a sentence after
  it interpreting the result.
* Include units and dimensional checks in derivations. A formula whose units
  cannot be explained is not sufficiently taught or verified.
* Connect local formulas to the complete data flow. Explain which processing
  layer owns the operation and why changing the order would be incorrect.
* When different references use different conventions, show the alternatives
  and state which convention this repository uses.
* Synthetic examples must reveal a mechanism or catch an error; they must not
  exist only to produce a visually pleasing result.

A good note should allow the learner to reconstruct the main result after
forgetting the final formula. If the reader can only remember a table or a
coefficient, the explanation is incomplete.

## Code Requirements

Prefer simple, readable, educational code over overly abstract code.

Simplicity must come from clear design, not from omitting behavior required
by realistic inputs, numerical stability, validation, or system integration.

Use:

* Python for algorithm prototypes
* NumPy for from-scratch ML / filtering algorithms
* Matplotlib for visualization
* C++17, Eigen, Sophus, Ceres, PCL, OpenCV, and ROS when requested

Do not introduce heavy dependencies unless necessary.

When adding dependencies, update the README or environment file.

## Practical Engineering Standard

Although this is a learning repository, its engineering baseline is
real-world usability. Treat each prototype as a step toward an implementation
that could process real sensor data, integrate with a robotics system, and be
defended in a production-oriented technical review.

For every algorithm or experiment:

* Start from the real problem definition, sensor interface, units, coordinate
  frames, timestamps, noise assumptions, and expected operating conditions.
* Do not simplify away essential engineering issues merely to make a demo
  work. Consider missing or irregular data, initialization, numerical
  stability, observability, calibration, synchronization, outliers, failure
  modes, and computational cost when they are relevant.
* A minimal implementation may intentionally defer advanced features, but
  every simplification must be explicit in code or documentation, justified,
  bounded by stated assumptions, and listed as a limitation or TODO.
* Never present a synthetic demonstration, idealized simulation, hard-coded
  parameter, or simplified model as evidence of real-world performance.
* Prefer algorithms and data flows that can later accept real logs without
  being rewritten from scratch. Keep parsing, preprocessing, estimation,
  evaluation, and visualization responsibilities separated.
* Validate mathematical conventions against the implementation, including
  units, frame directions, quaternion multiplication, perturbation convention,
  continuous/discrete noise definitions, and covariance dimensions.
* Add realistic boundary and failure tests in addition to nominal synthetic
  tests. When real data is unavailable, document the missing validation rather
  than lowering the acceptance standard.
* Use quantitative acceptance criteria where possible, such as error metrics,
  consistency checks, timing checks, residual statistics, or comparison with
  an independent reference implementation.
* Do not stop at a visually plausible plot or a test that only proves the code
  runs. Check whether the result is physically meaningful and whether the
  evaluation can reveal an incorrect implementation.
* Before calling a feature complete, distinguish clearly among educational
  prototype, verified MVP, and deployment-ready implementation. Do not use
  these labels interchangeably.

The default progression is:

1. Derive and document the complete real-world model.
2. Define the smallest justified MVP subset and its assumptions.
3. Implement the MVP with production-compatible interfaces and conventions.
4. Verify it with deterministic tests, numerical checks, and synthetic data.
5. Validate it on documented real data when available.
6. Record the remaining gap to deployment, including robustness, performance,
   integration, and field-test requirements.

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

If there are multiple possible implementations, choose the simplest version
that remains mathematically correct, testable, and compatible with the
real-world target. Do not choose a shortcut that hides required behavior or
would force the core algorithm to be rewritten for real data.

## Output Style

At the end of each task, provide:

1. Summary of changes
2. Commands to run
3. Expected output
4. What I should understand from this step
5. Recommended next task
