# Project1_GNSS_IMU

Educational GNSS/IMU integration project for robotics localization, multi-sensor fusion, and SLAM interview preparation.

Chinese version: [README.zh-CN.md](README.zh-CN.md)

The repository is intentionally organized as a learning portfolio:

- math notes in `docs/`
- Python prototypes in `python/`
- reproducible checks in `tests/`
- small synthetic or documented data in `data/`
- generated plots and experiment outputs in `results/`

The first phase focuses on a readable Python prototype before moving to C++17 and robotics tooling.

## Learning Roadmap

1. Coordinate frames and navigation states
2. Strapdown IMU mechanization basics
3. Kalman filter and extended Kalman filter
4. Error-state Kalman filter for GNSS/IMU fusion
5. Synthetic trajectory generation and visualization
6. C++17 implementation with Eigen
7. ROS/RViz integration for robotics-style demos

## Windows 11 Environment

Check available tools:

```powershell
git --version
python --version
cmake --version
```

If Python is not available, install it and reopen PowerShell:

```powershell
winget install Python.Python.3.12
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:MPLCONFIGDIR = "$PWD\.matplotlib-cache"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run future Python tests:

```powershell
pytest
```

## Current Status

The current Python phase includes:

- an educational 1D constant-velocity Kalman filter
- a general N-dimensional linear Kalman filter
- a range-bearing Extended Kalman Filter
- reproducible unit tests and Jacobian checks
- runnable visual demos and GNSS/IMU/ESKF learning notes
- a verified-MVP Allan analysis module for named-column real IMU CSV logs,
  including timestamp validation, rate/increment conversion, and auditable
  CSV/JSON/PNG outputs
- a production-oriented 15-state GNSS/IMU loose ESKF baseline with WGS-84
  mechanization, two-sample corrections, lever arm, NIS gating, Joseph updates,
  calibrated IMU matrices, fixed-lag delayed GNSS replay, constant time-offset
  scanning, and dataset1 replay outputs

```powershell
$env:PYTHONPATH = "$PWD\python"
python python\examples\demo_1d_kalman_filter.py
python python\examples\demo_kalman_filter.py
python python\examples\demo_extended_kalman_filter.py
python python\examples\run_dataset1_eskf.py --duration-s 60
python python\examples\calibrate_dataset1_gnss_time_offset.py --duration-s 30
python -m pytest tests
```

The Allan CSV tool is not deployment-ready: the repository still lacks a
traceable real stationary IMU dataset, automatic stationarity/temperature/
saturation checks, and equivalent-degrees-of-freedom confidence intervals.

The ESKF is a verified MVP / production-oriented baseline, not deployment-ready.
Online alignment, clock-drift estimation, temperature models, 21-state online
scale-factor estimation, and real-time deployment validation remain open.

## Notes for Generated Outputs

Files in `results/` should be treated as reproducible experiment outputs. Keep important figures small and documented, and avoid committing large generated files unless they are needed for explanation.
