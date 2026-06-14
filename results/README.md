# Results

Use this directory for generated plots, logs, and small experiment outputs.

Guidelines:

- Keep outputs reproducible from scripts.
- Document which command generated each important figure.
- Avoid committing large binary outputs.

`results/imu_allan/` is the default output directory for real-log Allan
analysis. It contains:

- `allan_deviation.csv`: machine-readable tau, pair counts, and axis results
- `analysis_metadata.json`: input conventions, sampling diagnostics, and limits
- `allan_deviation.png`: review plot

These files are analysis artifacts, not proof that the IMU model is suitable
for deployment. Preserve the source-log metadata and slope-fitting decisions
with any reported noise parameters.
