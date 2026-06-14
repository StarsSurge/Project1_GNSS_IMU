# Data

Use this directory for small synthetic datasets or instructions for downloading public datasets.

Guidelines:

- Mark synthetic data clearly.
- Do not commit large raw datasets.
- Put large local-only files under `data/raw/`, which is ignored by Git.
- Document dataset source, license, coordinate frame, and sensor units.

For a real IMU Allan analysis, document at least:

- timestamp column, unit, clock source, and synchronization method
- whether values are rates or increments
- gyro and accelerometer units and axis/frame convention
- sample rate, recording duration, warm-up time, and temperature range
- sensor model, firmware, configured bandwidth, and output data rate
- stationary mounting conditions and known vibration sources
- dropped-packet, saturation, motion, and outlier handling

The current repository does not include a real stationary IMU log. Place local
logs under `data/raw/` and analyze them with
`python/examples/analyze_imu_allan.py`.
