"""Educational 1D constant-velocity Kalman filter.

The state is:

    x = [position, velocity]^T

This module keeps the implementation small and explicit so the equations map
directly to the standard Kalman filter prediction and update formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass
class KalmanFilter1D:
    """1D Kalman filter with position and velocity in the state."""

    x: Array
    P: Array
    F: Array
    H: Array
    Q: Array
    R: Array

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        self.F = np.asarray(self.F, dtype=float)
        self.H = np.asarray(self.H, dtype=float)
        self.Q = np.asarray(self.Q, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self._check_shapes()

    def predict(self) -> tuple[Array, Array]:
        """Propagate state and covariance with the motion model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """Correct state and covariance with one position/velocity measurement.

        Returns:
            Updated state, updated covariance, Kalman gain, and residual.
        """
        z = np.asarray(z, dtype=float)
        if z.shape != (2, 1):
            raise ValueError(f"Expected z shape (2, 1), got {z.shape}")

        residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        B = self.P @ self.H.T
        K = np.linalg.solve(S.T, B.T).T

        self.x = self.x + K @ residual

        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x, self.P, K, residual

    def step(self, z: Array) -> tuple[Array, Array, Array, Array]:
        """Run one predict-update cycle."""
        self.predict()
        return self.update(z)

    def _check_shapes(self) -> None:
        expected_shapes = {
            "x": (2, 1),
            "P": (2, 2),
            "F": (2, 2),
            "H": (2, 2),
            "Q": (2, 2),
            "R": (2, 2),
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(self, name)
            if value.shape != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got {value.shape}"
                )


def create_constant_velocity_filter(
    dt: float = 1.0,
    initial_position: float = 0.0,
    initial_velocity: float = 1.0,
    initial_covariance: float = 1.0,
    process_noise_position: float = 0.1,
    process_noise_velocity: float = 0.1,
    measurement_noise_position: float = 4.0,
    measurement_noise_velocity: float = 1.0,
) -> KalmanFilter1D:
    """Create the same 1D filter used in the introductory example."""
    x = np.array([[initial_position], [initial_velocity]])
    P = np.eye(2) * initial_covariance
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = np.diag([process_noise_position, process_noise_velocity])
    R = np.diag([measurement_noise_position, measurement_noise_velocity])

    return KalmanFilter1D(x=x, P=P, F=F, H=H, Q=Q, R=R)
