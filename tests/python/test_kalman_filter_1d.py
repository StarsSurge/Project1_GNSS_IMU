import numpy as np

from gnss_imu import KalmanFilter1D, create_constant_velocity_filter


def test_one_step_matches_reference_result():
    kf = create_constant_velocity_filter(dt=1.0)
    z = np.array([[1.2], [0.9]])

    kf.predict()
    x_upd, P_upd, K, residual = kf.update(z)

    expected_x = np.array([[1.02387807], [0.96858594]])

    assert np.allclose(x_upd, expected_x)
    assert P_upd.shape == (2, 2)
    assert K.shape == (2, 2)
    assert residual.shape == (2, 1)


def test_step_runs_predict_and_update():
    kf = create_constant_velocity_filter(dt=1.0)
    z = np.array([[1.2], [0.9]])

    x_upd, _, _, _ = kf.step(z)

    assert np.allclose(x_upd, [[1.02387807], [0.96858594]])


def test_rejects_wrong_measurement_shape():
    kf = create_constant_velocity_filter(dt=1.0)
    kf.predict()

    with np.testing.assert_raises(ValueError):
        kf.update(np.array([1.2, 0.9]))


def test_rejects_wrong_state_shape():
    with np.testing.assert_raises(ValueError):
        KalmanFilter1D(
            x=np.array([0.0, 1.0]),
            P=np.eye(2),
            F=np.eye(2),
            H=np.eye(2),
            Q=np.eye(2),
            R=np.eye(2),
        )
