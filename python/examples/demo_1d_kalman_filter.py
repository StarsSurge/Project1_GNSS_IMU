import numpy as np

from gnss_imu import create_constant_velocity_filter


def print_matrix(name, value):
    print(f"{name}: shape={value.shape}")
    print(value)
    print()


def main():
    kf = create_constant_velocity_filter(dt=1.0)

    z = np.array([[1.2], [0.9]])

    print_matrix("x initial", kf.x)
    print_matrix("P initial", kf.P)

    x_pred, P_pred = kf.predict()
    print_matrix("x predicted", x_pred)
    print_matrix("P predicted", P_pred)

    x_upd, P_upd, K, residual = kf.update(z)
    print_matrix("z", z)
    print_matrix("residual", residual)
    print_matrix("K", K)
    print_matrix("x updated", x_upd)
    print_matrix("P updated", P_upd)


if __name__ == "__main__":
    main()
