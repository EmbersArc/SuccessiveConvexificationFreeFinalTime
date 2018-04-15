from sympy import *
import numpy as np
import cvxpy as cvx


def skew(v):
    return Matrix([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def dir_cosine(q):
    return Matrix([
        [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
        [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
        [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
    ])


def omega(w):
    return Matrix([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ])


class Model_6DoF:
    n_x = 14
    n_u = 3

    # Mass
    m_wet = 2  # 33000kg
    m_dry = 1  # 26000kg

    # Flight time guess
    t_f_guess = 3  # 10s

    # State constraints
    r_I_init = np.array((3., 2., 1))  # 2000m, 200m, 200m
    v_I_init = np.array((0, -1, -1))  # -300m/s, 50m/s, 50m/s
    q_B_I_init = np.array((1.0, 0.0, 0.0, 0.0))
    w_B_init = np.array((0., 0., 0.))

    r_I_final = np.array((0., 0., 0.))
    v_I_final = np.array((-1e-1, 0., 0.))
    q_B_I_final = np.array((1.0, 0.0, 0.0, 0.0))
    w_B_final = np.array((0., 0., 0.))

    x_init = np.concatenate(((m_wet,), r_I_init, v_I_init, q_B_I_init, w_B_init))
    x_final = np.concatenate(((m_dry,), r_I_final, v_I_final, q_B_I_final, w_B_final))

    w_B_max = np.deg2rad(60)

    # Angles
    cos_delta_max = np.cos(np.deg2rad(15))
    cos_theta_max = np.cos(np.deg2rad(90))
    tan_gamma_gs = np.tan(np.deg2rad(20))

    # Thrust limits
    T_max = 5  # 845000 kg*m/s^2
    T_min = 0.3

    # Angular moment of inertia
    J_B = np.diag((1e-2, 1e-2, 1e-2))  # 4500000kg*m^2, 4500000kg*m^2, 100000kg*m^2

    # Vector from thrust point to CoM
    r_T_B = np.array((-1e-2, 0., 0.))  # -20m

    # Gravity
    g_I = np.array((-1, 0., 0.))  # -9.81 m/s^2

    # Fuel consumption
    alpha_m = 0.005  # 1 / (282s * 9.81m/s^2))

    def get_equations(self):
        f = zeros(14, 1)

        x = Matrix(symbols('m rx ry rz vx vy vz q0 q1 q2 q3 wx wy wz', real=True))
        u = Matrix(symbols('ux uy uz', real=True))

        g_I = Matrix(self.g_I)
        r_T_B = Matrix(self.r_T_B)
        J_B = Matrix(self.J_B)

        C_B_I = dir_cosine(x[7:11, 0])
        C_I_B = C_B_I.transpose()

        f[0, 0] = - self.alpha_m * u.norm()
        f[1:4, 0] = x[4:7, 0]
        f[4:7, 0] = 1 / x[0, 0] * C_I_B * u + g_I
        f[7:11, 0] = 1 / 2 * omega(x[11:14, 0]) * x[7: 11, 0]
        f[11:14, 0] = J_B ** -1 * (skew(r_T_B) * u - skew(x[11:14, 0]) * J_B * x[11:14, 0])

        A = f.jacobian(x)
        B = f.jacobian(u)

        f_lambda = lambdify((x, u), f, "numpy")
        A_lambda = lambdify((x, u), A, "numpy")
        B_lambda = lambdify((x, u), B, "numpy")

        return f_lambda, A_lambda, B_lambda

    def initialize(self, X, U):
        print("Starting Initialization.")

        K = X.shape[1]

        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K
            m_k = (alpha1 * self.x_init[0] + alpha2 * self.x_final[0],)
            r_I_k = alpha1 * self.x_init[1:4] + alpha2 * self.x_final[1:4]
            v_I_k = alpha1 * self.x_init[4:7] + alpha2 * self.x_final[4:7]
            q_B_I_k = np.array((1.0, 0.0, 0.0, 0.0))
            w_B_k = alpha1 * self.x_init[11:14] + alpha2 * self.x_final[11:14]

            X[:, k] = np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k))
            U[:, k] = m_k * -self.g_I

        print("Initialization finished.")

    def get_constraints(self, X_, U_, X_last_, U_last_):
        K = X_.shape[1]

        # Boundary conditions:
        constraints = [
            X_[0, 0] == self.x_init[0],
            X_[1:4, 0] == self.x_init[1:4],
            X_[4:7, 0] == self.x_init[4:7],
            # X_[0, 7:11] == x_init[7:11],  # initial attitude is free
            X_[11:14, 0] == self.x_init[11:14],

            # X_[0, 0] == x_final[0], # final mass is free
            X_[1:, K - 1] == self.x_final[1:],

            U_[1:3, K - 1] == 0,
        ]

        constraints += [
            # State constraints:
            X_[0, :] >= self.m_dry,
            cvx.norm(X_[2: 4, :], axis=0) <= X_[1, :] / self.tan_gamma_gs,
            cvx.norm(X_[9:11, :], axis=0) <= np.sqrt((1 - self.cos_theta_max) / 2),
            cvx.norm(X_[11: 14, :], axis=0) <= self.w_B_max,

            # Control constraints:
            cvx.norm(U_, axis=0) <= U_[0, :] / self.cos_delta_max,
            cvx.norm(U_, axis=0) <= self.T_max,
        ]
        constraints += [
            self.T_min <= U_last_[:, k] / cvx.norm(U_last_[:, k]) * U_[:, k] for k in range(K)
        ]

        return constraints
