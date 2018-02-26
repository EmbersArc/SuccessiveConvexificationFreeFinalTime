import numpy as np

# Trajectory points
Kn = 50
iterations = 15

# Mass
m_wet = 2.0
m_dry = 1.0

# Flight time guess
t_f_guess = 4.

# Weight constants
w_nu = 1e5
w_delta = 1e-3
w_delta_sigma = 1e-1

# State constraints
r_I_init = np.array((4., 4., 0.))
r_I_final = np.array((0., 0., 0.))
v_I_init = np.array((-1e-1, 0., 0.))
v_I_final = np.array((0., 0., 0.))
q_B_I_final = np.array((1.0, 0.0, 0.0, 0.0))
w_B_init = np.array((0., 0., 0.))
w_B_final = np.array((0., 0., 0.))
w_B_max = 60.

# Gravity
g_I = np.array((0., 0., -1.))
g1, g2, g3 = g_I

# Angles
cos_delta_max = np.cos(np.deg2rad(20))
cos_theta_max = np.cos(np.deg2rad(90))
tan_gamma_gs = np.tan(np.deg2rad(20))

# Angular momentum
J_B_I = np.array((1e-2, 1e-2, 1e-2))
J_B1, J_B2, J_B3 = J_B_I

# Vector from thrust point to CoM
r_T_B = np.array((-1e-2, 0., 0.))
r_T_B1, r_T_B2, r_T_B3 = r_T_B

# Thrust limits
T_min = 0.3
T_max = 5.0

# Linearized state matrices

# A Matrix
def A(x, u, sigma_hat):
    m, r1, r2, r3, v1, v2, v3, q1, q2, q3, q4, w1, w2, w3 = x
    u1, u2, u3 = u

    return sigma_hat * np.array((
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        ((u1 * (2 * q3 ** 2 + 2 * q4 ** 2 - 1)) / m ** 2 - (u2 * (2 * q1 * q4 + 2 * q2 * q3)) / m ** 2 + (
                u3 * (2 * q1 * q3 - 2 * q2 * q4)) / m ** 2, 0, 0, 0, 0, 0, 0, (2 * q4 * u2) / m - (2 * q3 * u3) / m,
         (2 * q3 * u2) / m + (2 * q4 * u3) / m, (2 * q2 * u2) / m - (2 * q1 * u3) / m - (4 * q3 * u1) / m,
         (2 * q1 * u2) / m + (2 * q2 * u3) / m - (4 * q4 * u1) / m, 0, 0, 0),
        ((u2 * (2 * q2 ** 2 + 2 * q4 ** 2 - 1)) / m ** 2 + (u1 * (2 * q1 * q4 - 2 * q2 * q3)) / m ** 2 - (
                u3 * (2 * q1 * q2 + 2 * q3 * q4)) / m ** 2, 0, 0, 0, 0, 0, 0, (2 * q2 * u3) / m - (2 * q4 * u1) / m,
         (2 * q1 * u3) / m - (4 * q2 * u2) / m + (2 * q3 * u1) / m, (2 * q2 * u1) / m + (2 * q4 * u3) / m,
         (2 * q3 * u3) / m - (2 * q1 * u1) / m - (4 * q4 * u2) / m, 0, 0, 0),
        ((u3 * (2 * q2 ** 2 + 2 * q3 ** 2 - 1)) / m ** 2 - (u1 * (2 * q1 * q3 + 2 * q2 * q4)) / m ** 2 + (
                u2 * (2 * q1 * q2 - 2 * q3 * q4)) / m ** 2, 0, 0, 0, 0, 0, 0, (2 * q3 * u1) / m - (2 * q2 * u2) / m,
         (2 * q4 * u1) / m - (4 * q2 * u3) / m - (2 * q1 * u2) / m,
         (2 * q1 * u1) / m - (4 * q3 * u3) / m + (2 * q4 * u2) / m, (2 * q2 * u1) / m + (2 * q3 * u2) / m, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, -w1 / 2, -w2 / 2, -w3 / 2, -q2 / 2, -q3 / 2, -q4 / 2),
        (0, 0, 0, 0, 0, 0, 0, w1 / 2, 0, w3 / 2, -w2 / 2, q1 / 2, -q4 / 2, q3 / 2),
        (0, 0, 0, 0, 0, 0, 0, w2 / 2, -w3 / 2, 0, w1 / 2, q4 / 2, q1 / 2, -q2 / 2),
        (0, 0, 0, 0, 0, 0, 0, w3 / 2, w2 / 2, -w1 / 2, 0, -q3 / 2, q2 / 2, q1 / 2),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (J_B2 * w3 - J_B3 * w3) / J_B1, (J_B2 * w2 - J_B3 * w2) / J_B1),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -(J_B1 * w3 - J_B3 * w3) / J_B2, 0, -(J_B1 * w1 - J_B3 * w1) / J_B2),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (J_B1 * w2 - J_B2 * w2) / J_B3, (J_B1 * w1 - J_B2 * w1) / J_B3, 0)))


# B Matrix
def B(x, u, sigma_hat):
    m, r1, r2, r3, v1, v2, v3, q1, q2, q3, q4, w1, w2, w3 = x
    u1, u2, u3 = u

    return sigma_hat * np.array(((-(abs(u1) * np.sign(u1)) / (abs(u1) ** 2 + abs(u2) ** 2 + abs(u3) ** 2) ** (1 / 2),
                                  -(abs(u2) * np.sign(u2)) / (abs(u1) ** 2 + abs(u2) ** 2 + abs(u3) ** 2) ** (1 / 2),
                                  -(abs(u3) * np.sign(u3)) / (abs(u1) ** 2 + abs(u2) ** 2 + abs(u3) ** 2) ** (1 / 2)),
                                 (0, 0, 0),
                                 (0, 0, 0),
                                 (0, 0, 0),
                                 (-(2 * q3 ** 2 + 2 * q4 ** 2 - 1) / m, (2 * q1 * q4 + 2 * q2 * q3) / m,
                                  -(2 * q1 * q3 - 2 * q2 * q4) / m),
                                 (-(2 * q1 * q4 - 2 * q2 * q3) / m, -(2 * q2 ** 2 + 2 * q4 ** 2 - 1) / m,
                                  (2 * q1 * q2 + 2 * q3 * q4) / m),
                                 ((2 * q1 * q3 + 2 * q2 * q4) / m, -(2 * q1 * q2 - 2 * q3 * q4) / m,
                                  -(2 * q2 ** 2 + 2 * q3 ** 2 - 1) / m),
                                 (0, 0, 0),
                                 (0, 0, 0),
                                 (0, 0, 0),
                                 (0, 0, 0),
                                 (0, -r_T_B3 / J_B1, r_T_B2 / J_B1),
                                 (r_T_B3 / J_B2, 0, -r_T_B1 / J_B2),
                                 (-r_T_B2 / J_B3, r_T_B1 / J_B3, 0)))


# f Matrix (x_dot)
def f(x, u):
    m, r1, r2, r3, v1, v2, v3, q1, q2, q3, q4, w1, w2, w3 = x
    u1, u2, u3 = u

    return np.array((-(abs(u1) ** 2 + abs(u2) ** 2 + abs(u3) ** 2) ** (1 / 2),
                     v1,
                     v2,
                     v3,
                     g1 - (u1 * (2 * q3 ** 2 + 2 * q4 ** 2 - 1)) / m + (u2 * (2 * q1 * q4 + 2 * q2 * q3)) / m - (
                             u3 * (2 * q1 * q3 - 2 * q2 * q4)) / m,
                     g2 - (u2 * (2 * q2 ** 2 + 2 * q4 ** 2 - 1)) / m - (u1 * (2 * q1 * q4 - 2 * q2 * q3)) / m + (
                             u3 * (2 * q1 * q2 + 2 * q3 * q4)) / m,
                     g3 - (u3 * (2 * q2 ** 2 + 2 * q3 ** 2 - 1)) / m + (u1 * (2 * q1 * q3 + 2 * q2 * q4)) / m - (
                             u2 * (2 * q1 * q2 - 2 * q3 * q4)) / m,
                     - (q2 * w1) / 2 - (q3 * w2) / 2 - (q4 * w3) / 2,
                     (q1 * w1) / 2 + (q3 * w3) / 2 - (q4 * w2) / 2,
                     (q1 * w2) / 2 - (q2 * w3) / 2 + (q4 * w1) / 2,
                     (q1 * w3) / 2 + (q2 * w2) / 2 - (q3 * w1) / 2,
                     (r_T_B2 * u3 - r_T_B3 * u2 + J_B2 * w2 * w3 - J_B3 * w2 * w3) / J_B1,
                     -(r_T_B1 * u3 - r_T_B3 * u1 + J_B1 * w1 * w3 - J_B3 * w1 * w3) / J_B2,
                     (r_T_B1 * u2 - r_T_B2 * u1 + J_B1 * w1 * w2 - J_B2 * w1 * w2) / J_B3))