import numpy as np

# Trajectory points
K = 50
dt = 1 / (K - 1)
# Solver iterations
iterations = 30
# Numerical integration points
res = 25

# Mass
m_wet = 2.0
m_dry = 1.0

# Flight time guess
t_f_guess = 2.

# # Weight constants
w_nu = 1e5
w_delta = 1e-3
w_delta_sigma = 1e-1

# Exit conditions
nu_tol = 1e-10
delta_tol = 1e-3

# State constraints
r_I_init = np.array((4., 4., 4.))
v_I_init = np.array((-1, -4., 0.))
q_B_I_init = np.array((1.0, 0.0, 0.0, 0.0))
w_B_init = np.array((0., 0., 0.))

r_I_final = np.array((0., 0., 0.))
v_I_final = np.array((-1e-1, 0., 0.))
q_B_I_final = np.array((1.0, 0.0, 0.0, 0.0))
w_B_final = np.array((0., 0., 0.))

w_B_max = np.deg2rad(60)

# Angles
cos_delta_max = np.cos(np.deg2rad(20))
cos_theta_max = np.cos(np.deg2rad(90))
tan_gamma_gs = np.tan(np.deg2rad(20))

# Angular moment of inertia
J_B_I = np.array((1e-2, 1e-2, 1e-2))
J_B1, J_B2, J_B3 = J_B_I

# Vector from thrust point to CoM
r_T_B = np.array((-1e-2, 0., 0.))
r_T_B1, r_T_B2, r_T_B3 = r_T_B

# Gravity
g_I = np.array((-1., 0., 0.))
g_I1, g_I2, g_I3 = g_I

# Thrust limits
T_min = 1.0
T_max = 5.0

alpha_m = 0.01


# Linearized state matrices

# A Matrix
def A(x, u, sigma_hat):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = x
    u1, u2, u3 = u

    return sigma_hat * np.array((
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        ((u1 * (2 * x10 ** 2 + 2 * x11 ** 2 - 1)) / x1 ** 2 - (u2 * (2 * x8 * x11 + 2 * x9 * x10)) / x1 ** 2 + (
                u3 * (2 * x8 * x10 - 2 * x9 * x11)) / x1 ** 2, 0, 0, 0, 0, 0, 0, (2 * u2 * x11) / x1 - (
                 2 * u3 * x10) / x1, (2 * u2 * x10) / x1 + (2 * u3 * x11) / x1, (2 * u2 * x9) / x1 - (
                 4 * u1 * x10) / x1 - (2 * u3 * x8) / x1, (2 * u2 * x8) / x1 - (4 * u1 * x11) / x1 + (
                 2 * u3 * x9) / x1, 0, 0, 0),
        ((u2 * (2 * x9 ** 2 + 2 * x11 ** 2 - 1)) / x1 ** 2 + (u1 * (2 * x8 * x11 - 2 * x9 * x10)) / x1 ** 2 - (
                u3 * (2 * x8 * x9 + 2 * x10 * x11)) / x1 ** 2, 0, 0, 0, 0, 0, 0, (2 * u3 * x9) / x1 - (
                 2 * u1 * x11) / x1, (2 * u1 * x10) / x1 - (4 * u2 * x9) / x1 + (2 * u3 * x8) / x1, (
                 2 * u1 * x9) / x1 + (2 * u3 * x11) / x1, (2 * u3 * x10) / x1 - (4 * u2 * x11) / x1 - (
                 2 * u1 * x8) / x1, 0, 0, 0),
        ((u3 * (2 * x9 ** 2 + 2 * x10 ** 2 - 1)) / x1 ** 2 - (u1 * (2 * x8 * x10 + 2 * x9 * x11)) / x1 ** 2 + (
                u2 * (2 * x8 * x9 - 2 * x10 * x11)) / x1 ** 2, 0, 0, 0, 0, 0, 0, (2 * u1 * x10) / x1 - (
                 2 * u2 * x9) / x1, (2 * u1 * x11) / x1 - (2 * u2 * x8) / x1 - (4 * u3 * x9) / x1, (
                 2 * u1 * x8) / x1 + (2 * u2 * x11) / x1 - (4 * u3 * x10) / x1, (2 * u1 * x9) / x1 + (
                 2 * u2 * x10) / x1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, -x12 / 2, -x13 / 2, -x14 / 2, -x9 / 2, -x10 / 2, -x11 / 2),
        (0, 0, 0, 0, 0, 0, 0, x12 / 2, 0, x14 / 2, -x13 / 2, x8 / 2, -x11 / 2, x10 / 2),
        (0, 0, 0, 0, 0, 0, 0, x13 / 2, -x14 / 2, 0, x12 / 2, x11 / 2, x8 / 2, -x9 / 2),
        (0, 0, 0, 0, 0, 0, 0, x14 / 2, x13 / 2, -x12 / 2, 0, -x10 / 2, x9 / 2, x8 / 2),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (J_B2 * x14 - J_B3 * x14) / J_B1, (J_B2 * x13 - J_B3 * x13) / J_B1),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -(J_B1 * x14 - J_B3 * x14) / J_B2, 0, -(J_B1 * x12 - J_B3 * x12) / J_B2),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (J_B1 * x13 - J_B2 * x13) / J_B3, (J_B1 * x12 - J_B2 * x12) / J_B3, 0)

    ))


# B Matrix
def B(x, u, sigma_hat):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = x
    u1, u2, u3 = u

    return sigma_hat * np.array((
        (-(alpha_m * abs(u1) * np.sign(u1)) / np.linalg.norm(u),
         -(alpha_m * abs(u2) * np.sign(u2)) / np.linalg.norm(u),
         -(alpha_m * abs(u3) * np.sign(u3)) / np.linalg.norm(u)),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (-(2 * x10 ** 2 + 2 * x11 ** 2 - 1) / x1, (2 * x8 * x11 + 2 * x9 * x10) / x1,
         -(2 * x8 * x10 - 2 * x9 * x11) / x1),
        (-(2 * x8 * x11 - 2 * x9 * x10) / x1, -(2 * x9 ** 2 + 2 * x11 ** 2 - 1) / x1,
         (2 * x8 * x9 + 2 * x10 * x11) / x1),
        ((2 * x8 * x10 + 2 * x9 * x11) / x1, -(2 * x8 * x9 - 2 * x10 * x11) / x1,
         -(2 * x9 ** 2 + 2 * x10 ** 2 - 1) / x1),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, -r_T_B3 / J_B1, r_T_B2 / J_B1),
        (r_T_B3 / J_B2, 0, -r_T_B1 / J_B2),
        (-r_T_B2 / J_B3, r_T_B1 / J_B3, 0)
    ))


# f Matrix (x_dot)
def f(x, u):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = x
    u1, u2, u3 = u

    return np.array((
        -alpha_m * np.linalg.norm(u),
        x5,
        x6,
        x7,
        g_I1 - (u1 * (2 * x10 ** 2 + 2 * x11 ** 2 - 1)) / x1 + (u2 * (2 * x8 * x11 + 2 * x9 * x10)) / x1 - (
                u3 * (2 * x8 * x10 - 2 * x9 * x11)) / x1,
        g_I2 - (u2 * (2 * x9 ** 2 + 2 * x11 ** 2 - 1)) / x1 - (u1 * (2 * x8 * x11 - 2 * x9 * x10)) / x1 + (
                u3 * (2 * x8 * x9 + 2 * x10 * x11)) / x1,
        g_I3 - (u3 * (2 * x9 ** 2 + 2 * x10 ** 2 - 1)) / x1 + (u1 * (2 * x8 * x10 + 2 * x9 * x11)) / x1 - (
                u2 * (2 * x8 * x9 - 2 * x10 * x11)) / x1,
        - (x9 * x12) / 2 - (x10 * x13) / 2 - (x11 * x14) / 2,
        (x8 * x12) / 2 + (x10 * x14) / 2 - (x11 * x13) / 2,
        (x8 * x13) / 2 - (x9 * x14) / 2 + (x11 * x12) / 2,
        (x8 * x14) / 2 + (x9 * x13) / 2 - (x10 * x12) / 2,
        (r_T_B2 * u3 - r_T_B3 * u2 + J_B2 * x13 * x14 - J_B3 * x13 * x14) / J_B1,
        - (r_T_B1 * u3 - r_T_B3 * u1 + J_B1 * x12 * x14 - J_B3 * x12 * x14) / J_B2,
        (r_T_B1 * u2 - r_T_B2 * u1 + J_B1 * x12 * x13 - J_B2 * x12 * x13) / J_B3
    ))
