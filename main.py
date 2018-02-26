import picos as pic
import cvxopt as cvx
from parameters import *

X = np.empty(shape=[Kn, 14])
U = np.empty(shape=[Kn, 3])


def alpha(tk_1, t, tk):
    return (tk_1 - t) / (tk_1 - tk)


def beta(tk_1, t, tk):
    return (t - tk) / (tk_1 - tk)


# START INITIALIZATION--------------------------------------------------------------------------------------------------
for k in range(Kn):
    alpha1 = (Kn - k) / Kn
    alpha2 = k / Kn
    m_k = (alpha1 * m_wet + alpha2 * m_dry,)
    r_I_k = alpha1 * r_I_init
    v_I_k = alpha1 * v_I_init + alpha2 * v_I_final
    q_B_I_k = np.array((1.0, 0.0, 0.0, 0.0))
    w_B_k = np.array((0., 0., 0.))

    X[k, :] = np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k))
    U[k, :] = m_k * -g_I

sigma = t_f_guess

A_bar = np.zeros([Kn, 14, 14])
B_bar = np.zeros([Kn, 14, 3])
C_bar = np.zeros([Kn, 14, 3])
Sigma_bar = np.zeros([Kn, 14])
z_bar = np.zeros([Kn, 14])

for k in range(0, Kn - 1):
    tk = k / (Kn - 1)
    tk_1 = (k + 1) / (Kn - 1)

    A_bar[k, :, :] = np.exp(A(X[k, :], U[k, :], sigma) * (tk_1 - tk))

    res = 10
    for i in range(res):
        xi = tk + i / res * (tk_1 - tk)
        u_t = alpha(tk_1, xi, tk) * U[k, :] + beta(tk_1, xi, tk) * U[k + 1, :]
        Phi_xi = np.exp(A(X[k, :], u_t, sigma) * (xi - tk))

        B_bar[k, :, :] += np.matmul(Phi_xi, B(X[k, :], u_t, sigma)) * alpha(tk_1, xi, tk)
        C_bar[k, :, :] += np.matmul(Phi_xi, B(X[k, :], u_t, sigma)) * beta(tk_1, xi, tk)
        Sigma_bar[k, :] += np.matmul(Phi_xi, f(X[k, :], u_t))
        z_bar[k, :] += np.matmul(Phi_xi, - np.matmul(A(X[k, :], u_t, sigma), X[k, :])
                                 - np.matmul(B(X[k, :], u_t, sigma), U[k, :]))

    B_bar[k, :, :] *= (tk_1 - tk) / res
    C_bar[k, :, :] *= (tk_1 - tk) / res
    Sigma_bar[k, :] *= (tk_1 - tk) / res
    z_bar[k, :] *= (tk_1 - tk) / res

# END INITIALIZATION----------------------------------------------------------------------------------------------------


# X[0, k + 1, :] = np.matmul(A_bar, X[0, k, :]) + np.matmul(B_bar, U[0, k, :]) + \
#                  np.matmul(C_bar, U[0, k + 1, :]) + Sigma_bar * sigma + z_bar


# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------


P = pic.Problem()

H_23 = pic.new_param('H_23', [[0., 1., 0.], [0., 0., 1.]])
H_q = pic.new_param('H_q', [[0., 0., 1., 0.], [0., 0., 0., 1.]])

x_last = cvx.matrix(X)
u_last = cvx.matrix(U)

sigma_last = cvx.matrix(sigma)

# Optimization Variables
x_ = P.add_variable('x', (Kn, 14))
u_ = P.add_variable('u', (Kn, 3))
sigma_ = P.add_variable('sigma', 1)
delta_ = P.add_variable('delta', Kn)
nu_ = P.add_variable('nu', (Kn - 1, 14))

delta_sigma = P.add_variable('delta_sigma', 1)

# Boundary Condition
P.add_list_of_constraints([
    x_[0, 0] == m_wet,
    x_[0, 1:4] == r_I_init,
    x_[Kn - 1, 1:4] == r_I_final,
    x_[0, 4:7] == v_I_init,
    x_[Kn - 1, 4:7] == v_I_final,
    x_[Kn - 1, 7:11] == q_B_I_final,
    x_[0, 11:14] == w_B_init,
    x_[Kn - 1, 11:14] == w_B_final,
    u_[Kn - 1, 1] == 0,
    u_[Kn - 1, 2] == 0
])

# State Constraints
P.add_list_of_constraints([m_dry < x_[k][0] for k in range(Kn)])
P.add_list_of_constraints([pic.norm(H_23 * x_[k, 1:4].T) <= x_[k, 1] / tan_gamma_gs for k in range(Kn)])
P.add_list_of_constraints([pic.norm(H_q * x_[k, 7:11].T) <= np.sqrt((1 - cos_theta_max) / 2) for k in range(Kn)])
P.add_list_of_constraints([pic.norm(x_[k, 11:14]) < w_B_max for k in range(Kn)])

# Control Constraints
P.add_list_of_constraints([T_min <= u_last[k, :] / np.linalg.norm(u_last[k, :]) * u_[k] for k in range(Kn)])
P.add_list_of_constraints([pic.norm(u_[k]) <= T_max for k in range(Kn)])
P.add_list_of_constraints([pic.norm(u_[k]) <= u_[k][0] / cos_delta_max for k in range(Kn)])

# Dynamics
P.add_list_of_constraints([
    x_[k + 1, :].T ==
    cvx.matrix(A_bar[k, :, :]) * x_[k, :].T + cvx.matrix(B_bar[k, :, :]) * u_[k, :].T
    + cvx.matrix(C_bar[k, :, :]) * u_[k + 1, :].T + cvx.matrix(Sigma_bar[k, :]) * sigma_
    + cvx.matrix(z_bar[k, :]) + nu_[k, :].T
    for k in range(Kn - 1)
])

# Trust Regions
dx = [(x_[k, :] - x_last[k, :])|(x_[k, :] - x_last[k, :]) for k in range(Kn)]
du = [(u_[k, :] - u_last[k, :])|(u_[k, :] - u_last[k, :]) for k in range(Kn)]

P.add_list_of_constraints([
    dx[k] + du[k] <= delta_[k] for k in range(Kn)
])
P.add_constraint(abs(sigma_ - sigma_last) <= delta_sigma)

# Cost Function
print(P)
objective = sigma_ + abs(w_delta * delta_) + abs(w_delta_sigma * delta_sigma) + pic.sum(w_nu * nu_)
P.set_objective('min', objective)
P.solver_selection()
P.solve(verbose=True)

# for i in range(1, iterations):
