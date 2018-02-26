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
delta_sigma = P.add_variable('delta_sigma', 1)
nu_ = P.add_variable('nu', (Kn - 1, 14))

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
P.add_constraint(m_dry <= x_[:, 0])
P.add_list_of_constraints([abs(H_23 * x_[k, 1:4].T) <= x_[k, 1] / tan_gamma_gs for k in range(Kn)])
P.add_list_of_constraints([abs(H_q * x_[k, 7:11].T) <= np.sqrt((1 - cos_theta_max) / 2) for k in range(Kn)])
P.add_list_of_constraints([abs(x_[k, 11:14]) <= w_B_max for k in range(Kn)])

# Control Constraints
P.add_list_of_constraints([T_min <= u_last[k, :].T / np.linalg.norm(u_last[k, :]) * u_[k, :] for k in range(Kn)])
P.add_list_of_constraints([pic.norm(u_[k, :]) <= T_max for k in range(Kn)])
P.add_list_of_constraints([pic.norm(u_[k, :]) <= u_[k, 0] / cos_delta_max for k in range(Kn)])

# Dynamics
P.add_list_of_constraints([
    x_[k + 1, :].T ==
    cvx.matrix(A_bar[k, :, :]) * x_[k, :].T + cvx.matrix(B_bar[k, :, :]) * u_[k, :].T
    + cvx.matrix(C_bar[k, :, :]) * u_[k + 1, :].T + cvx.matrix(Sigma_bar[k, :]) * sigma_
    + cvx.matrix(z_bar[k, :]) + nu_[k, :].T
    for k in range(Kn - 1)
])

# Trust Regions
P.add_list_of_constraints([
    abs(x_[k, :] - x_last[k, :])**2 + abs(u_[k, :] - u_last[k, :])**2 <= delta_[k] for k in range(Kn)
])
P.add_constraint(abs(sigma_ - sigma_last) <= delta_sigma)
# P.convert_quad_to_socp()

# Cost Function
nu_sum = pic.sum([1 | nu_[k, :] for k in range(Kn - 1)])

objective = sigma_ + w_nu * nu_sum + w_delta * abs(delta_) ** 2 + w_delta_sigma * delta_sigma
P.set_objective('min', objective)
P.solver_selection()
print(P)


P.solve(verbose=True)

# for i in range(1, iterations):
