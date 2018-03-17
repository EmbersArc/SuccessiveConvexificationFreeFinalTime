import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from parameters import *
import pickle
import cvxpy as cvx
from scipy.integrate import odeint

X = np.empty(shape=[K, 14])
U = np.empty(shape=[K, 3])

x_init = np.concatenate(((m_wet,), r_I_init, v_I_init, q_B_I_init, w_B_init))
x_final = np.concatenate(((m_dry,), r_I_final, v_I_final, q_B_I_final, w_B_final))

# CVX ------------------------------------------------------------------------------------------------------------------
print("Setting up problem.")
# Variables:
X_ = cvx.Variable((K, 14))
U_ = cvx.Variable((K, 3))
sigma_ = cvx.Variable()
nu_ = cvx.Variable((14 * (K - 1)))
delta_ = cvx.Variable(K)
delta_s_ = cvx.Variable()

# Parameters:
A_bar_ = cvx.Parameter((K, 14 * 14))
B_bar_ = cvx.Parameter((K, 14 * 3))
C_bar_ = cvx.Parameter((K, 14 * 3))
Sigma_bar_ = cvx.Parameter((K, 14))
z_bar_ = cvx.Parameter((K, 14))
X_last_ = cvx.Parameter((K, 14))
U_last_ = cvx.Parameter((K, 3))
sigma_last_ = cvx.Parameter(nonneg=True)
w_delta_ = cvx.Parameter(nonneg=True)
w_nu_ = cvx.Parameter(nonneg=True)
w_delta_sigma_ = cvx.Parameter(nonneg=True)

# Boundary conditions:
constraints = [
    X_[0, 0] == x_init[0],
    X_[0, 1:4] == x_init[1:4],
    X_[0, 4:7] == x_init[4:7],
    # X_[0, 7:11] == x_init[7:11],  # initial attitude is free
    X_[0, 11:14] == x_init[11:14],

    # X_[0, 0] == x_final[0], # final mass is free
    X_[K - 1, 1:4] == x_final[1:4],
    X_[K - 1, 4:7] == x_final[4:7],
    X_[K - 1, 7:11] == x_final[7:11],
    X_[K - 1, 11:14] == x_final[11:14],

    U_[K - 1, 1] == 0,
    U_[K - 1, 2] == 0
]

# Dynamics:
for k in range(K - 1):
    constraints += [
        X_[k + 1, :] ==
        cvx.reshape(A_bar_[k, :], (14, 14)) * X_[k, :]
        + cvx.reshape(B_bar_[k, :], (14, 3)) * U_[k, :]
        + cvx.reshape(C_bar_[k, :], (14, 3)) * U_[k + 1, :]
        + Sigma_bar_[k, :] * sigma_
        + z_bar_[k, :]
        + nu_[k * 14:(k + 1) * 14]
    ]

# State constraints:
constraints += [X_[:, 0] >= m_dry]
for k in range(K):
    constraints += [
        tan_gamma_gs * cvx.norm(X_[k, 2: 4]) <= X_[k, 1],
        cos_theta_max <= 1 - 2 * cvx.sum_squares(X_[k, 9:11]),
        cvx.norm(X_[k, 11: 14]) <= w_B_max
    ]

# Control constraints:
for k in range(K):
    B_g = U_last_[k, :] / cvx.norm(U_last_[k, :])
    constraints += [
        T_min <= B_g * U_[k, :],
        cvx.norm(U_[k, :]) <= T_max,
        cos_delta_max * cvx.norm(U_[k, :]) <= U_[k, 0]
    ]

# Trust regions:
for k in range(K):
    dx = X_[k, :] - X_last_[k, :]
    du = U_[k, :] - U_last_[k, :]
    constraints += [
        cvx.sum_squares(dx) + cvx.sum_squares(du) <= delta_[k]
    ]
ds = sigma_ - sigma_last_
constraints += [cvx.norm(ds, 2) <= delta_s_]

# Objective:
objective = cvx.Minimize(
    sigma_ + w_nu_ * cvx.norm(nu_, 1) + w_delta_ * cvx.norm(delta_) + w_delta_sigma_ * cvx.norm(delta_s_, 1)
)

prob = cvx.Problem(objective, constraints)

print("Problem is " + ("valid." if prob.is_dcp() else "invalid."))
# CVX ------------------------------------------------------------------------------------------------------------------
# START INITIALIZATION--------------------------------------------------------------------------------------------------
print("Starting Initialization.")
sigma = t_f_guess

for k in range(K):
    alpha1 = (K - k) / K
    alpha2 = k / K
    m_k = (alpha1 * x_init[0] + alpha2 * x_final[0],)
    r_I_k = alpha1 * x_init[1:4] + alpha2 * x_final[1:4]
    v_I_k = alpha1 * x_init[4:7] + alpha2 * x_final[4:7]
    q_B_I_k = np.array((1.0, 0.0, 0.0, 0.0))
    w_B_k = alpha1 * x_init[11:14] + alpha2 * x_final[11:14]

    X[k, :] = np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k))
    U[k, :] = m_k * -g_I

print("Initialization finished.")


# END INITIALIZATION----------------------------------------------------------------------------------------------------
def ode_dPhidt(V, t, u_t, u_t1, sigma):
    u = u_t + t / dt * (u_t1 - u_t)
    dVdt = np.zeros((14 + 14 * 14,))
    x = V[0:14]
    Phi_A = V[14:210].reshape((14, 14))
    dVdt[0:14] = sigma * f(x, u)
    dVdt[14:210] = np.matmul(A(x, u, sigma), Phi_A).reshape(-1)

    return dVdt


def ode_dVdt(V, t, u_t, u_t1, sigma):
    # V = x(14), Phi_A(14x14), B_bar(14x3), C_bar(14x3), Simga_bar(14), z_bar(14)
    u = u_t + t / dt * (u_t1 - u_t)
    alpha = t / dt
    beta = 1 - t / dt
    dVdt = np.zeros((14 + 14 * 14 + 14 * 3 + 14 * 3 + 14 + 14,))
    x = V[0:14]

    V0 = np.zeros((14 + 14 * 14,))
    V0[0:14] = x
    V0[14:210] = np.eye(14).reshape(-1)
    Phi_A_xi = np.array(odeint(ode_dPhidt, V0, (t, dt), args=(u_t, u_t1, sigma)))[1, 14:].reshape((14, 14))

    dVdt[0:14] = sigma * f(x, u)
    dVdt[14:210] = np.matmul(A(x, u, sigma), Phi_A_xi).reshape(-1)
    dVdt[210:252] = np.matmul(Phi_A_xi, B(x, u, sigma)).reshape(-1) * alpha
    dVdt[252:294] = np.matmul(Phi_A_xi, B(x, u, sigma)).reshape(-1) * beta
    dVdt[294:308] = np.matmul(Phi_A_xi, f(x, u))
    z_t = -np.matmul(A(x, u, sigma), x) - np.matmul(B(x, u, sigma), u)
    dVdt[308:322] = np.matmul(Phi_A_xi, z_t)

    return dVdt


# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
for it in range(iterations):
    print("Iteration", it + 1)
    A_bar = np.zeros([K, 14, 14])
    B_bar = np.zeros([K, 14, 3])
    C_bar = np.zeros([K, 14, 3])
    Sigma_bar = np.zeros([K, 14])
    z_bar = np.zeros([K, 14])

    print("Calculating new transition matrices.")

    for k in range(0, K - 1):
        # TODO: fix numerical integration
        # find A_bar, B_bar, C_bar, Sigma_bar, z_bar
        V0 = np.zeros((322,))
        V0[0:14] = X[k, :]
        V0[14:210] = np.eye(14).reshape(-1)
        V = np.array(odeint(ode_dVdt, V0.squeeze(), (0, dt), args=(U[k, :], U[k + 1, :], sigma)))[1, :]

        A_bar[k, :, :] = V[14:210].reshape((14, 14))
        B_bar[k, :, :] = V[210:252].reshape((14, 3))
        C_bar[k, :, :] = V[252:294].reshape((14, 3))
        Sigma_bar[k, :] = V[294:308]
        z_bar[k, :] = V[308:322]

    # CVX ----------------------------------------------------------------------------------------------------------
    A_bar_.value = A_bar.reshape((K, 14 * 14), order='F')
    B_bar_.value = B_bar.reshape((K, 14 * 3), order='F')
    C_bar_.value = C_bar.reshape((K, 14 * 3), order='F')
    Sigma_bar_.value = Sigma_bar
    z_bar_.value = z_bar
    X_last_.value = X
    U_last_.value = U
    sigma_last_.value = sigma
    w_delta_.value = w_delta if it < 4 else w_delta * 1e3  # for faster convergence
    w_nu_.value = w_nu
    w_delta_sigma_.value = w_delta_sigma

    print("Solving problem.")

    prob.solve(verbose=True, solver='ECOS')
    # CVX ----------------------------------------------------------------------------------------------------------

    X = X_.value
    U = U_.value
    sigma = sigma_.value

    delta_norm = np.linalg.norm(delta_.value)
    nu_norm = np.linalg.norm(nu_.value, ord=1)
    print("Flight time:", sigma_.value)
    print("Delta_norm:", delta_norm)
    print("Nu_norm:", nu_norm)
    if delta_norm < delta_tol and nu_norm < nu_tol:
        print("Converged after", it + 1, "iterations.")
        break

pickle.dump(X, open("trajectory/X.p", "wb"))
pickle.dump(U, open("trajectory/U.p", "wb"))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 2], X[:, 3], X[:, 1], zdir='z')
ax.set_xlim(-r_I_init[0], r_I_init[0])
ax.set_ylim(-r_I_init[0], r_I_init[0])
ax.set_zlim(0, r_I_init[0])
plt.show()
