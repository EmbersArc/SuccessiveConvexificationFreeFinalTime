import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from parameters import *
import pickle
import cvxpy as cvx

X = np.empty(shape=[K, 14])
U = np.empty(shape=[K, 3])

x_init = np.concatenate(((m_wet,), r_I_init, v_I_init, q_B_I_init, w_B_init))
x_final = np.concatenate(((m_dry,), r_I_final, v_I_final, q_B_I_final, w_B_final))
sigma = t_f_guess

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
sigma_last_ = cvx.Parameter()

# Boundary conditions:
constraints = [
    X_[0, 0] == x_init[0],
    X_[0, 1:4] == x_init[1:4],
    X_[0, 4:7] == x_init[4:7],
    # X_[0, 7:11] == x_init[7:11], # initial attitude is free
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
    sigma_ + w_nu * cvx.norm(nu_, 1) + w_delta * cvx.norm(delta_) + w_delta_sigma * cvx.norm(delta_s_, 1)
)
prob = cvx.Problem(objective, constraints)
print("Problem is " + ("valid." if prob.is_dcp() else "invalid."))
# CVX ------------------------------------------------------------------------------------------------------------------

# START INITIALIZATION--------------------------------------------------------------------------------------------------
print("Starting Initialization.")

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
        # TODO: numerical integration
        # find A_bar, B_bar, C_bar, Sigma_bar, z_bar

        # CVX ----------------------------------------------------------------------------------------------------------
        A_bar_.value = A_bar.reshape((K, 14 * 14), order='F')
        B_bar_.value = B_bar.reshape((K, 14 * 3), order='F')
        C_bar_.value = C_bar.reshape((K, 14 * 3), order='F')
        Sigma_bar_.value = Sigma_bar
        z_bar_.value = z_bar
        X_last_.value = X
        U_last_.value = U
        sigma_last_.value = sigma

        print("Solving problem.")

        prob.solve(verbose=True, solver='ECOS')
        # CVX ----------------------------------------------------------------------------------------------------------

        X = X_.value
        U = U_.value

        delta_norm = np.linalg.norm(delta_.value)
        nu_norm = np.linalg.norm(nu_.value, ord=1)
        print("Flight time:", sigma_.value)
        print("Delta_norm:", delta_norm)
        print("Nu_norm:", nu_norm)
        if delta_norm < delta_tol and nu_norm < nu_tol:
            print("Converged after", it + 1, "iterations.")
    break

pickle.dump(X, open("trajectory/X.p", "wb"))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 2], X[:, 3], X[:, 1], zdir='z')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 5)
plt.show()
