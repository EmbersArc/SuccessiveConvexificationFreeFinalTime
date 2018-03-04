import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from parameters import *
import matlab.engine

print("Starting Matlab.")
eng = matlab.engine.start_matlab()


# convert numpy array to matlab matrix
def toml(A):
    return matlab.double(A)


X = np.empty(shape=[K, 14])
U = np.empty(shape=[K, 3])

# START INITIALIZATION--------------------------------------------------------------------------------------------------
print("Starting Initialization.")

for k in range(K):
    alpha1 = (K - k) / K
    alpha2 = k / K
    m_k = (alpha1 * m_wet + alpha2 * m_dry,)
    r_I_k = alpha1 * r_I_init
    v_I_k = alpha1 * v_I_init + alpha2 * v_I_final
    q_B_I_k = np.array((1.0, 0.0, 0.0, 0.0))
    w_B_k = np.array((0., 0., 0.))

    X[k, :] = np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k))
    U[k, :] = m_k * -g_I

sigma = t_f_guess
x_init = X[0, :].copy()

print("Initialization finished.")
# END INITIALIZATION----------------------------------------------------------------------------------------------------

# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
for it in range(5):
    print("Iteration", it)
    A_bar = np.zeros([K, 14, 14])
    B_bar = np.zeros([K, 14, 3])
    C_bar = np.zeros([K, 14, 3])
    Sigma_bar = np.zeros([K, 14])
    z_bar = np.zeros([K, 14])

    print("Calculating new transition matrices.")

    dt = 1 / (K - 1)
    for k in range(0, K - 1):
        tk = k / (K - 1)

        A_bar[k, :, :] = np.eye(14) + A(X[k, :], U[k, :], sigma) * dt
        # x_t = X[k, :].copy()
        for i in range(res):
            xi = tk + i / res * dt
            a_k = (tk + dt - xi) / dt
            b_k = (xi - tk) / dt
            u_t = a_k * U[k, :] + b_k * U[k + 1, :]
            # x_t += f(x_t, u_t) * dt
            Phi_xi = np.eye(14) + A(X[k, :], u_t, sigma) * (xi - tk)  # matrix exponential approximation

            B_bar[k, :, :] += np.matmul(Phi_xi, B(X[k, :], u_t, sigma)) * a_k

            C_bar[k, :, :] += np.matmul(Phi_xi, B(X[k, :], u_t, sigma)) * b_k

            Sigma_bar[k, :] += np.matmul(Phi_xi, f(X[k, :], u_t))

            z_bar[k, :] += np.matmul(Phi_xi, - np.matmul(A(X[k, :], u_t, sigma), X[k, :])
                                     - np.matmul(B(X[k, :], u_t, sigma), u_t))

        B_bar[k, :, :] *= dt / res
        C_bar[k, :, :] *= dt / res
        Sigma_bar[k, :] *= dt / res
        z_bar[k, :] *= dt / res

    print("Sending problem to CVX.")
    # array passing very slow
    X_sol, U_sol, s_sol, done = eng.solve_socp(toml(A_bar), toml(B_bar), toml(C_bar), toml(Sigma_bar), toml(z_bar),
                                               toml(X), toml(U), sigma, toml(x_init), K, nargout=4)

    if done:
        print("Done after ", it, "iterations.")
        break

    X = np.asarray(X_sol)
    U = np.asarray(U_sol)
    sigma = s_sol

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 1], X[:, 2], X[:, 3], zdir='x')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 5)
plt.show()
