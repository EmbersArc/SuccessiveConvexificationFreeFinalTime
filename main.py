import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from parameters import *
import matlab.engine
from scipy.integrate import odeint

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

def dPsidTau(Psi, i, x_hat, u_hat, sigma_hat):
    i = int(i)
    if i > res - 1:
        x_interp = x_hat[res - 1, :] + (x_hat[res - 1, :] - x_hat[res - 2, :]) * (i - res)
        u_interp = u_hat[res - 1, :] + (u_hat[res - 1, :] - u_hat[res - 2, :]) * (i - res)
        return (A(x_interp, u_interp, sigma_hat) * Psi.reshape((14, 14))).reshape(-1)
    else:
        return (A(x_hat[i, :], u_hat[i, :], sigma_hat) * Psi.reshape((14, 14))).reshape(-1)


# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
for it in range(2):
    print("Iteration", it)
    A_bar = np.zeros([K, 14, 14])
    B_bar = np.zeros([K, 14, 3])
    C_bar = np.zeros([K, 14, 3])
    Sigma_bar = np.zeros([K, 14])
    z_bar = np.zeros([K, 14])

    print("Calculating new transition matrices.")

    dt = 1 / (K - 1)
    for k in range(0, K - 1):
        a = np.linspace(0, 1, res)
        b = np.linspace(1, 0, res)
        t_k = k / (K - 1)
        x_hat = np.zeros([res, 14])
        x_hat[0, :] = X[k, :]
        u_hat = np.vstack((np.linspace(U[k, 0], U[k + 1, 0], res),
                           np.linspace(U[k, 1], U[k + 1, 1], res),
                           np.linspace(U[k, 2], U[k + 1, 2], res))).T

        sigma_hat = sigma
        for i in range(1, res):
            x_hat[i, :] = (x_hat[i - 1, :] * res + f(x_hat[i - 1, :], u_hat[i, :]) * sigma_hat * dt) / res

        Phi = odeint(dPsidTau, np.eye(14).reshape(-1), range(res), args=(x_hat, u_hat, sigma_hat)).reshape((res, 14, 14))

        A_bar[k, :, :] = Phi[res - 1, :, :]

        for i in range(1, res):
            B_bar[k, :, :] += np.matmul(Phi[i, :, :], B(x_hat[i, :], u_hat[i, :], sigma_hat)) * a[i]

            C_bar[k, :, :] += np.matmul(Phi[i, :, :], B(x_hat[i, :], u_hat[i, :], sigma_hat)) * b[i]

            Sigma_bar[k, :] += np.matmul(Phi[i, :, :], f(x_hat[i, :], u_hat[i, :]))

            z_bar[k, :] += np.matmul(Phi[i, :, :], - np.matmul(A(x_hat[i, :], u_hat[i, :], sigma_hat), x_hat[i, :])
                                     - np.matmul(B(x_hat[i, :], u_hat[i, :], sigma_hat), u_hat[i, :]))

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
