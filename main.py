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
x_init = X[0, :]

print("Initialization finished.")

# END INITIALIZATION----------------------------------------------------------------------------------------------------

# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
for it in range(iterations):
    A_bar = np.zeros([K, 14, 14])
    B_bar = np.zeros([K, 14, 3])
    C_bar = np.zeros([K, 14, 3])
    Sigma_bar = np.zeros([K, 14])
    z_bar = np.zeros([K, 14])

    for k in range(0, K - 1):
        tk = k / (K - 1)
        tk_1 = (k + 1) / (K - 1)

        A_bar[k, :, :] = np.eye(14) + A(X[k, :], U[k, :], sigma) * (tk_1 - tk)

        for i in range(res):
            xi = tk + i / res * (tk_1 - tk)
            u_t = alpha(tk_1, xi, tk) * U[k, :] + beta(tk_1, xi, tk) * U[k + 1, :]
            Phi_xi = np.eye(14) + A(X[k, :], U[k, :], sigma) * (xi - tk)  # matrix exponential approximation

            B_bar[k, :, :] += np.matmul(Phi_xi, B(X[k, :], u_t, sigma)) * alpha(tk_1, xi, tk)

            C_bar[k, :, :] += np.matmul(Phi_xi, B(X[k, :], u_t, sigma)) * beta(tk_1, xi, tk)

            Sigma_bar[k, :] += np.matmul(Phi_xi, f(X[k, :], u_t))

            z_bar[k, :] += np.matmul(Phi_xi, - np.matmul(A(X[k, :], u_t, sigma), X[k, :])
                                     - np.matmul(B(X[k, :], u_t, sigma), U[k, :]))

        B_bar[k, :, :] *= (tk_1 - tk) / res
        C_bar[k, :, :] *= (tk_1 - tk) / res
        Sigma_bar[k, :] *= (tk_1 - tk) / res
        z_bar[k, :] *= (tk_1 - tk) / res

    # array passing very slow
    X_sol, U_sol, s_sol, done = eng.solve_socp(toml(A_bar), toml(B_bar), toml(C_bar), toml(Sigma_bar), toml(z_bar),
                                               toml(X), toml(U), sigma, toml(x_init), K, nargout=4)

    if done:
        print("Done after ", it, "iterations.")
        break

    X = np.asarray(X_sol)
    U = np.asarray(U_sol)
    sigma = s_sol

