import numpy as np
from scipy.integrate import odeint


class Discretize:
    def __init__(self, m, dt, K):
        self.K = K
        self.m = m
        self.A_bar = np.zeros([m.n_x * m.n_x, K - 1])
        self.B_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.C_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.Sigma_bar = np.zeros([m.n_x, K - 1])
        self.z_bar = np.zeros([m.n_x, K - 1])

        # vector indices for flat matrices
        self.idx = [
            m.n_x,
            m.n_x * (1 + m.n_x),
            m.n_x * (1 + m.n_x + m.n_u),
            m.n_x * (1 + m.n_x + m.n_u + m.n_u),
            m.n_x * (1 + m.n_x + m.n_u + m.n_u + 1),
            m.n_x * (1 + m.n_x + m.n_u + m.n_u + 2),
        ]
        self.f, self.A, self.B = m.get_equations()

        # integration initial condition
        self.V0 = np.zeros((self.idx[5],))
        self.V0[self.idx[0]:self.idx[1]] = np.eye(m.n_x).reshape(-1)

        # vector indices for flat matrices
        self.dt = dt

    def calculate(self, X, U, sigma):
        for k in range(self.K - 1):
            self.V0[0:self.m.n_x] = X[:, k]
            V = np.array(odeint(self.ode_dVdt, self.V0, (0, self.dt), args=(U[:, k], U[:, k + 1], sigma)))[1, :]

            # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
            Phi = V[self.idx[0]:self.idx[1]].reshape((self.m.n_x, self.m.n_x))
            self.A_bar[:, k] = V[self.m.n_x:self.idx[1]].reshape((self.m.n_x, self.m.n_x)).flatten(order='F')
            self.B_bar[:, k] = np.matmul(Phi, V[self.idx[1]:self.idx[2]].reshape((self.m.n_x, self.m.n_u))).flatten(
                order='F')
            self.C_bar[:, k] = np.matmul(Phi, V[self.idx[2]:self.idx[3]].reshape((self.m.n_x, self.m.n_u))).flatten(
                order='F')
            self.Sigma_bar[:, k] = np.matmul(Phi, V[self.idx[3]:self.idx[4]])
            self.z_bar[:, k] = np.matmul(Phi, V[self.idx[4]:self.idx[5]])
        return self.A_bar, self.B_bar, self.C_bar, self.Sigma_bar, self.z_bar

    # ODE function to compute dVdt
    # V = [x(14), Phi_A(14x14), B_bar(14x3), C_bar(14x3), Simga_bar(14), z_bar(14)]
    def ode_dVdt(self, V, t, u_t, u_t1, sigma):
        alpha = t / self.dt
        beta = 1 - alpha
        dVdt = np.empty((self.idx[5],))
        x = V[0:self.idx[0]]
        u = u_t + alpha * (u_t1 - u_t)

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(V[self.idx[0]:self.idx[1]].reshape((14, 14)))

        A_subs = sigma * self.A(x, u)
        B_subs = sigma * self.B(x, u)
        f_subs = self.f(x, u)

        dVdt[0:self.idx[0]] = sigma * f_subs.transpose()
        dVdt[self.idx[0]:self.idx[1]] = np.matmul(A_subs, V[self.idx[0]:self.idx[1]].reshape((14, 14))).reshape(-1)
        dVdt[self.idx[1]:self.idx[2]] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * alpha
        dVdt[self.idx[2]:self.idx[3]] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * beta
        dVdt[self.idx[3]:self.idx[4]] = np.matmul(Phi_A_xi, f_subs).transpose()
        z_t = -np.matmul(A_subs, x) - np.matmul(B_subs, u)
        dVdt[self.idx[4]:self.idx[5]] = np.matmul(Phi_A_xi, z_t)

        return dVdt
