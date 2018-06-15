import numpy as np
from scipy.integrate import odeint
from numba import jit


class Discretize:
    def __init__(self, m, dt, K):
        self.K = K
        self.m = m
        self.n_x = m.n_x
        self.n_u = m.n_u

        self.A_bar = np.zeros([m.n_x * m.n_x, K - 1])
        self.B_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.C_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.Sigma_bar = np.zeros([m.n_x, K - 1])
        self.z_bar = np.zeros([m.n_x, K - 1])

        # vector indices for flat matrices
        self.x_ind = slice(0, m.n_x)
        self.A_bar_ind = slice(m.n_x, m.n_x * (1 + m.n_x))
        self.B_bar_ind = slice(m.n_x * (1 + m.n_x), m.n_x * (1 + m.n_x + m.n_u))
        self.C_bar_ind = slice(m.n_x * (1 + m.n_x + m.n_u), m.n_x * (1 + m.n_x + m.n_u + m.n_u))
        self.Sigma_bar_ind = slice(m.n_x * (1 + m.n_x + m.n_u + m.n_u), m.n_x * (1 + m.n_x + m.n_u + m.n_u + 1))
        self.z_bar_ind = slice(m.n_x * (1 + m.n_x + m.n_u + m.n_u + 1), m.n_x * (1 + m.n_x + m.n_u + m.n_u + 2))

        self.f, self.A, self.B = m.get_equations()

        # integration initial condition
        self.V0 = np.zeros((m.n_x * (1 + m.n_x + m.n_u + m.n_u + 2),))
        self.V0[self.A_bar_ind] = np.eye(m.n_x).reshape(-1)

        # vector indices for flat matrices
        self.dt = dt

    def calculate(self, X, U, sigma):
        for k in range(self.K - 1):
            self.V0[self.x_ind] = X[:, k]
            V = np.array(odeint(self.ode_dVdt, self.V0, (0, self.dt), args=(U[:, k], U[:, k + 1], sigma))[1, :])

            # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
            # flatten matrices in column-major (Fortran) order for CVXPY
            Phi = V[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order='F')
            self.B_bar[:, k] = np.matmul(Phi, V[self.B_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.C_bar[:, k] = np.matmul(Phi, V[self.C_bar_ind].reshape((self.n_x, self.n_u))).flatten(order='F')
            self.Sigma_bar[:, k] = np.matmul(Phi, V[self.Sigma_bar_ind])
            self.z_bar[:, k] = np.matmul(Phi, V[self.z_bar_ind])

        return self.A_bar, self.B_bar, self.C_bar, self.Sigma_bar, self.z_bar

    # ODE function to compute dVdt
    # V = [x, Phi_A, B_bar, C_bar, Simga_bar, z_bar]
    def ode_dVdt(self, V, t, u_t, u_t1, sigma):
        alpha = t / self.dt
        beta = 1 - alpha
        dVdt = np.empty((len(self.V0),))
        x = V[self.x_ind]
        u = u_t + alpha * (u_t1 - u_t)

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(V[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        A_subs = sigma * self.A(x, u)
        B_subs = sigma * self.B(x, u)
        f_subs = self.f(x, u)

        dVdt[self.x_ind] = sigma * f_subs.transpose()
        dVdt[self.A_bar_ind] = np.matmul(A_subs, V[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dVdt[self.B_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * alpha
        dVdt[self.C_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * beta
        dVdt[self.Sigma_bar_ind] = np.matmul(Phi_A_xi, f_subs).transpose()
        z_t = -np.matmul(A_subs, x) - np.matmul(B_subs, u)
        dVdt[self.z_bar_ind] = np.matmul(Phi_A_xi, z_t)

        return dVdt
