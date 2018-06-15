import cvxpy as cvx
import numpy as np


class SCProblem:
    def __init__(self, m, K):
        # Variables:
        self.X_v = cvx.Variable((m.n_x, K))
        self.U_v = cvx.Variable((m.n_u, K))
        self.sigma_v = cvx.Variable()

        self.nu_norm_v = cvx.Variable()
        self.delta_norm_v = cvx.Variable()
        self.delta_s_v = cvx.Variable()

        # Slack Variables
        self.nu_v = cvx.Variable((m.n_x, K - 1))
        self.delta_v = cvx.Variable(K)

        # Parameters:
        self.A_bar_p = cvx.Parameter((m.n_x * m.n_x, K - 1))
        self.B_bar_p = cvx.Parameter((m.n_x * m.n_u, K - 1))
        self.C_bar_p = cvx.Parameter((m.n_x * m.n_u, K - 1))
        self.Sigma_bar_p = cvx.Parameter((m.n_x, K - 1))
        self.z_bar_p = cvx.Parameter((m.n_x, K - 1))
        self.X_last_p = cvx.Parameter((m.n_x, K))
        self.U_last_p = cvx.Parameter((m.n_u, K))
        self.sigma_last_p = cvx.Parameter(nonneg=True)
        self.w_delta_p = cvx.Parameter(nonneg=True)
        self.w_nu_p = cvx.Parameter(nonneg=True)
        self.w_delta_sigma_p = cvx.Parameter(nonneg=True)

        constraints = []

        # Model constraints:
        constraints += m.get_constraints(self.X_v, self.U_v, self.X_last_p, self.U_last_p)

        # Dynamics:
        rhs = [
            cvx.reshape(self.A_bar_p[:, k], (m.n_x, m.n_x)) * self.X_v[:, k]
            + cvx.reshape(self.B_bar_p[:, k], (m.n_x, m.n_u)) * self.U_v[:, k]
            + cvx.reshape(self.C_bar_p[:, k], (m.n_x, m.n_u)) * self.U_v[:, k + 1]
            + self.Sigma_bar_p[:, k] * self.sigma_v
            + self.z_bar_p[:, k]
            + self.nu_v[:, k]
            for k in range(K - 1)
        ]
        rhs = cvx.vstack(rhs)
        constraints += [self.X_v[:, 1:].T == rhs]

        # Trust regions:
        dx = self.X_v - self.X_last_p
        du = self.U_v - self.U_last_p
        ds = self.sigma_v - self.sigma_last_p
        constraints += [cvx.sum(cvx.square(dx), axis=0) + cvx.sum(cvx.square(du), axis=0) <= self.delta_v]
        constraints += [cvx.square(ds) <= self.delta_s_v]

        # Slack variables:
        constraints += [
            cvx.norm(self.delta_v) <= self.delta_norm_v,
            cvx.norm(self.nu_v, 1) <= self.nu_norm_v
        ]

        model_objective = m.get_objective(self.X_v, self.U_v, self.X_last_p, self.U_last_p)

        # Objective:
        objective = cvx.Minimize(
            self.sigma_v
            + self.w_nu_p * self.nu_norm_v
            + self.w_delta_p * self.delta_norm_v
            + self.w_delta_sigma_p * self.delta_s_v
            + model_objective
        )

        # Flight time positive:
        constraints += [self.sigma_v >= 0]

        self.prob = cvx.Problem(objective, constraints)

    def check_dcp(self):
        print('Problem is ' + ('valid.' if self.prob.is_dcp() else 'invalid.'))

    def update_values(self, A_bar, B_bar, C_bar, Sigma_bar, z_bar, X, U, sigma, w_delta, w_nu, w_delta_sigma):
        self.A_bar_p.value = A_bar
        self.B_bar_p.value = B_bar
        self.C_bar_p.value = C_bar
        self.Sigma_bar_p.value = Sigma_bar
        self.z_bar_p.value = z_bar
        self.X_last_p.value = X
        self.U_last_p.value = U
        self.sigma_last_p.value = sigma
        self.w_delta_p.value = w_delta
        self.w_nu_p.value = w_nu
        self.w_delta_sigma_p.value = w_delta_sigma

    def solve(self, **kwargs):
        try:
            self.prob.solve(**kwargs)
        except cvx.SolverError:
            pass
        
    def get_solution(self):
        return self.X_v.value, self.U_v.value, self.sigma_v.value

    def get_solver_stats(self):
        info = self.prob.solver_stats
        return f'''
Time for setup: {info.setup_time}
Time for solver: {info.solve_time}
        '''

    def check_convergence(self, delta_tol, nu_tol):

        converged = self.delta_norm_v.value < delta_tol and self.nu_norm_v < nu_tol

        return converged

    def get_convergence_info(self):
        if self.delta_norm_v.value is not None:
            info = f'''
Trust Region Norm: {self.delta_norm_v.value}
Virtual Control Norm: {self.nu_norm_v.value}
Total Time': {self.sigma_v.value}
            '''
        else:
            info = 'No solution available.'

        return info
