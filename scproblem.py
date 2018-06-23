import cvxpy as cvx


class SCProblem:
    """
    Defines a standard Successive Convexification problem for a given model.

    :param m: The model object
    :param K: Number of discretization points
    """

    def __init__(self, m, K):
        # Variables:
        self.X_v = cvx.Variable((m.n_x, K))
        self.U_v = cvx.Variable((m.n_u, K))
        self.sigma_v = cvx.Variable()

        self.nu_norm_v = cvx.Variable()
        self.delta_norm_v = cvx.Variable()
        self.delta_s_v = cvx.Variable()

        # Objective Variables
        self.nu_v = cvx.Variable((m.n_x, K - 1))
        self.delta_v = cvx.Variable(K)

        # Parameters:
        self.par = dict()
        self.par['A_bar'] = cvx.Parameter((m.n_x * m.n_x, K - 1))
        self.par['B_bar'] = cvx.Parameter((m.n_x * m.n_u, K - 1))
        self.par['C_bar'] = cvx.Parameter((m.n_x * m.n_u, K - 1))
        self.par['Sigma_bar'] = cvx.Parameter((m.n_x, K - 1))
        self.par['z_bar'] = cvx.Parameter((m.n_x, K - 1))

        self.par['X_last'] = cvx.Parameter((m.n_x, K))
        self.par['U_last'] = cvx.Parameter((m.n_u, K))
        self.par['Sigma_last'] = cvx.Parameter(nonneg=True)

        self.par['E'] = cvx.Parameter((m.n_x, m.n_x), nonneg=True)

        self.par['weight_sigma'] = cvx.Parameter(nonneg=True)
        self.par['weight_delta'] = cvx.Parameter(nonneg=True)
        self.par['weight_nu'] = cvx.Parameter(nonneg=True)
        self.par['weight_delta_sigma'] = cvx.Parameter(nonneg=True)

        # Constraints:
        constraints = []

        # Model:
        constraints += m.get_constraints(self.X_v, self.U_v, self.par['X_last'], self.par['U_last'])

        # Dynamics:
        rhs = [
            cvx.reshape(self.par['A_bar'][:, k], (m.n_x, m.n_x)) * self.X_v[:, k]
            + cvx.reshape(self.par['B_bar'][:, k], (m.n_x, m.n_u)) * self.U_v[:, k]
            + cvx.reshape(self.par['C_bar'][:, k], (m.n_x, m.n_u)) * self.U_v[:, k + 1]
            + self.par['Sigma_bar'][:, k] * self.sigma_v
            + self.par['z_bar'][:, k]
            + self.par['E'] * self.nu_v[:, k]
            for k in range(K - 1)
        ]
        rhs = cvx.vstack(rhs)
        constraints += [self.X_v[:, 1:].T == rhs]

        # Trust regions:
        dx = self.X_v - self.par['X_last']
        du = self.U_v - self.par['U_last']
        ds = self.sigma_v - self.par['Sigma_last']
        constraints += [cvx.sum(cvx.square(dx), axis=0) + cvx.sum(cvx.square(du), axis=0) <= self.delta_v]
        constraints += [cvx.square(ds) <= self.delta_s_v]

        # Slack variables:
        constraints += [
            cvx.norm(self.delta_v) <= self.delta_norm_v,
            cvx.norm(self.nu_v, 1) <= self.nu_norm_v
        ]

        # Objective:
        model_objective = m.get_objective(self.X_v, self.U_v, self.par['X_last'], self.par['U_last'])
        sc_objective = cvx.Minimize(
            self.par['weight_sigma'] * self.sigma_v
            + self.par['weight_nu'] * self.nu_norm_v
            + self.par['weight_delta'] * self.delta_norm_v
            + self.par['weight_delta_sigma'] * self.delta_s_v
        )
        objective = sc_objective  # + model_objective

        # Flight time positive:
        constraints += [self.sigma_v >= 0.1]

        self.prob = cvx.Problem(objective, constraints)

    def check_dcp(self):
        print('Problem is ' + ('valid.' if self.prob.is_dcp() else 'invalid.'))

    def update_parameters(self, **kwargs):
        """
        All parameters have to be filled before calling solve().
        Takes the following arguments as keywords:

        A_bar
        B_bar
        C_bar
        Sigma_bar
        z_bar
        X_last
        U_last
        Sigma_last
        E
        weight_sigma
        weight_delta
        weight_delta_sigma
        weight_nu
        """

        for key in kwargs:
            self.par[key].value = kwargs[key]

    def solve(self, **kwargs):
        error = False
        try:
            self.prob.solve(**kwargs)
        except cvx.SolverError:
            error = True

        stats = self.prob.solver_stats
        print()

        info = {
            'setup_time': stats.setup_time,
            'solver_time': stats.solve_time,
            'iterations': stats.num_iters,
            'solver_error': error
        }

        return info

    def get_solution(self):
        return self.X_v.value, self.U_v.value, self.sigma_v.value

    def check_convergence(self, delta_tol, nu_tol):
        if self.delta_norm_v.value is not None:
            converged = self.delta_norm_v.value < delta_tol and self.nu_norm_v.value < nu_tol
            info = {
                'delta_norm': self.delta_norm_v.value,
                'nu_norm': self.nu_norm_v.value,
                'sigma': self.sigma_v.value
            }
        else:
            converged = False
            print('No solution available. Call solve() first.')
            info = {}

        return converged, info
