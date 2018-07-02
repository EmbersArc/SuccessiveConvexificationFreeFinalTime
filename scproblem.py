import cvxpy as cvx


class SCProblem:
    """
    Defines a standard Successive Convexification problem for a given model.

    :param m: The model object
    :param K: Number of discretization points
    """

    def __init__(self, m, K):
        # Variables:
        self.var = dict()
        self.var['X'] = cvx.Variable((m.n_x, K))
        self.var['U'] = cvx.Variable((m.n_u, K))
        self.var['sigma'] = cvx.Variable()
        self.var['nu'] = cvx.Variable((m.n_x, K - 1))

        # Parameters:
        self.par = dict()
        self.par['A_bar'] = cvx.Parameter((m.n_x * m.n_x, K - 1))
        self.par['B_bar'] = cvx.Parameter((m.n_x * m.n_u, K - 1))
        self.par['C_bar'] = cvx.Parameter((m.n_x * m.n_u, K - 1))
        self.par['S_bar'] = cvx.Parameter((m.n_x, K - 1))
        self.par['z_bar'] = cvx.Parameter((m.n_x, K - 1))

        self.par['X_last'] = cvx.Parameter((m.n_x, K))
        self.par['U_last'] = cvx.Parameter((m.n_u, K))
        self.par['sigma_last'] = cvx.Parameter(nonneg=True)

        self.par['E'] = cvx.Parameter((m.n_x, m.n_x), nonneg=True)

        self.par['weight_sigma'] = cvx.Parameter(nonneg=True)
        self.par['weight_nu'] = cvx.Parameter(nonneg=True)
        self.par['radius_trust_region'] = cvx.Parameter(nonneg=True)

        # Constraints:
        constraints = []

        # Model:
        constraints += m.get_constraints(self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])

        # Dynamics:
        constraints += [
            self.var['X'][:, k + 1] ==
            cvx.reshape(self.par['A_bar'][:, k], (m.n_x, m.n_x)) * self.var['X'][:, k]
            + cvx.reshape(self.par['B_bar'][:, k], (m.n_x, m.n_u)) * self.var['U'][:, k]
            + cvx.reshape(self.par['C_bar'][:, k], (m.n_x, m.n_u)) * self.var['U'][:, k + 1]
            + self.par['S_bar'][:, k] * self.var['sigma']
            + self.par['z_bar'][:, k]
            + self.par['E'] * self.var['nu'][:, k]
            for k in range(K - 1)
        ]

        # Trust regions:
        dx = self.var['X'] - self.par['X_last']
        du = self.var['U'] - self.par['U_last']
        ds = self.var['sigma'] - self.par['sigma_last']
        constraints += [cvx.norm(dx, 1) <= self.par['radius_trust_region']]
        constraints += [cvx.norm(du, 1) <= self.par['radius_trust_region']]
        constraints += [cvx.norm(ds, 1) <= self.par['radius_trust_region']]

        # Objective:
        model_objective = m.get_objective(self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])
        sc_objective = cvx.Minimize(
            self.par['weight_sigma'] * self.var['sigma'] + self.par['weight_nu'] * cvx.norm(self.var['nu'], 1)
        )
        objective = sc_objective
        if model_objective is not None:
            objective += model_objective

        # Flight time positive:
        constraints += [self.var['sigma'] >= 0.1]

        self.prob = cvx.Problem(objective, constraints)

    def set_parameters(self, **kwargs):
        """
        All parameters have to be filled before calling solve().
        Takes the following arguments as keywords:

        A_bar
        B_bar
        C_bar
        S_bar
        z_bar
        X_last
        U_last
        sigma_last
        E
        weight_sigma
        weight_nu
        radius_trust_region
        """

        for key in kwargs:
            if key in self.par:
                self.par[key].value = kwargs[key]
            else:
                print(f'Parameter \'{key}\' does not exist.')

    def print_available_parameters(self):
        print('Parameter names:')
        for key in self.par:
            print(f'\t {key}')
        print('\n')

    def print_available_variables(self):
        print('Variable names:')
        for key in self.var:
            print(f'\t {key}')
        print('\n')

    def get_variable(self, name):
        """
        :param name: Name of the variable.
        :return The value of the variable.

        The following variables can be accessed:
        X
        U
        sigma
        nu
        """

        if name in self.var:
            return self.var[name].value
        else:
            print(f'Variable \'{name}\' does not exist.')
            return None

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
