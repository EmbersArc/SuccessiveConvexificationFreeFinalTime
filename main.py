from time import time
import numpy as np

from model_6dof import Model_6DoF
from parameters import *
from discretization import Integrator
from visualization.plot3d import plot3d
from scproblem import SCProblem
from utils import format_line, save_arrays

"""
Python implementation of 'Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time' paper
by Michael Szmuk and Behçet Açıkmeşe.

Implementation by Sven Niederberger (s-niederberger@outlook.com)
"""

m = Model_6DoF()

# state and input
X = np.empty(shape=[m.n_x, K])
U = np.empty(shape=[m.n_u, K])

# INITIALIZATION--------------------------------------------------------------------------------------------------------
sigma = m.t_f_guess
X, U = m.initialize_trajectory(X, U)

# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
all_X = [X]
all_U = [U]

integrator = Integrator(m, K)
problem = SCProblem(m, K)

last_linear_cost = None

converged = False
for it in range(iterations):
    t0_it = time()
    print('-' * 50)
    print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)
    print('-' * 50)

    t0_tm = time()
    A_bar, B_bar, C_bar, S_bar, z_bar = integrator.calculate_discretization(X, U, sigma)
    print(format_line('Time for transition matrices', time() - t0_tm, 's'))

    problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                           X_last=X, U_last=U, sigma_last=sigma,
                           weight_sigma=w_sigma, weight_nu=w_nu, weight_delta=w_delta)

    while True:
        info = problem.solve(verbose=verbose_solver, solver=solver)
        print(format_line('Solver Error', info['solver_error']))

        X = problem.get_variable('X')
        U = problem.get_variable('U')
        sigma = problem.get_variable('sigma')

        delta_norm = problem.get_variable('delta_norm')
        sigma_norm = problem.get_variable('sigma_norm')
        nu_norm = np.linalg.norm(problem.get_variable('nu'), np.inf)

        print(format_line('delta_norm', delta_norm))
        print(format_line('sigma_norm', sigma_norm))
        print(format_line('nu_norm', nu_norm))

        if delta_norm < 1e-3 and sigma_norm < 1e-3 and nu_norm < 1e-7:
            converged = True

        w_delta *= 1.5

        problem.set_parameters(weight_delta=w_delta)

        break

    print('')
    print(format_line('Time for iteration', time() - t0_it, 's'))
    print('')

    all_X.append(X)
    all_U.append(U)

    if converged:
        print(f'Converged after {it + 1} iterations.')
        break

all_X = np.stack(all_X)
all_U = np.stack(all_U)

# save trajectory to file for visualization
save_arrays('visualization/trajectory/all/', {'X': all_X, 'U': all_U, 'sigma': sigma})

# plot trajectory
plot3d(all_X, all_U)
