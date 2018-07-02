from time import time
import numpy as np

from model_6dof import Model_6DoF
from parameters import *
from discretization import Integrator
from visualization.plot3d import plot3d
from scproblem import SCProblem
from utils import format_line, save_arrays

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
                           E=m.E,
                           radius_trust_region=r_delta,
                           weight_sigma=w_sigma, weight_nu=w_nu)

    while True:
        # solve problem
        info = problem.solve(verbose=False, solver=solver)

        # print(format_line('Setup Time', info['setup_time'], 's'))
        # print(format_line('Solver Time', info['solver_time'], 's'))
        # print(format_line('Solver Iterations', info['iterations']))
        print(format_line('Solver Error', info['solver_error']))

        # update values
        X = problem.get_variable('X')
        U = problem.get_variable('U')
        sigma = problem.get_variable('sigma')

        # check if solution has converged
        linear_cost = np.linalg.norm(problem.get_variable('nu'), 1)
        nonlinear_cost = np.linalg.norm(X - integrator.integrate_nonlinear_piecewise(X, U, sigma), ord=1)

        if last_linear_cost is None:
            last_linear_cost = linear_cost
            sigma_last = sigma
            break

        actual_change = last_linear_cost - linear_cost
        predicted_change = last_linear_cost - nonlinear_cost

        print('')
        print(format_line('Virtual Control Cost', linear_cost))
        print(format_line('Total Time', sigma, 's'))
        print('')
        print(format_line('Actual change', actual_change))
        print(format_line('Predicted change', predicted_change))

        if abs(predicted_change) < 0.000001:
            converged = True
            break
        else:
            rho = actual_change / predicted_change
            if rho < rho_0:
                r_delta /= alpha
                print(f'rho={rho} too small. Solving again with radius={r_delta}')
            else:
                if rho < rho_1:
                    print('Decreasing radius.')
                    r_delta /= alpha
                elif rho >= rho_2:
                    print('Increasing radius.')
                    r_delta *= beta

                last_linear_cost = linear_cost
                break

        problem.set_parameters(radius_trust_region=r_delta)

        print('-' * 50)

    print('')
    print(format_line('Time for iteration', time() - t0_it, 's'))
    print('')

    all_X.append(X)
    all_U.append(U)

    if converged:
        print('Converged after', it + 1, 'iterations.')
        break

all_X = np.stack(all_X)
all_U = np.stack(all_U)

# save trajectory to file for visualization
save_arrays('visualization/trajectory/final/', {'X': m.x_redim(X), 'U': m.u_redim(U), 'sigma': sigma})
save_arrays('visualization/trajectory/all/', {'X': all_X, 'U': all_U, 'sigma': sigma})

# plot trajectory
plot3d(all_X, all_U)
