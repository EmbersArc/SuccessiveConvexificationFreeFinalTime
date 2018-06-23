import time
import numpy as np

from model_6dof import Model_6DoF
from parameters import *
from discretization import Discretize
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
w_sigma = 0

# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
all_X = [X]
all_U = [U]

disc = Discretize(m, K)
prob = SCProblem(m, K)

for it in range(iterations):
    t0_it = time.time()
    print('-' * 50)
    print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)
    print('-' * 50)

    t0_tm = time.time()
    A_bar, B_bar, C_bar, Sigma_bar, z_bar = disc.calculate(X, U, sigma)
    print(format_line('Time for transition matrices', time.time() - t0_tm, 's'))

    # pass parameters to model
    prob.update_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, Sigma_bar=Sigma_bar, z_bar=z_bar,
                           X_last=X, U_last=U, Sigma_last=sigma,
                           E=m.E,
                           weight_sigma=w_sigma, weight_delta=w_delta,
                           weight_delta_sigma=w_delta_sigma, weight_nu=w_nu)

    # solve problem
    info = prob.solve(verbose=False, solver=solver)

    print(format_line('Setup Time', info['setup_time'], 's'))
    print(format_line('Solver Time', info['solver_time'], 's'))
    print(format_line('Solver Iterations', info['iterations']))
    print(format_line('Solver Error', info['solver_error']))

    # update values
    X, U, sigma = prob.get_solution()

    # check if solution has converged
    converged, info = prob.check_convergence(delta_tol, nu_tol)

    print('\n')
    print(format_line('Trust Region Cost', info['delta_norm']))
    print(format_line('Virtual Control Cost', info['nu_norm']))
    print(format_line('Total Time', info['sigma'], 's'))

    all_X.append(X)
    all_U.append(U)

    print('\n')
    print(format_line('Time for iteration', time.time() - t0_it, 's'))
    print('\n')

    if converged and w_sigma == 0:
        w_sigma = 1
        w_delta_sigma = 1e-1
        print("Initialization has converged.")
    elif converged and w_sigma == 1:
        print('Converged after', it + 1, 'iterations.')
        save_arrays('visualization/trajectory/final/', {'X': m.x_redim(X), 'U': m.u_redim(U), 'sigma': sigma})

        break

    w_delta *= 1.2

all_X = np.stack(all_X)
all_U = np.stack(all_U)

# save trajectory to file for visualization
np.save('visualization/trajectory/X.npy', all_X)
np.save('visualization/trajectory/U.npy', all_U)
np.save('visualization/trajectory/sigma.npy', sigma)

# plot trajectory
plot3d(all_X, all_U)
