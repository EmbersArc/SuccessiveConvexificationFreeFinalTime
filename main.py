import time
import numpy as np

from model_6dof import Model_6DoF
from parameters import *
from discretization import Discretize
from visualization.plot3d import plot3d
from scproblem import SCProblem


def format_line(name, value, unit=""):
    name += ":"
    if isinstance(value, (float, np.ndarray)):
        value = f"{value:{0}.{2}}"

    return f"{name.ljust(42)}{value}{unit}"


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

disc = Discretize(m, K)
prob = SCProblem(m, K)

for it in range(iterations):
    t0_it = time.time()
    print('-' * 50)
    print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)
    print('-' * 50)

    t0_tm = time.time()
    A_bar, B_bar, C_bar, Sigma_bar, z_bar = disc.calculate(X, U, sigma)
    print(format_line("Time for transition matrices", time.time() - t0_tm, "s"))

    w_delta *= 1.2

    # pass parameters to model
    prob.update_values(A_bar, B_bar, C_bar, Sigma_bar, z_bar, X, U, sigma, w_delta, w_nu, w_delta_sigma, m.E)

    # solve problem
    info = prob.solve(verbose=False, solver='ECOS')

    print(format_line("Setup Time", info["setup_time"], "s"))
    print(format_line("Solver Time", info["solver_time"], "s"))
    print(format_line("Solver Iterations", info["iterations"]))
    print(format_line("Solver Error", info["solver_error"]))

    # update values
    X, U, sigma = prob.get_solution()

    # save solution for plotting
    all_X.append(X)
    all_U.append(U)

    # check if solution has converged
    converged, info = prob.check_convergence(delta_tol, nu_tol)

    print("\n")
    print(format_line("Trust Region Cost", info["delta_norm"]))
    print(format_line("Virtual Control Cost", info["nu_norm"]))
    print(format_line("Total Time", info["sigma"]))

    print("\n")
    print(format_line("Time for iteration", time.time() - t0_it, "s"))

    print("\n")

    if converged:
        print('Converged after', it + 1, 'iterations.')
        break

all_X = np.stack(all_X)
all_U = np.stack(all_U)

# save trajectory to file for visualization
np.save('visualization/trajectory/X.npy', all_X)
np.save('visualization/trajectory/U.npy', all_U)
np.save('visualization/trajectory/sigma.npy', sigma)

# plot trajectory
plot3d(all_X, all_U)
