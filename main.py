import pickle
import time
import numpy as np
import cvxpy as cvx

from model_6dof import Model_6DoF
from parameters import *
from discretization import Discretize
from visualization.plot3d import plot3d
from scproblem import SCProblem

m = Model_6DoF()

# state and input
X = np.empty(shape=[m.n_x, K])
U = np.empty(shape=[m.n_u, K])

# INITIALIZATION--------------------------------------------------------------------------------------------------------
sigma = m.t_f_guess
m.initialize(X, U)

# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
all_X = [X]
all_U = [U]

disc = Discretize(m, dt, K)
prob = SCProblem(m, K)

for it in range(iterations):
    t0_it = time.time()
    print('\n')
    print('------------------')
    print(f'-- Iteration {str(it + 1).zfill(2)} --')
    print('------------------')

    print('Calculating new transition matrices.')
    t0_tm = time.time()
    A_bar, B_bar, C_bar, Sigma_bar, z_bar = disc.calculate(X, U, sigma)
    t_tm = time.time() - t0_tm
    print('Time for transition matrices:', t_tm)

    # if it > 5:
    #     w_delta *= 1.2

    # pass parameters to model
    prob.update_values(A_bar, B_bar, C_bar, Sigma_bar, z_bar, X, U, sigma, w_delta, w_nu, w_delta_sigma)

    print('Solving problem.')
    prob.solve(verbose=True, solver='MOSEK')
    print(prob.get_solver_stats())

    print('Time for iteration:', time.time() - t0_it)

    # update values
    X, U, sigma = prob.get_solution()

    all_X.append(X)
    all_U.append(U)

    converged = prob.check_convergence(delta_tol, nu_tol)

    print(prob.get_convergence_info())

    # print status
    if converged:
        print('Converged after', it + 1, 'iterations.')
        break

all_X = np.stack(all_X)
all_U = np.stack(all_U)

# save trajectory to file for visualization
pickle.dump(all_X, open('visualization/trajectory/X.p', 'wb'))
pickle.dump(all_U, open('visualization/trajectory/U.p', 'wb'))
pickle.dump(sigma, open('visualization/trajectory/sigma.p', 'wb'))

# plot trajectory
plot3d(all_X, all_U)
