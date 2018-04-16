import pickle
import time
import numpy as np
import cvxpy as cvx

from model_6dof import Model_6DoF
from parameters import *
from discretization import Discretize
from visualization.plot3d import plot3d

m = Model_6DoF()

# state and input
X = np.empty(shape=[m.n_x, K])
U = np.empty(shape=[m.n_u, K])

# CVX ------------------------------------------------------------------------------------------------------------------
print("Setting up problem.")
# Variables:
X_ = cvx.Variable((m.n_x, K))
U_ = cvx.Variable((m.n_u, K))
sigma_ = cvx.Variable()
nu_ = cvx.Variable((m.n_x * (K - 1)))
delta_ = cvx.Variable(K)
delta_s_ = cvx.Variable()

# Parameters:
A_bar_ = cvx.Parameter((m.n_x * m.n_x, K - 1))
B_bar_ = cvx.Parameter((m.n_x * m.n_u, K - 1))
C_bar_ = cvx.Parameter((m.n_x * m.n_u, K - 1))
Sigma_bar_ = cvx.Parameter((m.n_x, K - 1))
z_bar_ = cvx.Parameter((m.n_x, K - 1))
X_last_ = cvx.Parameter((m.n_x, K))
U_last_ = cvx.Parameter((m.n_u, K))
sigma_last_ = cvx.Parameter(nonneg=True)
w_delta_ = cvx.Parameter(nonneg=True)
w_nu_ = cvx.Parameter(nonneg=True)
w_delta_sigma_ = cvx.Parameter(nonneg=True)

constraints = []

# Dynamics:
constraints += [
    X_[:, k + 1] ==
    cvx.reshape(A_bar_[:, k], (m.n_x, m.n_x)) * X_[:, k]
    + cvx.reshape(B_bar_[:, k], (m.n_x, m.n_u)) * U_[:, k]
    + cvx.reshape(C_bar_[:, k], (m.n_x, m.n_u)) * U_[:, k + 1]
    + Sigma_bar_[:, k] * sigma_
    + z_bar_[:, k]
    + nu_[k * m.n_x:(k + 1) * m.n_x]
    for k in range(K - 1)
]

# Trust regions:
dx = X_ - X_last_
du = U_ - U_last_
ds = sigma_ - sigma_last_
constraints += [
    cvx.square(cvx.norm(dx, axis=0)) + cvx.square(cvx.norm(du, axis=0)) <= delta_,
    cvx.norm(ds) <= delta_s_
]

constraints += m.get_constraints(X_, U_, X_last_, U_last_)

# Objective:
objective = cvx.Minimize(
    sigma_ + w_nu_ * cvx.norm(nu_, 1) + w_delta_ * cvx.norm(delta_) + w_delta_sigma_ * cvx.norm(delta_s_, 1)
)

prob = cvx.Problem(objective, constraints)

print("Problem is " + ("valid." if prob.is_dcp() else "invalid."))
# CVX ------------------------------------------------------------------------------------------------------------------

# INITIALIZATION--------------------------------------------------------------------------------------------------------

sigma = m.t_f_guess
m.initialize(X, U)

# START SUCCESSIVE CONVEXIFICATION--------------------------------------------------------------------------------------
all_X = [X]
all_U = [U]

disc = Discretize(m, dt, K)

for it in range(iterations):
    t0_it = time.time()
    print("\n")
    print("------------------")
    print("-- Iteration", str(it + 1).zfill(2), "--")
    print("------------------")

    print("Calculating new transition matrices.")
    t0_tm = time.time()
    A_bar, B_bar, C_bar, Sigma_bar, z_bar = disc.calculate(X, U, sigma)
    t_it = time.time() - t0_tm

    # CVX --------------------------------------------------------------------------------------------------------------
    # pass parameters to model (CVXPY uses Fortran order)
    A_bar_.value = A_bar
    B_bar_.value = B_bar
    C_bar_.value = C_bar
    Sigma_bar_.value = Sigma_bar
    z_bar_.value = z_bar
    X_last_.value = X
    U_last_.value = U
    sigma_last_.value = sigma
    w_delta_.value = w_delta
    w_nu_.value = w_nu
    w_delta_sigma_.value = w_delta_sigma

    print("Solving problem.")
    try:
        prob.solve(verbose=True, solver='ECOS', max_iters=100)
    except cvx.error.SolverError:
        # can sometimes ignore a solver error
        pass

    info = prob.solver_stats
    print("Time for transition matrices:", t_it)
    print("Time for setup:", info.setup_time)
    print("Time for solver:", info.solve_time)
    print("Time for iteration:", time.time() - t0_it)
    print("\n")

    # CVX --------------------------------------------------------------------------------------------------------------
    # update values
    X = X_.value
    U = U_.value
    sigma = sigma_.value

    all_X.append(X)
    all_U.append(U)

    # print status
    delta_norm = np.linalg.norm(delta_.value)
    nu_norm = np.linalg.norm(nu_.value, ord=1)
    print("Flight time:", sigma_.value)
    print("Delta_norm:", delta_norm)
    print("Nu_norm:", nu_norm)
    if delta_norm < delta_tol and nu_norm < nu_tol:
        print("Converged after", it + 1, "iterations.")
        break

all_X = np.stack(all_X)
all_U = np.stack(all_U)

# save trajectory to file for visualization
pickle.dump(all_X, open("visualization/trajectory/X.p", "wb"))
pickle.dump(all_U, open("visualization/trajectory/U.p", "wb"))
pickle.dump(sigma, open("visualization/trajectory/sigma.p", "wb"))

# plot trajectory
plot3d(all_X, all_U)
