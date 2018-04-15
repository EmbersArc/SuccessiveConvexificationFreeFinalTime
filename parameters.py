# Trajectory points
K = 50
dt = 1 / (K - 1)

# Max solver iterations
iterations = 15

# Weight constants
w_nu = 1e5  # 1e5
w_delta = 1e-3  # 1e-3
w_delta_sigma = 1e-1  # 1e-1

# Exit conditions
nu_tol = 1e-8
delta_tol = 1e-3
