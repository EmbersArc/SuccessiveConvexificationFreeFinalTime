# Trajectory points
K = 50

# Max solver iterations
iterations = 50

# Weight constants
# flight time
w_sigma = 1
# virtual control
w_nu = 1e5  # 1e5
# state and input trust region
w_delta = 1e-3  # 1e-3
# time trust region
w_delta_sigma = 1e-1  # 1e-1

solver = ['ECOS', 'MOSEK'][0]

# Exit conditions
nu_tol = 1e-8
delta_tol = 1e-2
