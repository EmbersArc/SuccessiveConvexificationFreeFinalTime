# Trajectory points
K = 50

# Max solver iterations
iterations = 30

# Weight constants
w_sigma = 1e-1  # difference in flight time
w_delta = 1e-3  # difference in state/input
w_nu = 1e5  # virtual control

solver = ['ECOS', 'MOSEK'][0]
verbose_solver = False
