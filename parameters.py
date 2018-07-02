# Trajectory points
K = 50

# Max solver iterations
iterations = 100

# Weight constants
# flight time
w_sigma = 1.0
# virtual control
w_nu = 1e3

# initial trust region radius
r_delta = 1
# trust region variables
rho_0 = 0.0
rho_1 = 0.25
rho_2 = 0.9
alpha = 2.0
beta = 3.2

solver = ['ECOS', 'MOSEK'][1]

