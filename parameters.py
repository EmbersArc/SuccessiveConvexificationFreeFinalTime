# Trajectory points
K = 50

# Max solver iterations
iterations = 100

# Weight constants
# flight time
w_sigma = 1
# virtual control
w_nu = 1e4

# initial trust region radius
tr_radius = 1
# trust region variables
rho_0 = 0.0
rho_1 = 0.25
rho_2 = 0.9
alpha = 1.4
beta = 1.8

solver = ['ECOS', 'MOSEK'][0]
