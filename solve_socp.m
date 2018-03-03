function [x, u, s] = solve_socp(A, B, C, S, z, X_last, U_last, sigma_last, x_init, K)

w_nu = 1e5;
w_delta = 1e-3;
w_delta_sigma = 1e-1;

gamma_gs = 20;
delta_max = 20;
theta_max = 90;
w_max = 60;
m_dry = 1;
T_min = 0.4;
T_max = 5;


cvx_begin
% -------------------------------------------------------------------------

variable X(K, 14)
variable U(K, 3)
variable sig
variable nu(14 * (K-1))
variable delta(K)
variable delta_s

minimize(sig + w_nu * norm(nu, 1) + w_delta * norm(delta, 2) + w_delta_sigma * norm(delta_s, 1))
subject to

% Boundary conditions:
X(1, 1) == x_init(1);
X(1, 2:4) == x_init(2:4);
X(1, 5:7) == x_init(5:7);
X(1, 12:14) == x_init(12:14);

X(K, 2:4) == 0;
X(K, 5:7) == [-1e-1 0 0];
X(K, 8:11) == [1 0 0 0];
X(K, 12:14) == 0;

U(K, 2) == 0;
U(K, 3) == 0;

% Dynamics:
for k = 1:K-1
    X(k+1, :)' == squeeze(A(k, :, :)) * X(k, :)' + squeeze(B(k, :, :)) * U(k, :)' + squeeze(C(k, :, :)) * U(k+1, :)' + S(k, :)' * sig + z(k, :)' + nu(14 * k - 13 : 14 * k)
end

% State constraints:
X(:, 1) >= m_dry;
for k = 1:K
    tan(deg2rad(gamma_gs)) * norm([X(k, 3) X(k, 4)], 2) <= X(k, 2);
    cos(deg2rad(theta_max)) <= 1 - 2 * (X(k, 9)^2 + X(k, 10)^2);
    norm(X(k, 12:14), 2) <= deg2rad(w_max);
end

% Control constraints:
for k = 1:K
    T_min <= U_last(k, :)' / norm(U_last(k, :)) * U(k, :);
    norm(U(k, :), 2) <= T_max;
    cos(deg2rad(delta_max)) * norm(U(k, :)) <= U(k, 1);
end

% Trust regions:
for k = 1:K
    dx = X(k, :) - X_last(k, :);
    du = U(k, :) - U_last(k, :);
    dx*dx' + du*du' <= delta(k);
end
ds = sig - sigma_last;
norm(ds, 1) <= delta_s;

% -------------------------------------------------------------------------
cvx_end

x = X;
u = U;
s = sig;


end

