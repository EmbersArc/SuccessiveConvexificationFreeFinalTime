clear
clc
addpath('helper_functions')

% input and state
u = sym('u',[3,1]);
x = sym('x',[14,1]);

% gravity vector
g_I = sym('g_I',[3,1]);
% vector from CoM to thrust
r_T_B = sym('r_T_B',[3,1]);
% inertia
J_B = diag(sym('J_B',[3,1]));
% fuel consumption rate
alpha_m = sym('alpha_m');
% rotation matrix from I to B
C_I_B = transpose(quat2rotmsym(x(8:11)));

% nonlinear state transition
m_dot = -alpha_m * norm(u);
r_dot = x(5:7);
v_dot = 1/x(1) * C_I_B * u + g_I;
q_dot = 1/2 * omega(x(12:14)) * x(8:11);
w_dot = J_B \ (skew(r_T_B) * u - skew(x(12:14)) * J_B * x(12:14));

f = [m_dot; r_dot; v_dot; q_dot; w_dot];

% linearized system
A = jacobian(f,x);
B = jacobian(f,u);

f = simplify(f)
A = simplify(A)
B = simplify(B)

ccode(A,'file','A')
ccode(B,'file','B')
ccode(f,'file','f')
