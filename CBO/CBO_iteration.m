% One iteration of consensus based optimization (CBO)
%
% This function performs one iteration of CBO.
% 
% 
% [V,v_alpha] = CBO_iteration(E,parametersCBO,V)
% 
% input:    dt            = time step size
%           E             = objective function E (as anonymous function)
%           parametersCBO = suitable parameters for CBO
%                         = [T, dt, N, lambda, gamma, learning_rate, sigma, alpha]
%               - T       = time horizon
%               - dt      = time step size
%               - N       = number of particles
%               - lambda  = consensus drift parameter
%               - gamma   = gradient drift parameter
%               - l._r.   = learning rate associated with gradient drift
%               - sigma   = exploration/noise parameter
%               - alpha   = weight/temperature parameter alpha
%           V             = former positions of the particles
%           
% output:   V             = positions of the particles afterwards
%           v_alpha       = current empirical consensus point
%

function [V,v_alpha] = CBO_iteration(E,parametersCBO,V)

% get parameters
[d,~] = size(V);
dt = parametersCBO('dt');
N = parametersCBO('N');
lambda = parametersCBO('lambda');
gamma = parametersCBO('gamma');
learning_rate = parametersCBO('learning_rate');
sigma = parametersCBO('sigma');
alpha = parametersCBO('alpha');


% compute current consensus point v_alpha
v_alpha = compute_valpha(E,alpha,V);

% Brownian motion for exploration term
dB = randn(d,N);


% % particle iteration step (according to SDE)

% consensus drift and exploration term
V = V - lambda*(V-v_alpha*ones(1,N))*dt + sigma*abs(V-v_alpha*ones(1,N))*sqrt(dt).*dB;

% gradient drift term
h = 10^-3;
gradE = zeros(d,N);
for i = 1:d
    dV = h*zeros(d,N);
    dV(i,:) = ones(1,N);
    gradE = (E(V+h*dV)-E(V-h*dV))/(2*h);
end
V = V - gamma*learning_rate*gradE*dt;

end
