% Consensus based optimization (CBO)
%
% This function performs CBO.
% 
% 
% [vstar_approx] = CBO(E,parametersCBO,V0)
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
%           V0            = initial position of the particles
%           
% output:   vstar_approx  = approximation to vstar
%

function [vstar_approx] = CBO(E,parametersCBO,V0)

% get parameters
T = parametersCBO('T');
dt = parametersCBO('dt');

% initialization
V= V0;

% CBO
for k = 1:T/dt
    
    [V,v_alpha] = CBO_iteration(E,parametersCBO,V);
    
end

vstar_approx = v_alpha;

end
