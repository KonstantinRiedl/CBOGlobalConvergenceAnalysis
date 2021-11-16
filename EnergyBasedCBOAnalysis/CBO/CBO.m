% Consensus based optimization (CBO)
%
% This function performs CBO.
% 
% 
% [vstar_approx] = CBO(E, parametersCBO, V0)
% 
% input:    E             = objective function E (as anonymous function)
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

function [vstar_approx] = CBO(E, parametersCBO, V0)

% get parameters
T = parametersCBO('T');
dt = parametersCBO('dt');
alpha = parametersCBO('alpha');
anisotropic = parametersCBO('anisotropic');

% initialization
V = V0;

% CBO
if anisotropic
	disp('CBO with ANisotropic diffusion used.')
else
	disp('CBO with isotropic diffusion used.')
end
for k = 1:T/dt
    
    % % CBO iteration
    % compute current consensus point v_alpha
    v_alpha = compute_valpha(E, alpha, V);

    % position updates of one iteration of CBO
    V = CBO_update(E, parametersCBO, v_alpha, V);
    
end

v_alpha = compute_valpha(E, alpha, V);
vstar_approx = v_alpha;

end
