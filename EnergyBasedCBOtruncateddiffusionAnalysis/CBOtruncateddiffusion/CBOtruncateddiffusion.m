% Consensus based optimization with truncated diffusion
%
% This function performs CBO with truncated diffusion.
% 
% 
% [vstar_approx] = CBOtruncateddiffusion(E, grad_E, parametersCBOtruncateddiffusion, V0)
% 
% input:    E             = objective function E (as anonymous function)
%           grad_E        = gradient of objective function E (as anonymous function)
%           parametersCBOtruncateddiffusion = suitable parameters for CBOtruncateddiffusion
%                         = [T, dt, N, lambda, gamma, learning_rate, sigma, alpha]
%               - T       = time horizon
%               - dt      = time step size
%               - N       = number of particles
%               - lambda  = consensus drift parameter
%               - gamma   = gradient drift parameter
%               - l._r.   = learning rate associated with gradient drift
%               - sigma   = exploration/noise parameter
%               - M       = bound on exploration/noise term
%               - alpha   = weight/temperature parameter alpha
%           V0            = initial position of the particles
%           
% output:   vstar_approx  = approximation to vstar
%

function [vstar_approx] = CBOtruncateddiffusion(E, grad_E, parametersCBOtruncateddiffusion, V0)

% get parameters
T = parametersCBOtruncateddiffusion('T');
dt = parametersCBOtruncateddiffusion('dt');
alpha = parametersCBOtruncateddiffusion('alpha');

% initialization
V = V0;

% % CBO
% if anisotropic
% 	disp('CBO with ANisotropic diffusion used.')
% else
% 	disp('CBO with isotropic diffusion used.')
% end

for k = 1:T/dt
    
    % % CBO iteration
    % compute current consensus point v_alpha
    v_alpha = compute_valpha(E, alpha, V);

    % position updates of one iteration of CBO
    V = CBOtruncateddiffusion_update(E, grad_E, parametersCBOtruncateddiffusion, v_alpha, V);
    
end

v_alpha = compute_valpha(E, alpha, V);
vstar_approx = v_alpha;

end
