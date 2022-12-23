% Position updates of one iteration of consensus based optimization (CBO)
%
% This function performs the position updates of one iteration of CBO.
% 
% 
% [V] = CBO_update(E, grad_E, parametersCBO, v_alpha, V)
% 
% input:    E             = objective function E (as anonymous function)
%           grad_E        = gradient of objective function E (as anonymous function)
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
%           v_alpha       = current empirical consensus point
%           V             = former positions of the particles
%           
% output:   V             = positions of the particles afterwards
%

function [V] = CBO_update(E, grad_E, parametersCBO, v_alpha, V)

% get parameters
d = size(V,1);
dt = parametersCBO('dt');
N = size(V,2);
lambda = parametersCBO('lambda');
gamma = parametersCBO('gamma');
learning_rate = parametersCBO('learning_rate');
anisotropic = parametersCBO('anisotropic');
sigma = parametersCBO('sigma');


% Brownian motion for exploration term
dB = randn(d,N) + (1-isreal(V))*1i*randn(d,N);


% % particle update step (according to SDE)
% consensus drift term
V = V - lambda*(V-v_alpha*ones(1,N))*dt;
% exploration/noise term
if anisotropic
    V = V + sigma*abs(V-v_alpha*ones(1,N))*sqrt(dt).*dB;
else
    V = V + sigma*vecnorm(V-v_alpha*ones(1,N),2,1)*sqrt(dt).*dB;
end


% gradient drift term
if gamma>0
    if isnan(grad_E(0))
        h = 10^-3;
        gradE = zeros(d,N);
        for i = 1:d
            dV = zeros(d,N);
            dV(i,:) = ones(1,N);
            gradE(i,:) = (E(V+h*dV)-E(V-h*dV))/(2*h);
        end
    else
        gradE = grad_E(V);
    end
    V = V - gamma*learning_rate*gradE*dt;
end

end
