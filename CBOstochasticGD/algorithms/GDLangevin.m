% Gradient Descent and Langevin Dynamics
%
% This function performs Gradient Descent and the Langevin Dynamics (SGD).
% 
% 
% [vstar_approx] = GDLangevin(E, ~, parametersGD, V0)
% 
% input:    E             = objective function E (as anonymous function)
%           parametersCH  = suitable parameters for CH
%                         = [K, lambda, gamma, learning_rate, sigma, alpha]
%               - K       = number of time steps
%               - dt      = time step size
%               - l._r.   = learning rate
%               - sigma   = noise parameter in Langevin dynamics
%           V0            = initial position
%           
% output:   vstar_approx  = approximation to vstar
%

function [vstar_approx] = GDLangevin(E, grad_E, parametersGD, V0)

% get parameters
d = size(V0,1);
K = parametersGD('K');
dt = parametersGD('dt');
learning_rate = parametersGD('learning_rate');
sigma = parametersGD('sigma');

% initialization
V = V0;

% GD / annealed Langevin
for k = 1:K
    
    % % GD / annealed Langevin iteration
    % gradient computation
    if isnan(grad_E(0))
        h = 10^-5;
        gradE = zeros(d,1);
        for i = 1:d
            dV = zeros(d,1);
            dV(i,:) = ones(1,1);
            gradE(i,:) = (E(V+h*dV)-E(V-h*dV))/(2*h);
        end
    else
        gradE = grad_E(V);
    end

    % position updates of one iteration of GD / annealed Langevin
    V = V - learning_rate*gradE*dt + sigma/log(k+1)*sqrt(dt)*randn(d,1);

end

vstar_approx = V;

end