% Position updates of one iteration of consensus based optimization with memory effects (CBOmemorygradient)
%
% This function performs the position updates of one iteration of CBO with memory effects.
% 
% 
% [X, Y, objective_function_Y] = CBOmemorygradient_update(E, grad_E, parametersCBO, y_alpha, X, Y)
% 
% input:    E                  = objective function E (as anonymous function)
%           grad_E             = gradient of objective function E (as anonymous function)
%           parametersCBO      = suitable parameters for CBO
%                              = [T, dt, N, lambda1, lambda2, gamma, learning_rate, anisotropic1, sigma1, anisotropic2, sigma2, alpha]
%               - T            = time horizon
%               - dt           = time step size
%               - N            = number of particles
%               - memory       = whether memory effects are used or not
%               - lambda1      = drift towards in-time best parameter
%               - lambda2      = drift towards global and in-time best (consensus) parameter
%               - gamma        = gradient drift parameter 
%               - l._r.        = learning rate associated with gradient drift
%                                (lambda3 = gamma*l._r.)
%               - anisotropic1 = exploration/noise 1 type
%               - sigma1       = exploration/noise parameter 1
%               - anisotropic2 = exploration/noise 2 type
%               - sigma2       = exploration/noise parameter 2
%               - alpha        = weight/temperature parameter alpha
%           y_alpha            = current empirical consensus point
%           X                  = former positions of the particles
%           Y                  = former best positions of the particles
%           
% output:   X                  = positions of the particles afterwards
%           Y                  = best positions of the particles afterwards
%

function [X, Y, objective_function_Y] = CBOmemorygradient_update(E, grad_E, parametersCBOmemorygradient, y_alpha, X, Y, objective_function_Y)

% get parameters
d = size(X,1);
dt = parametersCBOmemorygradient('dt');
N = size(X,2);
memory = parametersCBOmemorygradient('memory');
lambda1 = parametersCBOmemorygradient('lambda1');
lambda2 = parametersCBOmemorygradient('lambda2');
gamma = parametersCBOmemorygradient('gamma');
learning_rate = parametersCBOmemorygradient('learning_rate');
anisotropic1 = parametersCBOmemorygradient('anisotropic1');
sigma1 = parametersCBOmemorygradient('sigma1');
anisotropic2 = parametersCBOmemorygradient('anisotropic2');
sigma2 = parametersCBOmemorygradient('sigma2');
beta = parametersCBOmemorygradient('beta');
kappa = parametersCBOmemorygradient('kappa');


% % particle update step (according to SDE)
% drift towards the global and in-time best position y_alpha
X = X - lambda1*(X-y_alpha*ones(1,N))*dt;
% drift towards the in-time best particle position (individually for each particle, i.e., locally)
if memory
    X = X - lambda2*(X-Y)*dt;
end
% exploration/noise terms
dB1 = randn(d,N) + (1-isreal(X))*1i*randn(d,N);
if anisotropic1
    X = X + sigma1*abs(X-y_alpha*ones(1,N))*sqrt(dt).*dB1;
else
    X = X + sigma1*vecnorm(X-y_alpha*ones(1,N),2,1)*sqrt(dt).*dB1;
end
if memory
    dB2 = randn(d,N) + (1-isreal(X))*1i*randn(d,N);
    if anisotropic2
        X = X + sigma2*abs(X-Y)*sqrt(dt).*dB2;
    else
        X = X + sigma2*vecnorm(X-Y,2,1)*sqrt(dt).*dB2;
    end
end

% gradient drift term
if gamma>0
    if isnan(grad_E(0))
        h = 10^-3;
        gradE = zeros(d,N);
        for i = 1:d
            dX = h*zeros(d,N);
            dX(i,:) = ones(1,N);
            gradE(i,:) = (E(X+h*dX)-E(X-h*dX))/(2*h);
        end
    else
        gradE = grad_E(X);
    end
    X = X - gamma*learning_rate*gradE*dt;
end

% % update of the in-time best particle positions (for each particle)
if memory
    if strcmp(beta, 'inf') || beta==-1
        if nargin == 7
            objective_function_X = E(X);
            Y = Y + (X-Y).*double((objective_function_Y-objective_function_X)>0);
            objective_function_Y = min(objective_function_Y, objective_function_X);
        else
            Y = Y + (X-Y).*double(E(Y)-E(X)>0);
            objective_function_Y = nan;
        end
    else
        if nargin == 7
            objective_function_X = E(X);
            Y = Y + kappa*(X-Y).*S_beta(E, beta, X, Y, objective_function_X, objective_function_Y)*dt;
            objective_function_Y = E(Y);
        else
            Y = Y + kappa*(X-Y).*S_beta(E, beta, X, Y)*dt;
            objective_function_Y = nan;
        end
    end
else
    Y = X;
    if nargin == 7
        objective_function_Y = E(X);
    else
        objective_function_Y = nan;
    end
end

end
