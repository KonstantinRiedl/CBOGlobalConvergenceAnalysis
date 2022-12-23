% Consensus based optimization with memory effects (CBOmemorygradient)
%
% This function performs CBO with memory effects.
% 
% 
% [xstar_approx] = CBOmemorygradient(E, grad_E, parametersCBOmemorygradient, X0)
% 
% input:    E                           = objective function E (as anonymous function)
%           grad_E                      = gradient of objective function E (as anonymous function)
%           parametersCBOmemorygradient = suitable parameters for CBOmemorygradient
%                                       = [T, dt, N, lambda1, lambda2, gamma, learning_rate, anisotropic1, sigma1, anisotropic2, sigma2, alpha]
%               - T                     = time horizon
%               - dt                    = time step size
%               - N                     = number of particles
%               - memory                = whether memory effects are used or not
%               - lambda1               = drift towards global and in-time best (consensus) parameter
%               - lambda2               = drift towards in-time best parameter
%               - gamma                 = gradient drift parameter
%               - l._r.                 = learning rate associated with gradient drift
%               - anisotropic1          = exploration/noise 1 type
%               - sigma1                = exploration/noise parameter 1
%               - anisotropic2          = exploration/noise 2 type
%               - sigma2                = exploration/noise parameter 2
%               - alpha                 = weight/temperature parameter alpha
%           X0                          = initial position of the particles
%           
% output:   xstar_approx                = approximation to xstar
%

function [xstar_approx] = CBOmemorygradient(E, grad_E, parametersCBOmemorygradient, X0)

% get parameters
T = parametersCBOmemorygradient('T');
dt = parametersCBOmemorygradient('dt');
memory = parametersCBOmemorygradient('memory');
alpha = parametersCBOmemorygradient('alpha');
anisotropic1 = parametersCBOmemorygradient('anisotropic1');
anisotropic2 = parametersCBOmemorygradient('anisotropic2');

% initialization
X = X0;
Y = X;

% % CBOmemorygradient
% if ~memory
%     if anisotropic1
%         disp('CBO without memory and with ANisotropic diffusion used.')
%     else
%         disp('CBO without memory and with isotropic diffusion used.')
%     end
% else
%     if anisotropic1 && anisotropic2
%         disp('CBO with memory and ANisotropic diffusion used.')
%     elseif ~anisotropic1 && anisotropic2
%         disp('CBO with memory and ANisotropic diffusion only locally used.')
%     elseif anisotropic1 && ~anisotropic2
%         disp('CBO with memory and ANisotropic diffusion only globally used.')
%     else
%         disp('CBO with memory and isotropic diffusion used.')
%     end
% end

for k = 1:T/dt
    
    % % CBO iteration
    % compute current consensus point y_alpha
    y_alpha = compute_yalpha(E, alpha, Y);

    % position updates of one iteration of CBO
    [X, Y] = CBOmemorygradient_update(E, grad_E, parametersCBOmemorygradient, y_alpha, X, Y);
    
end

y_alpha = compute_yalpha(E, alpha, Y);
xstar_approx = y_alpha;

end
