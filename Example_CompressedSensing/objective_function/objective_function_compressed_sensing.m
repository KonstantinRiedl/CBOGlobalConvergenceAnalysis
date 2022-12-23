% Objective-function function
%
% This function returns the objective function E as an anonymous function
% for the compressed sensing problem.
%
% The function E maps columnwise from R^{d\times N} to R^N, i.e., for a 
% matrix in R^{d\times N} the function is applied to every column and 
% returns a row vector in R^N (matrix in R^{1\times N})
% 
% 
% [E, grad_E, A, Y] = objective_function_compressed_sensing(xstar, M, p, noise_to_signal_ratio)
% 
% input:    xstar                 = signal (to be recovered)
%           M                     = number of measurements
%           p                     = power of sparsity-enforcing norm
%           noise_to_signal_ratio = noise level in measurements
%           
% output:   E                     = objective function E of compressed sensing problem
%                                   (as anonymous function)
%           grad_E                = gradient of objective function E of compressed sensing problem
%                                   (as anonymous function)
%           A                     = measurement matrix
%           Y                     = measurements
%


function [E, grad_E, A, Y] = objective_function_compressed_sensing(xstar, M, p, noise_to_signal_ratio)

d = size(xstar, 1);
s = sum(xstar > 0);

% measurement noise
w = abs(randn(M, 1));
w = noise_to_signal_ratio * norm(xstar)/norm(w) * w;

% % measurements
% random measurement matrix
A = 1/sqrt(M) * randn(M, d);
Y = A * xstar + w;


% objective E (E is a function mapping columnwise from R^{d\times N} to R^N)
if p==1
    penalty = 0.25;
elseif p==0.8
    penalty = 0.1;
elseif p==0.5
    penalty = 0.01;
else
    penalty = 0.25;
end
E = @(x) vecnorm(A*x - Y, 2, 1).^2 + penalty*vecnorm(x, p, 1);

% gradient of objective E
if p==1
    grad_E = @(x) 2*A'*(A*x - Y) + penalty*sign(x);
else
    grad_E = @(x) 2*A'*(A*x - Y) + penalty*p*x.*abs(x)^(p-2);
end

end