% Comparison between current positions X and in-time best positions Y
%
% This function compares the current positions X with the in-time best 
% positions Y with respect to the objective function E using the formula
%
%    s_betaXY = 1/2 * (1 + tanh(beta*(E(Y)-E(X)))).
%
% It returns values between 0 and 1 as follows: 
%    closer to 0   if Y remains the better position (i.e. E(X)>E(Y))
%    closer to 1   if X is the better position, thus the new Y (i.e. E(X)<E(Y))
% Those valuing ensures that in the limiting case beta = inf, the SDE 
%    dY_t = (X_t-Y_t)s_betaXY
% encodes the update rule Y_new = X if E(X)<E(Y_old) and Y_old else.
% 
% 
% [s_betaXY] = S_beta(E, beta, X, Y, objective_function_X, objective_function_Y)
% 
% input:    E                    = objective function E (as anonymous function)
%           beta                 = weight/temperature parameter alpha
%           X                    = current positions x_i of particles
%           Y                    = in-time best positions y_i of particles
%           objective_function_X = objective values of positions X
%           objective_function_Y = objective values of positions Y
%           
% output:   s_betaXY             = approximated indicator 
%

function [s_betaXY] = S_beta(E, beta, X, Y, objective_function_X, objective_function_Y)

if nargin == 6
    s_betaXY = 1/2*(1+tanh(beta*(objective_function_Y-objective_function_X)));
else
    s_betaXY = 1/2*(1+tanh(beta*(E(Y)-E(X))));
end


clear E beta X Y

end