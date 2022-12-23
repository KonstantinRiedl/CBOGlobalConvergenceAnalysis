% Global and in-time best position
%
% This function computes the global and in-time best position based on the
% positions of a given number N of particles y_i, which themselves are the 
% in-time best positions of the particles x_i, according to the formula
%
%    y_alpha = sum_i=1^N y_i*w_alpha(y_i)/(sum_j=1^N w_alpha(y_j)),
%
% where w_alpha(y) = exp(-alpha*E(y)).
% For (numerical) stability reasons, we modify the definition of w_alpha(y)
% to w_alpha(y) = exp(-alpha*(E(y)-Emin)), where Emin is the minimal energy
% among the particles. We note, that this does not influence the
% theoretical value of y_alpha.
% 
% 
% [y_alpha] = compute_yalpha(E, alpha, Y, objective_function_Y)
% 
% input:    E                    = objective function E (as anonymous function)
%           alpha                = weight/temperature parameter alpha
%           Y                    = positions y_i of particles used for computation
%                                  of the global and in-time best position y_alpha
%           objective_function_Y = objective values of positions of particle
%           
% output:   y_alpha              = global and in-time best position
%

function [y_alpha] = compute_yalpha(E, alpha, Y, objective_function_Y)

% energies of the individual particles
if nargin==4
    Es = objective_function_Y;
else
    Es = E(Y);
end

% minimal energy among the individual particles
Emin = min(Es);

% computation of current empirical consensus point y_alpha
w_alpha = exp(-alpha*(Es-Emin));
y_alpha = sum((Y.*w_alpha),2);
y_alpha = 1/sum(w_alpha)*y_alpha;

clear E Es Emin alpha Y w_alpha

end
