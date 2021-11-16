% Consensus point computation
%
% This function computes the current consensus point based on the positions
% of a given number N of particles v_i according to the formula
%
%    v_alpha = sum_i=1^N v_i*w_alpha(v_i)/(sum_j=1^N w_alpha(v_j)),
%
% where w_alpha(v) = exp(-alpha*E(v)).
% For (numerical) stability reasons, we modify the definition of w_alpha(v)
% to w_alpha(v) = exp(-alpha*E(v)-Emin), where Emin is the minimal energy
% among the particles. We note, that this does not influence the 
% theoretical value of v_alpha.
% 
% 
% [v_alpha] = compute_valpha(E, alpha, V)
% 
% input:    E             = objective function E (as anonymous function)
%           alpha         = weight/temperature parameter alpha
%           V             = positions v_i of particles used for computation
%                           of current empirical consensus point v_alpha
%           
% output:   v_alpha       = current empirical consensus point
%

function [v_alpha] = compute_valpha(E, alpha, V)

% energies of the individual particles
Es = E(V);

% minimal energy among the individual particles
Emin = min(Es);

% computation of current empirical consensus point v_alpha
w_alpha = exp(-alpha*(Es-Emin));
v_alpha = sum((V.*w_alpha),2);
v_alpha = 1/sum(w_alpha)*v_alpha;

end
