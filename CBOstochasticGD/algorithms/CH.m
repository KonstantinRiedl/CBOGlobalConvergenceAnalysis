% Consensus Hopping
%
% This function performs the Consensus Hopping Scheme.
% 
% 
% [vstar_approx] = ConsensusHopping(E, ~, parametersCBO, V0)
% 
% input:    E             = objective function E (as anonymous function)
%           parametersCH  = suitable parameters for CH
%                         = [K, N, sigma, alpha]
%               - K       = number of time steps
%               - N       = number of particles to sample
%               - sigma   = exploration/sampling width
%               - alpha   = weight/temperature parameter alpha
%           V0            = initial position
%           
% output:   vstar_approx  = approximation to vstar
%

function [vstar_approx] = CH(E, parametersCH, V0, CHvariant)

% get parameters
K = parametersCH('K');
d = size(V0,1);
N = parametersCH('N');
alpha = parametersCH('alpha');
sigma = parametersCH('sigma');

% initialization
V = V0 + sigma*randn(d,N);

% % CH
for k = 1:K
    
    % % CH iteration
    % compute current consensus point v_alpha
    v_alpha = compute_valpha(E, alpha, V);

    % sample points around v_alpha for next iteration
    if strcmp(CHvariant, "ConsensusHopping_scheduledvanishingBM")
        if mod(k,100)==0
            parametersCH('sigma') = parametersCH('sigma') * 0.98;
        end
    end
    V = v_alpha + parametersCH('sigma')*randn(d,N);
    
end

v_alpha = compute_valpha(E, alpha, V);
vstar_approx = v_alpha;

end