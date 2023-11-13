% Comparison between CBO with and without truncated diffusion
%
% This script compares the decay behavior of CBO with and without truncated
% diffusion.
%

%%
clear; clc; close all;

co = set_color();
co = co([1,2,3,4,5,6,7,8,9],:);


%% Settings for Easy Handling and Notes

% plot settings
semilogy_plot = 0; % show decays in semilogy plot
normalized = 1; % normalize energy functional V


%% Energy Function E

% dimension
d = 4;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'Rastrigin';
[E, grad_E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'CBO');

% % truncation parameter of CBO
M = [1,4,16,64,Inf];


%% Parameters of CBO Algorithm

% time horizon
T = 4;

% discrete time size
dt = 0.01;
 
% number of particles
N = 100;

% lambda (parameter of consensus drift term)
lambda = 1;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% type of diffusion
anisotropic = 0;
% sigma (parameter of exploration term)
sigma = 0.2;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^1;

 
%% Initialization
V0mean = 8*ones(d,1);
V0std = 32;


%% Error Metrics

% % Functional V (called Vstar)
% isotropic case
Vstar = NaN(length(d),1+T/dt);

for i = 1:length(M)
    
    % global minimizer
    vstar = zeros(d,1);
    
    parametersCBOtruncateddiffusion = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma', 'M'},...
                                                     {  T,   dt,   N,   alpha,   lambda,   gamma,   learning_rate,   anisotropic,   sigma,   M(i)});
    
    
    % % Initialization
    V0 = V0mean+sqrt(V0std)*randn(d,N);
    
    % normalization of error metrics
    if normalized
        normal_V = 1/2*sum(vecnorm(V0-vstar).^2)/N;
    end
    
    % % Initialization of error metrics
    Vstar(i,1) = normal_V;
    if normalized
        Vstar(i,1) = Vstar(i,1)/normal_V;
    end
    
    V = V0;
    for k = 1:T/dt
        
        % % CBO iteration
        % compute current consensus point v_alpha
        v_alpha = compute_valpha(E, alpha, V);

        % position updates of one iteration of CBO
        V = CBOtruncateddiffusion_update(E, grad_E, parametersCBOtruncateddiffusion, v_alpha, V);

        % % Computation of Error Metrics
        % Energy Functional V
        Vstar(i,k+1) = 1/2*sum(vecnorm(V-vstar,2,1).^2)/N;
        
        if normalized
            Vstar(i,k+1) = Vstar(i,k+1)/normal_V;
        end

    end

end

%% Plotting of Error Metrics

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

f = figure('Position', [1700 800 600 400]);
for i = 1:length(M)
    if ~normalized
        label_V = ['$\mathcal{V}(\widehat\rho^N_t)$, $ M=\,$',num2str(M(i))];
    else
        label_V = ['$\mathcal{V}(\widehat\rho^N_t)/\mathcal{V}(\rho_0)$, $ M=\,$',num2str(M(i))];
    end
    if ~semilogy_plot
        errormetric_plot = plot(0:dt:T,Vstar(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_V);
    else
        errormetric_plot = semilogy(0:dt:T,Vstar(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_V);
    end
    hold on
end

xlim([0,T])
xticks([0 0.5 1 1.5 2 2.5 3 3.5 4])
if ~semilogy_plot
    % normal plot
    ylim([0,1])
    yticks([0 0.25 0.5 0.75 1])
else
    % semilogy plot
    %ylim([5*10^-3,1])
end

% rate of decay reference line (from theory)
if anisotropic
    if ~semilogy_plot
        rate_plot = plot(0:dt:T,exp(-(2*lambda-sigma^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', '-.','DisplayName','$\exp\!\big(\!-(2\lambda-\sigma^2)t\big)$');
    else
        rate_plot = semilogy(0:dt:T,exp(-(2*lambda-sigma^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', '-.','DisplayName','$\exp\!\big(\!-(2\lambda-\sigma^2)t\big)$');
    end
else
    if ~semilogy_plot
        rate_plot = plot(0:dt:T,exp(-(2*lambda-d*sigma^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName','$\exp\!\big(\!-(2\lambda-d\sigma^2)t\big)$');
    else
        rate_plot = semilogy(0:dt:T,exp(-(2*lambda-d*sigma^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName','$\exp\!\big(\!-(2\lambda-d\sigma^2)t\big)$');
    end
end
if ~semilogy_plot
    rate_plot = plot(0:dt:T,exp(-(2*lambda)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', '-','DisplayName','$\exp\!\big(\!-2\lambda t\big)$');
else
    rate_plot = semilogy(0:dt:T,exp(-(2*lambda)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', '-','DisplayName','$\exp\!\big(\!-2\lambda t\big)$');
end
    
ax = gca;
ax.FontSize = 13;

xlabel('$t$','Interpreter','latex','FontSize',15)
if ~semilogy_plot
    legend('Interpreter','latex','FontSize',15,'Location','northeast')
else
    legend('Interpreter','latex','FontSize',15,'Location','southwest')
end


