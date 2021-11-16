% CBO comparison between our functional V and Var
%
% This script compares the decay behavior of our functional V with the
% variance of the particles for different initializations of CBO.
% Such plot is used in Figure 2(b) in "Consensus-based optimization methods
% converge globally in mean-field law"
%

%%
clear; clc; close all;

co = set_color();
co = co([1,2,3,5],:);


%% Settings for Easy Handling and Notes
% 
% use pre-set CBO setting (overrides manually chosen parameters)
pre_setparameters = 0;

% save plot
pdfexport = 0;

% plot settings
semilogy_plot = 0; % show decays in semilogy plot
normalized = 1; % normalize energy functional V


%% Energy Function E

% % dimension of the ambient space
d = 1;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, parametersE, parametersCBO, ~] = objective_function(objectivefunction, d, 'CBO');

% range of x
xrange = 100*parametersE(:,1)';

% global minimizer
vstar = 0; %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of CBO Algorithm

% time horizon
T = 2.5;

% discrete time size
dt = 0.01;
 
% number of particles
N = 320000;

% lambda (parameter of consensus drift term)
lambda = 1;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% type of diffusion
anisotropic = 0; % does not have an effect in 1d
% sigma (parameter of exploration term)
sigma = 0.5;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^15;

 
%% Various Different Initializations
% Mean of Initialization mu+sqrt(var)*randn(d,N);
V0mu = [1,2,3,4];
V0var = 0.8;


%% Use pre-set setting
if pre_setparameters==1
    T = parametersCBO('T');
    dt = parametersCBO('dt');
    N = parametersCBO('N');
    lambda = parametersCBO('lambda');
    gamma = parametersCBO('gamma');
    learning_rate = parametersCBO('learning_rate');
    anisotropic = parametersCBO('anisotropic');
    sigma = parametersCBO('sigma');
    alpha = parametersCBO('alpha');
else
    parametersCBO = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma'},...
                                   {  T,   dt,   N,   alpha,   lambda,  gamma,    learning_rate,   anisotropic, sigma});
end


%% Error Metrics

% Variance
Variance = NaN(length(V0mu),1+T/dt);
% Functional V (called Vstar)
constVstar = 1; % scaling constant
Vstar = NaN(length(V0mu),1+T/dt);

for m = 1:length(V0mu)
    
    % % Initialization
    V0 = V0mu(m)+sqrt(V0var)*randn(d,N);
    V = V0;
    
    % normalization of error metrics
    if normalized
        normal_Vstar = 1/2*sum((V0-vstar).^2)/N;
        normal_V = 1/2*sum((V0-sum(V0,2)/N).^2)/N;
    end
    
    % % Initialization of error metrics
    Vstar(m,1) = 1/constVstar*1/2*sum((V0-vstar).^2)/N;
    Expectation = sum(V0,2)/N;
    Variance(m,1) = 1/2*sum((V0-Expectation).^2)/N;
    if normalized
        Vstar(m,1) = Vstar(m,1)/normal_Vstar;
        Variance(m,1) = Variance(m,1)/normal_V;
    end
    

    % CBO Algorithm
    for k = 1:T/dt

        % % CBO iteration
        % compute current consensus point v_alpha
        v_alpha = compute_valpha(E, alpha, V);

        % position updates of one iteration of CBO
        V = CBO_update(E, parametersCBO, v_alpha, V);

        % % Computation of Error Metrics
        % Energy Functional V
        Vstar(m,k+1) = 1/constVstar*1/2*sum((V-vstar).^2)/N;
        % Variance
        Expectation = sum(V,2)/N;
        Variance(m,k+1) = 1/2*sum((V-Expectation).^2)/N;
        
        if normalized
            Vstar(m,k+1) = Vstar(m,k+1)/normal_Vstar;
            Variance(m,k+1) = Variance(m,k+1)/normal_V;
        end

    end

end

%% Plotting of Error Metrics

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

f = figure('Position', [1700 800 600 400]);
for m = 1:length(V0mu)
    if ~normalized
        label_Var = ['$\mathrm{Var}(\widehat\rho^N_t), \rho_0=\mathcal{N}($',num2str(V0mu(m)),'$,$',num2str(V0var),'$)$'];
    else
        label_Var = ['$\mathrm{Var}(\widehat\rho^N_t)/\mathrm{Var}(\rho_0), \rho_0=\mathcal{N}($',num2str(V0mu(m)),'$,$',num2str(V0var),'$)$'];
    end
    if ~semilogy_plot
        errormetric_plot_V = plot(0:dt:T,Variance(m,:), "color", co(m,:), 'LineWidth', 2, 'LineStyle', '--','DisplayName',label_Var);
    else
        errormetric_plot_V = semilogy(0:dt:T,Variance(m,:), "color", co(m,:), 'LineWidth', 2, 'LineStyle', '--','DisplayName',label_Var);
    end
    hold on
end
for m = 1:length(V0mu)
    if ~normalized
        if constVstar==1
            label_Vstar = ['$\mathcal{V}(\widehat\rho^N_t), \rho_0=\mathcal{N}($',num2str(V0mu(m)),'$,$',num2str(V0var),'$)$'];
        else
            label_Vstar = ['$\mathcal{V}(\widehat\rho^N_t)/',num2str(constVstar),', \rho_0=\mathcal{N}($',num2str(V0mu(m)),'$,$',num2str(V0var),'$)$'];
        end
    else
        label_Vstar = ['$\mathcal{V}(\widehat\rho^N_t)/\mathcal{V}(\rho_0), \rho_0=\mathcal{N}($',num2str(V0mu(m)),'$,$',num2str(V0var),'$)$'];
    end
    if ~semilogy_plot
        errormetric_plot_Vstar = plot(0:dt:T,Vstar(m,:), "color", co(m,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Vstar);
    else
        errormetric_plot_Vstar = semilogy(0:dt:T,Vstar(m,:), "color", co(m,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Vstar);
    end
    hold on;
end

xlim([0,T])
xticks([0 0.5 1 1.5 2 2.5 3 3.5 4])
if ~semilogy_plot
    % normal plot
    ylim([0,1.5])
    yticks([0 0.5 1 1.5])
else
    % semilogy plot
    ylim([5*10^-3,2])
end

% rate of decay reference line (from theory)
if ~semilogy_plot
    plot(0:dt:T,exp(-(2*lambda-d*sigma^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName','$\exp\!\big(\!-(2\lambda-d\sigma^2)t\big)$');
else
    semilogy(0:dt:T,exp(-(2*lambda-d*sigma^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName','$\exp\!\big(\!-(2\lambda-d\sigma^2)t\big)$');
end

ax = gca;
ax.FontSize = 13;

xlabel('$t$','Interpreter','latex','FontSize',15)
if ~semilogy_plot
    legend('Interpreter','latex','FontSize',15,'Location','northeast')
else
    legend('Interpreter','latex','FontSize',15,'Location','southwest')

end


%% Save Image
if pdfexport
    
    if anisotropic
        print(f,['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/VandVarforVariousV0_',objectivefunction, '_anisotropic'],'-dpdf');
        % save parameters
        save(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/VandVarforVariousV0_',objectivefunction,'_anisotropic_param'], 'objectivefunction', 'E', 'anisotropic', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mu', 'V0var')
    else
        print(f,['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/VandVarforVariousV0_',objectivefunction, '_isotropic'],'-dpdf');
        % save parameters
        save(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/VandVarforVariousV0_',objectivefunction,'_isotropic_param'], 'objectivefunction', 'E', 'anisotropic', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mu', 'V0var')
    end
    
end

