% CBO Comparison between J and Var
%
% This script compares the decay behavior of our functional J with the
% variance of the particles for different initializations of CBO.
% Such plot is used in Figure 2(b).
%

%%
clear; clc; close all;

co = set_color();
co = co([1,2,3,5],:);


%% Settings for Easy Handling and Notes
% 
% error metrics to be plotted
Metrics = "all";
% use pre-set CBO setting (overrides manually chosen parameters)
pre_setparameters = 0;

% save plot
pdfexport = 0;


%% Energy Function E

% % dimension of the ambient space
d = 1; % only 1d due to the convex envelope computation

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
% lopsided W-shaped function in 1d
objectivefunction = 'Rastrigin';
[E, parametersE, parametersCBO, ~] = objective_function(objectivefunction,d);

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
% sigma (parameter of exploration term)
sigma = 0.5;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^15;

 
%% Various Different Initializations
% Mean of Initialization mu+1*randn(d,N);
mu = [0.5,1,1.5,2];
var = 0.5;


%% Use pre-set setting
if pre_setparameters==1
    T = parametersCBO('T');
    dt = parametersCBO('dt');
    N = parametersCBO('N');
    lambda = parametersCBO('lambda');
    gamma = parametersCBO('gamma');
    learning_rate = parametersCBO('learning_rate');
    sigma = parametersCBO('sigma');
    alpha = parametersCBO('alpha');
else
    parametersCBO = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma'},...
                                   {  T,   dt,   N,   alpha,   lambda,   gamma,   learning_rate,   sigma});
end


%% Convex Envelope Ec of E

% % computation of the convex hull of the energy function E
% we exploit the relationship between convex functions and their epigraphs
convhull_dom = linspace(xrange(1), xrange(2), 10^6);
Ec = [convhull_dom; E(convhull_dom)]';
[indices,~] = convhull(Ec);

convhull_x = Ec(indices,1); convhull_x(end) = [];
convhull_y = Ec(indices,2); convhull_y(end) = [];


%% Error Metrics

% Variance
Variance = NaN(length(mu),1+T/dt);
% J
J = NaN(length(mu),1+T/dt);

% % Initialization (for convenience, the error metrics are normalized)
J(:,1) = ones(length(mu),1);
Variance(:,1) = ones(length(mu),1);

for m = 1:length(mu)
    
    V0 = mu(m)+var*randn(d,N);
    V = V0;

    normal_J = sum(interp1(convhull_x,convhull_y,V0,'linear')-E(vstar))/N;
    normal_V = sum((V0-sum(V0)/N).^2)/N;

    % CBO Algorithm
    for k = 1:T/dt

        % % CBO iteration
        [V,v_alpha] = CBO_iteration(E,parametersCBO,V);

        % % Computation of Error Metrics
        % Functional J
        J(m,k+1) = sum(interp1(convhull_x,convhull_y,V,'linear')-E(vstar))/N;
        J(m,k+1) = J(m,k+1)/normal_J;
        % Variance
        Expectation = sum(V,2)/N;
        Variance(m,k+1) = sum((V-Expectation).^2)/N;
        Variance(m,k+1) = Variance(m,k+1)/normal_V;

    end

end

%% Plotting of Error Metrics

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

f = figure('Position', [1700 800 600 400]);
for m = 1:length(mu)
    label_Var = ['$\mathrm{Var}(\widehat\rho^N_t)/\mathrm{Var}(\rho_0), \rho_0=\mathcal{N}($',num2str(mu(m)),'$,$',num2str(var),'$)$'];
    errormetric_plot_V = plot(0:dt:T,Variance(m,:), "color", co(m,:), 'LineWidth', 2, 'LineStyle', '--','DisplayName',label_Var);
    hold on
end
for m = 1:length(mu)
    label_J = ['$\mathcal{J}(\widehat\rho^N_t)/\mathcal{J}(\rho_0), \rho_0=\mathcal{N}($',num2str(mu(m)),'$,$',num2str(var),'$)$'];
    errormetric_plot_J = plot(0:dt:T,J(m,:), "color", co(m,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_J);
    hold on;
end
xlim([0,T])
ylim([0,1.5])

ax = gca;
ax.FontSize = 13;

xlabel('$t$','Interpreter','latex','FontSize',15)
legend('Interpreter','latex','FontSize',15)


%% Save Image
if pdfexport
    print(f,['images_videos/JandVarforVariousV0_',objectivefunction],'-dpdf');

    % save parameters
    save(['images_videos/JandVarforVariousV0_',objectivefunction,'_param'], 'objectivefunction', 'E', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'mu')
end

