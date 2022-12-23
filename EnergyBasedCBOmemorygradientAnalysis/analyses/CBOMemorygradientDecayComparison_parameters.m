% Comparison between different parameter settings of CBOMemoryGradient
%
% This script compares the decay behavior of our functional V for different
% parameters of the dynamics with the theoretically expected rates.
%

%%
clear; clc; close all;

co = set_color();
co = co([1,2,3,4,5,6,7],:);


%% Settings for Easy Handling and Notes
% 
% use pre-set CBOMemoryGradient setting (overrides manually chosen parameters)
pre_setparameters = 0;

% save plot
pdfexport = 0;

% 
H_type = 'yplusstar'; % star or yplusstar or y

% plot settings
semilogy_plot = 1; % show decays in semilogy plot
normalized = 1; % normalize energy functional V

% % parameter for comparison (with values)
% (this overwrites the one from below)
parameter_of_interest = 'lambda2';
parameter_values_of_interest = [0,0.1,0.2,0.3,0.4];
% 'kappa';        [];
% 'lambda1';      [];
% 'lambda2';      [0,0.1,0.2,0.3,0.4];
% 'sigma1';       [0.02,0.08,0.2,0.4];
% 'sigma2';       [];
% 'alpha';        [1,2,4,10,100];
% 'beta';         [1,2,4,10,100];


%% Energy Function E

% % dimension of the ambient space
d = 1;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'Rastrigin';
[E, grad_E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'CBO');

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
xrange = 100*xrange_plot;


% global minimizer
xstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of CBOMemoryGradient Algorithm

% time horizon
T = 4;

% discrete time size
dt = 0.02;

% number of particles
N = 320000;

% memory
memory = 1; % 0 or 1
% lambda2, sigma2, kappa and beta have no effect for memory=0.

%
kappa = 1/dt;

% lambda1 (drift towards global and in-time best (consensus) parameter)
lambda1 = 1;
% lambda2 (drift towards in-time best parameter)
lambda2 = 0;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% exploration/noise 1 type
anisotropic1 = 1;
% sigma (exploration/noise parameter 1)
sigma1 = sqrt(0.4);
% exploration/noise 2 type
anisotropic2 = 1;
% sigma (exploration/noise parameter 2)
sigma2 = lambda2*sigma1;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 1000;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta

 
%% Initialization

X0mean = 4*ones(d,1);
X0std = 4;
X0 = X0mean+X0std*randn(d,N);
Y0 = X0;


%% Error Metrics

% % Functional 
Hstarfunctional = NaN(length(parameter_values_of_interest),1+T/dt);
Hstaryfunctional = NaN(length(parameter_values_of_interest),1+T/dt);
Hyfunctional = NaN(length(parameter_values_of_interest),1+T/dt);
Hyplusstarfunctional = NaN(length(parameter_values_of_interest),1+T/dt);

for i = 1:length(parameter_values_of_interest)
    
    parametersCBOmemory = containers.Map({'T', 'dt', 'N', 'memory', 'kappa', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'gamma', 'learning_rate', 'alpha', 'beta'},...
                                         {  T,   dt,   N,   memory,   kappa,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   gamma,   learning_rate,   alpha,  beta});
    
    % setting parameter of interest
    parametersCBOmemory(parameter_of_interest) = parameter_values_of_interest(i);
    if strcmp(parameter_of_interest, 'lambda1')
        parametersCBOmemory('sigma1') = parametersCBOmemory('lambda1')*parametersCBOmemory('sigma2');
    end
    
    X = X0;
    Y = Y0;
    
    % normalization of error metrics
    if normalized
        normal_Hstarfunctional = sum(vecnorm(X-xstar,2,1).^2)/N;
        normal_Hstaryfunctional = sum(vecnorm(Y-xstar,2,1).^2)/N;
        normal_Hyplusstarfunctional = sum(vecnorm(X-Y,2,1).^2)/N + sum(vecnorm(X-xstar,2,1).^2)/N;
    end
    
    % % Initialization of error metrics
    Hstarfunctional(i,1) = normal_Hstarfunctional;
    Hstaryfunctional(i,1) = normal_Hstaryfunctional;
    Hyplusstarfunctional(i,1) = normal_Hyplusstarfunctional;
    Hyfunctional(i,1) = 0;
    if normalized
        Hstarfunctional(i,1) = Hstarfunctional(i,1)/normal_Hstarfunctional;
        Hstaryfunctional(i,1) = Hstaryfunctional(i,1)/normal_Hstaryfunctional;
        Hyplusstarfunctional(i,1) = Hyplusstarfunctional(i,1)/normal_Hyplusstarfunctional;
    end
    

    % CBOMemoryGradient Algorithm 
    for k = 1:T/dt
        
        % % CBOMemoryGradient iteration
        % compute global and in-time best position y_alpha
        y_alpha = compute_yalpha(E, alpha, Y);

        % position updates of one iteration of CBOMemoryGradient
        [X, Y] = CBOmemorygradient_update(E, grad_E, parametersCBOmemory, y_alpha, X, Y);
        
        % % Computation of Error Metrics
        % Energy Functional Hstar
        Hstarfunctional(i,k+1) = sum(vecnorm(X-xstar,2,1).^2)/N;%%%%%%%%%%%%% + sum(vecnorm(X-Y,2,1).^2)/N;
        Hstaryfunctional(i,k+1) = sum(vecnorm(Y-xstar,2,1).^2)/N;
        % Energy Functional Hyplusstar
        Hyplusstarfunctional(i,k+1) = sum(vecnorm(X-Y,2,1).^2)/N + sum(vecnorm(X-xstar,2,1).^2)/N;
        % Energy Functional Hy
        Hyfunctional(i,k+1) = sum(vecnorm(X-Y,2,1).^2)/N;
        
        if normalized
            Hstarfunctional(i,k+1) = Hstarfunctional(i,k+1)/normal_Hstarfunctional;
            Hstaryfunctional(i,k+1) = Hstaryfunctional(i,k+1)/normal_Hstaryfunctional;
            Hyplusstarfunctional(i,k+1) = Hyplusstarfunctional(i,k+1)/normal_Hyplusstarfunctional;
        end

    end

end

%% Plotting of Error Metrics

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

if strcmp(parameter_of_interest,'d')
	d = parameter_values_of_interest;
elseif strcmp(parameter_of_interest,'kappa')
    parameter_of_interest_string = '$\kappa=\,$';
    kappa = parameter_values_of_interest;
elseif strcmp(parameter_of_interest,'lambda1')
    parameter_of_interest_string = '$\lambda_1=\,$';
    lambda1 = parameter_values_of_interest;
elseif strcmp(parameter_of_interest,'lambda2')
    parameter_of_interest_string = '$\lambda_2=\,$';
    lambda2 = parameter_values_of_interest;
elseif strcmp(parameter_of_interest,'sigma1')
    parameter_of_interest_string = '$\sigma_1=\,$';
    sigma1 = parameter_values_of_interest;
elseif strcmp(parameter_of_interest,'sigma2')
    parameter_of_interest_string = '$\sigma_2=\,$';
    sigma1 = parameter_values_of_interest;
elseif strcmp(parameter_of_interest,'alpha')
    parameter_of_interest_string = '$\alpha=\,$';
    alpha = parameter_values_of_interest;
elseif strcmp(parameter_of_interest,'beta')
    parameter_of_interest_string = '$\beta=\,$';
    beta = parameter_values_of_interest;
else
    error('parameter_of_interest not known')
end
    
f = figure('Position', [1700 800 600 400]);


for i = 1:length(parameter_values_of_interest)
    if ~normalized
        label_Hfunctional = ['$\mathcal{H}(\widehat\rho^N_t)$, ', parameter_of_interest_string,num2str(parameter_values_of_interest(i))];
    else
        label_Hfunctional = ['$\mathcal{H}(\widehat\rho^N_t)/\mathcal{H}(\rho_0)$, ', parameter_of_interest_string ,num2str(parameter_values_of_interest(i))];
    end
    
    if strcmp(H_type, 'star')
        if ~semilogy_plot
            errormetric_plot = plot(0:dt:T,Hstarfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
            hold on
            errorymetric_plot = plot(0:dt:T,Hstaryfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-.');
        else
            errormetric_plot = semilogy(0:dt:T,Hstarfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
            hold on
            errorymetric_plot = semilogy(0:dt:T,Hstaryfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-.');
        end
    elseif strcmp(H_type, 'y')
        if ~semilogy_plot
            errormetric_plot = plot(0:dt:T,Hyfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
        else
            errormetric_plot = semilogy(0:dt:T,Hyfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
        end
    elseif strcmp(H_type, 'yplusstar')
        if ~semilogy_plot
            errormetric_plot = plot(0:dt:T,Hyplusstarfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
        else
            errormetric_plot = semilogy(0:dt:T,Hyplusstarfunctional(i,:), "color", co(i,:), 'LineWidth', 2, 'LineStyle', '-','DisplayName',label_Hfunctional);
        end
    else
        error('H_type error. H_type not know.')
    end
    hold on
end


xlim([0,T])
xticks([0:0.5:T])
if ~semilogy_plot
    % normal plot
    ylim([0,1])
    yticks([0 0.25 0.5 0.75 1])
else
    % semilogy plot
    %ylim([5*10^-3,1])
end

% rate of decay reference line (from theory)
for i = 1:length(parameter_values_of_interest)
    
    if strcmp(parameter_of_interest,'d')
        d = parameter_values_of_interest(i);
    elseif strcmp(parameter_of_interest,'kappa')
        kappa = parameter_values_of_interest(i);
    elseif strcmp(parameter_of_interest,'lambda1')
        lambda1 = parameter_values_of_interest(i);
    elseif strcmp(parameter_of_interest,'lambda2')
        lambda2 = parameter_values_of_interest(i);
    elseif strcmp(parameter_of_interest,'sigma1')
        sigma1 = parameter_values_of_interest(i);
    elseif strcmp(parameter_of_interest,'sigma2')
        sigma2 = parameter_values_of_interest(i);
    elseif strcmp(parameter_of_interest,'alpha')
        alpha = parameter_values_of_interest(i);
    elseif strcmp(parameter_of_interest,'beta')
        beta = parameter_values_of_interest(i);
    else
        error('parameter_of_interest not known')
    end
    
    
    %label_rate = ['$\exp\!\big(\!-(2\lambda_1-\sigma_1^2)t\big)$, ', parameter_of_interest_string, num2str(parameter_values_of_interest(i))];
    label_rate = ['$\exp\!\big(\!-(2\lambda_1-\sigma_1^2)t\big)$'];
    if ~semilogy_plot
        rate_plot = plot(0:dt:T,exp(-(2*lambda1-sigma1^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName',label_rate);
    else
        rate_plot = semilogy(0:dt:T,exp(-(2*lambda1-sigma1^2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':','DisplayName',label_rate);
    end
    if i>1
        rate_plot.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    hold on
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
    print(f,['images_videos/VforAn_isotropicforVariousdim_',objectivefunction],'-dpdf');

    % save parameters
    save(['images_videos/VforAn_isotropicforVariousdim_',objectivefunction,'_param'], 'objectivefunction', 'E', 'xstar', 'd', 'T', 'dt', 'N', 'memory', 'alpha', 'beta', 'kappa', 'lambda1', 'lambda2', 'gamma', 'learning_rate', 'anisotropic1', 'anisotropic2', 'sigma1', 'sigma2', 'X0mean', 'X0std')
end

