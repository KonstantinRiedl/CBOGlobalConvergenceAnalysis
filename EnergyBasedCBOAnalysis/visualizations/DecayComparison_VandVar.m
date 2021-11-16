% CBO comparison between our functional V and Var
%
% This script illustrates the optimization procedure of CBO for 1d
% objective functions while comparing the decay behavior of our functional 
% V with the variance of the particles.
%

%%
clear; clc; close all;

co = set_color();


%% Settings for Easy Handling and Notes
% 
% decice if time steps require pressing some arbitrary key
manual_steps = 0;
% plotting error metric during or after iterations
errorplots_meanwhile = 1;
% plotting empirical expectation of the particles
show_expectation = 0;
% use pre-set CBO setting (overrides manually chosen parameters)
pre_setparameters = 0;

% save video
savevideo = 0;

% plot settings
semilogy_plot = 0; % show decays in semilogy plot
normalized = 1; % normalize energy functional V



%% Energy Function E

% % dimension of the ambient space
d = 1;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, parametersE, parametersCBO, parametersInitialization] = objective_function(objectivefunction, d, 'CBO');

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
xrange = 100*xrange_plot;

% global minimizer
vstar = 0; %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of CBO Algorithm

% time horizon
T = 2.5;

% discrete time size
dt = 0.1;
 
% number of particles
N = 320;

% lambda (parameter of consensus drift term)
lambda = 1;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% type of diffusion
anisotropic = 1;
% sigma (parameter of exploration term)
sigma = 0.5;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^15;
 
 
%% Initialization
V0mean = 4;
V0std = sqrt(0.8);


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
    V0mean = parametersInitialization('V0mean');
    V0std = parametersInitialization('V0std');
else
    parametersCBO = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma'},...
                                   {  T,   dt,   N,   alpha,   lambda,  gamma,    learning_rate,   anisotropic, sigma});
    parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                              {  V0mean,   V0std});
end


%% CBO Algorithm - Part 1
%initialization
V0 = V0mean+V0std*randn(d,N);
V = V0;


%% Error Metrics

% Variance
Variance = NaN(1,1+T/dt);
% Functional V (called Vstar)
Vstar = NaN(1,1+T/dt);

% % Initialization (for convenience, the error metrics are normalized)
Vstar(1) = 1/2*sum((V0-vstar/N).^2)/N;
Variance(1) = 1/2*sum((V0-sum(V0,2)/N).^2)/N;
if normalized
	normal_Vstar = Vstar(1);
	normal_V = Variance(1);
    Vstar(1) = Vstar(1)/normal_Vstar;
    Variance(1) = Variance(1)/normal_V;
end


%% Plotting

% % plot setting
if errorplots_meanwhile
    figure('Position', [1200 800 1100 400])
    subplot(1,2,1)
else
    figure('Position', [1200 800 500 400])
end

% % plotting energy function E
Eplot = fplot(E, xrange_plot, "color", co(1,:), 'LineWidth', 2);
xlim(xrange_plot)
ylim(yrange_plot)
hold on
% % plot global minimizer of energy function E
vstarplot = plot(vstar, E(vstar), '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:));
hold on

legend([Eplot, vstarplot], 'Objective function $\mathcal{E}$','Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',15)

ax = gca;
ax.FontSize = 13;
title('Setting','Interpreter','latex','FontSize',18)
pause(dt)
if manual_steps
    pause()
end

if savevideo
    frame(1) = getframe(gcf);
end

ax = gca;
ax.FontSize = 13;


%% CBO Algorithm - Part 2

% plot initial setting
if errorplots_meanwhile
    subplot(1,2,1)
end
fprintf("t=0\n")
title(sprintf("CBO at time t=0"),'Interpreter','latex','FontSize',18)
V_plot = scatter(V0, E(V0), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
hold on
legend([Eplot, vstarplot, V_plot], 'Objective function $\mathcal{E}$','Global minimizer $v^*$','Particles $V_0^i$','Location','northwest','Interpreter','latex','FontSize',15)

% plotting of error metrics (meanwhile)
if errorplots_meanwhile
    subplot(1,2,2)
    [errormetric_plot_Vstar, errormetric_plot_V] = plot_errormetric(T,dt,Vstar,Variance, semilogy_plot,normalized,lambda,d,anisotropic,sigma);
    ax = gca;
    ax.FontSize = 13;
    
    title('Decay behavior of $\mathrm{Var}(\widehat\rho^N_t)$ and $\mathcal{V}(\widehat\rho^N_t)$','Interpreter','latex','FontSize',18)

end

if savevideo
    frame(2) = getframe(gcf);
end

% CBO
for k = 1:T/dt
    
    pause(dt)
    if manual_steps
        pause()
    end

    t = k*dt;
    fprintf("t=%d\n", t)
    if errorplots_meanwhile
        subplot(1,2,1)
    end
    title(sprintf("CBO at time t=%d",t),'Interpreter','latex','FontSize',18)
    
    
    % % CBO iteration
    % compute current consensus point v_alpha
    v_alpha = compute_valpha(E, alpha, V);

    % position updates of one iteration of CBO
    V = CBO_update(E, parametersCBO, v_alpha, V);
    
    
    % % Visualization of the way CBO optimizes non-convex functions
    % remove all old plotting objects
    delete(V_plot)
    if k~=1
        delete(valpha_plot)
        if show_expectation
            delete(Expectation_plot)
        end
    end
    
    % plotting of particles
    if errorplots_meanwhile
        subplot(1,2,1)
    end
    V_plot = scatter(V, E(V), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
    % plotting of expectation
    Expectation = sum(V)/N;
    if show_expectation
        Expectation_plot = plot(Expectation, E(Expectation), '.', 'MarkerSize', 20, 'LineWidth', 1.8, "color", 0.9*co(3,:));
    end
    % plotting of consensus point
    valpha_plot = plot(v_alpha, E(v_alpha), '.', 'MarkerSize', 20, 'LineWidth', 1.8, "color", co(2,:));
    
    if show_expectation
        legend([Eplot, vstarplot, V_plot, Expectation_plot, valpha_plot], 'Objective function $\mathcal{E}$','Global minimizer $v^*$','Particles $V_t^i$', 'Average particle $\textbf{E}\overline{V}_t$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',15)
    else
        legend([Eplot, vstarplot, V_plot, valpha_plot], 'Objective function $\mathcal{E}$','Global minimizer $v^*$','Particles $V_t^i$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',15)
    end
    
    
    % % Computation of Error Metrics
    % Energy Functional V
    Vstar(k+1) = 1/2*sum((V-vstar).^2)/N;
    % Variance
    Variance(k+1) = 1/2*sum((V-Expectation).^2)/N;
    
    if normalized
        Vstar(k+1) = Vstar(k+1)/normal_Vstar;
        Variance(k+1) = Variance(k+1)/normal_V;
    end
    
    
    % plotting of error metrics (meanwhile)
    if errorplots_meanwhile
        delete(errormetric_plot_Vstar)
        delete(errormetric_plot_V)
        subplot(1,2,2)
        [errormetric_plot_Vstar, errormetric_plot_V] = plot_errormetric(T,dt,Vstar,Variance, semilogy_plot,normalized,lambda,d,anisotropic,sigma);
    end
    
    if savevideo
        frame(k+2) = getframe(gcf);
    end
    
end
v_alpha = compute_valpha(E,alpha,V);
fprintf("global minimizer (numerically): %d\n", vstar)
fprintf("final consensus point         : %d\n", v_alpha)


% plotting of error metrics (afterwards)
if ~errorplots_meanwhile
    figure('Position', [1700 800 500 400])
    plot_errormetric(T,dt,Vstar,Variance, semilogy_plot,normalized,lambda,d,anisotropic,sigma);
    
    ax = gca;
    ax.FontSize = 13;
end

%% Save Video
if savevideo
    if anisotropic
        video = VideoWriter(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/VandVar_',objectivefunction,'_anisotropic'],'MPEG-4');
    else
        video = VideoWriter(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/VandVar_',objectivefunction,'_isotropic'],'MPEG-4');
    end
    video.FrameRate = 8;
    open(video);
    writeVideo(video,frame(1));
    writeVideo(video,frame(2));
    for k = 1:T/dt
        writeVideo(video,frame(k+2));
    end
    close(video);
    % save parameters
    if anisotropic
        save(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/VandVar_',objectivefunction,'_anisotropic_param'], 'objectivefunction', 'E', 'anisotropic', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std')
    else
        save(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/VandVar_',objectivefunction,'_isotropic_param'], 'objectivefunction', 'E', 'anisotropic', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std')
    end
end


%% error metric plotting routine
function [errormetric_plot_Vstar, errormetric_plot_V] = plot_errormetric(T,dt,Vstar,Variance, semilogy_plot,normalized,lambda,d,anisotropic,sigma)
co = set_color();

if ~semilogy_plot
    errormetric_plot_V = plot(0:dt:T,Variance, "color", co(2,:), 'LineWidth', 2, 'LineStyle', '--');
    hold on;
    errormetric_plot_Vstar = plot(0:dt:T,Vstar, "color", co(1,:), 'LineWidth', 2, 'LineStyle', '-');
else
    errormetric_plot_V = semilogy(0:dt:T,Variance, "color", co(2,:), 'LineWidth', 2, 'LineStyle', '--');
    hold on;
    errormetric_plot_Vstar = semilogy(0:dt:T,Vstar, "color", co(1,:), 'LineWidth', 2, 'LineStyle', '-');
end
if ~semilogy_plot
    xlim([0,T])
    ylim([0,1.5])
else
    xlim([0,T])
    ylim([5*10^-3,2])
end


% rate of decay reference line (from theory)
if ~anisotropic
    if ~semilogy_plot
        decayrate = plot(0:dt:T,exp(-2*(lambda-d*sigma^2/2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':');
    else
        decayrate = semilogy(0:dt:T,exp(-2*(lambda-d*sigma^2/2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':');
    end
else
    if ~semilogy_plot
        decayrate = plot(0:dt:T,exp(-2*(lambda-sigma^2/2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':');
    else
        decayrate = semilogy(0:dt:T,exp(-2*(lambda-sigma^2/2)*[0:dt:T]), "color", 0.4*[1,1,1], 'LineWidth', 2, 'LineStyle', ':');
    end

end

if ~anisotropic
    if ~normalized
        legend([errormetric_plot_V, errormetric_plot_Vstar,decayrate], '$\mathrm{Var}(\widehat\rho^N_t)$','$\mathcal{V}(\widehat\rho^N_t)$','$\exp\!\big(\!-2(\lambda-d\sigma^2/2)\big)$','Interpreter','latex','FontSize',15)
    else
        legend([errormetric_plot_V, errormetric_plot_Vstar,decayrate], '$\mathrm{Var}(\widehat\rho^N_t)/\mathrm{Var}(\rho_0)$','$\mathcal{V}(\widehat\rho^N_t)/\mathcal{V}(\rho_0)$','$\exp\!\big(\!-(2\lambda-d\sigma^2)t\big)$','Interpreter','latex','FontSize',15)
    end
else
    if ~normalized
        legend([errormetric_plot_V, errormetric_plot_Vstar,decayrate], '$\mathrm{Var}(\widehat\rho^N_t)$','$\mathcal{V}(\widehat\rho^N_t)$','$\exp\!\big(\!-2(\lambda-\sigma^2/2)\big)$','Interpreter','latex','FontSize',15)
    else
        legend([errormetric_plot_V, errormetric_plot_Vstar,decayrate], '$\mathrm{Var}(\widehat\rho^N_t)/\mathrm{Var}(\rho_0)$','$\mathcal{V}(\widehat\rho^N_t)/\mathcal{V}(\rho_0)$','$\exp\!\big(\!-(2\lambda-\sigma^2)t\big)$','Interpreter','latex','FontSize',15)
    end
end
xlabel('$t$','Interpreter','latex','FontSize',14)

end



