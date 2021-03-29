% CBO Comparison between J and Var
%
% This script illustrates the optimization procedure of CBO for 1d
% objective functions while comparing the decay behavior of our functional 
% J with the variance of the particles.
%

%%
clear; clc; close all;

co = set_color();


%% Settings for Easy Handling and Notes
% 
% decice if time steps require pressing some arbitrary key
manual_steps = 0;
% error metrics to be plotted
Metrics = "all";
% plotting error metric during or after iterations
errorplots_meanwhile = 1;
% plotting empirical expectation of the particles
show_expectation = 0;
% use pre-set CBO setting (overrides manually chosen parameters)
pre_setparameters = 0;

% save video
savevideo = 0;


%% Energy Function E

% % dimension of the ambient space
d = 1; % only 1d due to the convex envelope computation and plotting

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
% lopsided W-shaped function in 1d
objectivefunction = 'Rastrigin';
[E, parametersE, parametersCBO, parametersInitialization] = objective_function(objectivefunction,d);

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
N = 32000;

% lambda (parameter of consensus drift term)
lambda = 1;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 0.5;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^15;
 
 
%% Initialization
V0mean = 4;
V0std = 1;


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
    V0mean = parametersInitialization('V0mean');
    V0std = parametersInitialization('V0std');
else
    parametersCBO = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma'},...
                                   {  T,   dt,   N,   alpha,   lambda,  gamma,    learning_rate,   sigma});
    parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                              {  V0mean,   V0std});
end


%% Convex Envelope Ec of E

% % computation of the convex hull of the energy function E
% we exploit the relationship between convex functions and their epigraphs
convhull_dom = linspace(xrange(1), xrange(2), 10^6);
Ec = [convhull_dom; E(convhull_dom)]';
[indices,~] = convhull(Ec);

convhull_x = Ec(indices,1); convhull_x(end) = [];
convhull_y = Ec(indices,2); convhull_y(end) = [];


%% CBO Algorithm - Part 1
%initialization
V0 = V0mean+V0std*randn(d,N);
V = V0;


%% Error Metrics

% Variance
Variance = NaN(1,1+T/dt);
% J
J = NaN(1,1+T/dt);

% % Initialization (for convenience, the error metrics are normalized)
J(1) = 1;
Variance(1) = 1;
normal_J = sum(interp1(convhull_x,convhull_y,V0,'linear')-E(vstar))/N;
normal_V = sum((V0-sum(V0)/N).^2)/N;


%% Plotting

% % plot setting
if errorplots_meanwhile
    figure('Position', [1200 800 1100 400])
    subplot(1,2,1)
else
    figure('Position', [1200 800 500 400])
end

% % plotting convex envelope of E
Ecplot = plot(convhull_x, convhull_y, "color", co(6,:), 'LineWidth', 2, 'LineStyle', '--');
hold on
% % plotting energy function E
Eplot = fplot(E, xrange_plot, "color", co(1,:), 'LineWidth', 2);
xlim(xrange_plot)
ylim(yrange_plot)
hold on
% % plot global minimizer of energy function E
vstarplot = plot(vstar, E(vstar), '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:));
hold on

title('Setting','Interpreter','latex','FontSize',16)
legend([Eplot, Ecplot, vstarplot], 'Objective function $\mathcal{E}$','Convex envelope $\mathcal{E}^c$','Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',12)
ax = gca;
ax.FontSize = 13;
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
title(sprintf("CBO at time t=0"),'Interpreter','latex','FontSize',16)
V_plot = scatter(V0, E(V0), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
hold on
legend([Eplot, Ecplot, vstarplot, V_plot], 'Objective function $\mathcal{E}$','Convex envelope $\mathcal{E}^c$','Global minimizer $v^*$','Particles $V_0^i$','Location','northwest','Interpreter','latex','FontSize',12)

% plotting of error metrics (meanwhile)
if errorplots_meanwhile
    subplot(1,2,2)
    [errormetric_plot_J, errormetric_plot_V] = plot_errormetric(Metrics,T,dt,J,Variance);
    
    ax = gca;
    ax.FontSize = 13;
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
    title(sprintf("CBO at time t=%d",t),'Interpreter','latex','FontSize',16)
    
    
    % % CBO iteration
    [V,v_alpha] = CBO_iteration(E,parametersCBO,V);
    
    
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
        legend([Eplot, Ecplot, vstarplot, V_plot, Expectation_plot, valpha_plot], 'Objective function $\mathcal{E}$','Convex envelope $\mathcal{E}^c$','Global minimizer $v^*$','Particles $V_t^i$', 'Average particle $\textbf{E}\overline{V}_t$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',12)
    else
        legend([Eplot, Ecplot, vstarplot, V_plot, valpha_plot], 'Objective function $\mathcal{E}$','Convex envelope $\mathcal{E}^c$','Global minimizer $v^*$','Particles $V_t^i$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',12)
    end
    
    
    % % Computation of Error Metrics
    % Functional J
    J(k+1) = sum(interp1(convhull_x,convhull_y,V,'linear')-E(vstar))/N;
    J(k+1) = J(k+1)/normal_J;
    % Variance
    Variance(k+1) = sum((V-Expectation).^2)/N;
    Variance(k+1) = Variance(k+1)/normal_V;
    
    
    % plotting of error metrics (meanwhile)
    if errorplots_meanwhile
        delete(errormetric_plot_J)
        delete(errormetric_plot_V)
        subplot(1,2,2)
        [errormetric_plot_J, errormetric_plot_V] = plot_errormetric(Metrics,T,dt,J,Variance);
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
    plot_errormetric(Metrics,T,dt,J,Variance);
    
    ax = gca;
    ax.FontSize = 13;
end

%% Save Video
if savevideo
    video = VideoWriter(['images_videos/JandVar_',objectivefunction],'MPEG-4');
    video.FrameRate = 8;
    open(video);
    writeVideo(video,frame(1));
    writeVideo(video,frame(2));
    for k = 1:T/dt
        writeVideo(video,frame(k+2));
    end
    close(video);
    % save parameters
    save(['images_videos/JandVar_',objectivefunction,'_param'], 'objectivefunction', 'E', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std')
end


%% error metric plotting routine
function [errormetric_plot_J, errormetric_plot_V] = plot_errormetric(Metrics,T,dt,J,Variance)
co = set_color();
if or(Metrics=="J",Metrics=="all")
    errormetric_plot_J = plot(0:dt:T,J, "color", co(1,:), 'LineWidth', 2, 'LineStyle', '-');
    hold on;
end
if Metrics=="all"
    hold on;
end
if or(Metrics=="V",Metrics=="all")
    errormetric_plot_V = plot(0:dt:T,Variance, "color", co(2,:), 'LineWidth', 2, 'LineStyle', '--');
end
xlim([0,T])
ylim([0,1.5])
xlabel('$t$','Interpreter','latex','FontSize',14)
title('Decay behavior of $\mathcal{J}(\widehat\rho^N_t)$ and $\mathrm{Var}(\widehat\rho^N_t)$','Interpreter','latex','FontSize',16)

if Metrics=="J"
    legend('$\mathcal{J}(\widehat\rho^N_t)=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})d\rho^N_t(v)$','Interpreter','latex','FontSize',12)
elseif Metrics=="V"
    legend('$\mathrm{Var}(\widehat\rho^N_t)$','Interpreter','latex','FontSize',12)
else
    legend('$\mathcal{J}(\widehat\rho^N_t)=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})d\widehat\rho^N_t(v)$','$\mathrm{Var}(\widehat\rho^N_t)$','Interpreter','latex','FontSize',12)
end
end



