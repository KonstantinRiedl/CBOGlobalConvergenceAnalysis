% An illustration of the CBOmemorygradient dynamics
%
% This script visualizes the terms steering the dynamics of CBO with memory.
%

%%
clear; clc; close all;

co = set_color();


%% Settings for Easy Handling and Notes
% 
% save plot
pdfexport = 0;

f = figure('Position', [1200 800 600 240]);


%% Energy Function E

% % dimension of the ambient space
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'Rastrigin';
[E, grad_E, ~, ~, ~] = objective_function(objectivefunction, d, 'CBOMemoryGradient');


%% Parameters of CBO Algorithm

% discrete time size
dt = 0.5;
 
% number of exemplary particles
N = 1;
% number of other particles
%N_other = 14;


% lambda1 (drift towards in-time best parameter)
lambda1 = 0.8;
% lambda2 (drift towards global and in-time best parameter)
lambda2 = 1;
% gamma (parameter of gradient drift term)
gamma = 1;
learning_rate = 0.03;

% type of diffusion for noise term 1
anisotropic1 = 1;
% sigma (parameter of noise term 1)
sigma1 = 0.3;
% type of diffusion for noise term 2
anisotropic2 = 1;
% sigma (parameter of noise term 2)
sigma2 = 0.3;
% type of diffusion for noise term 3
anisotropic3 = 1;
% sigma (parameter of noise term 3)
sigma3 = 0.3;

% alpha (weight in Gibbs measure for global and in-time best position computation)
alpha = 10;


%%
% [X,Y] = meshgrid(-2:.002:5,-1:.002:2);
% XY = [X(:)';Y(:)'];
% Z = E(XY);
% Z = reshape(Z,size(X));
% 
% Eplot = surf(X,Y,Z,'FaceAlpha',0.25); % 0.5 und 0.25
% Eplot.EdgeColor = 'None';
% hold on
% 
% contour(X,Y,Z,20);
% hold on
% view(2)


%% Particles and Histories

% % % Xes and X_histories
% exemplary V
X = [1.7    -0.07  -0.32   0.45   0.76  0.35;
     0.7  -0.3   0.45  -0.72   -0.14  0.68];
N = length(X);

X_history0 = [ 4.4,  3.7,  3,   2.4;
             0.4,  0.9,  0.13,  1.3];
X_history1 = [ -2.2,   -1.48,  -0.9,   -0.64;
              -0.2,  0.14, -0.7, 0.02];
X_history2 = [ -2.7,   -1.58,  -0.94,   -0.5;
              1.08,  1.24, 1.18, 0.8];
X_history3 = [ 2.1,   1.8,  1.3,   0.9;
              -2.7, -1.3,  -0.7, -0.5];
X_history4 = [ 4.3,   3.13,  2.15,   1.3;
              -1, -0.28,  -0.76, 0.2];
X_history5 = [ 1.8, 1.8, 1.8, 0.7;
             2, 2, 2, 1.27];
X_history = [X_history0, X_history1, X_history2, X_history3, X_history4, X_history5];


%% Plotting
% % x_star and v_alpha
% x_star
x_star = [0; 0];
p_x_star = plot(x_star(1), x_star(2), '*', 'MarkerSize', 12, 'LineWidth', 1.8, "color", co(5,:)); hold on

% y_alpha
Y = X;
for n=1:N
    hs = (1:length(X_history0))+(n-1)*length(X_history0);
    X_history_n = [X_history(:,hs), X(:,n)];
    [~,ind_local_best] = min(E(X_history_n));
    Y(:,n) = X_history_n(:,ind_local_best);
end
y_alpha = compute_yalpha(E, alpha, Y);


% % contributions for velocity of particle X(:,1)
% contribution through global best diffusion term
for n = 1:8
    dB = randn(d,1);
    if anisotropic1
        exploration_term = sigma2*abs(X(:,1)-y_alpha*ones(1,1))*sqrt(dt).*dB;
    else
        exploration_term = sigma2*vecnorm(X(:,1)-y_alpha*ones(1,1),2,1)*sqrt(dt).*dB;
    end
    quiver(X(1,1),X(2,1),exploration_term(1,1),exploration_term(2,1), 'LineWidth', 1.6, "color", 0.75*co(5,:)); hold on
    hold on
end

% contribution through local best diffusion term
X_history_0 = [X_history0, X(:,1)];
[~,ind_local_best] = min(E(X_history_0));
for n = 1:8
    dB = randn(d,1);
    if anisotropic2
        local_best_exploration_term = sigma1*abs(X(:,1)-X_history_0(:,ind_local_best))*sqrt(dt).*dB;
    else
        local_best_exploration_term = sigma1*vecnorm(X(:,1)-X_history_0(:,ind_local_best),2,1)*sqrt(dt).*dB;
    end
    quiver(X(1,1),X(2,1),local_best_exploration_term(1,1),local_best_exploration_term(2,1), 'LineWidth', 1.6, "color", co(5,:)); hold on
    hold on
end

% contribution through gradient diffusion term
h = 0.001;
gradient_drift = learning_rate*[(E(X(:,1)+h*[1;0])-E(X(:,1)-h*[1;0]))/(2*h); (E(X(:,1)+h*[0;1])-E(X(:,1)-h*[0;1]))/(2*h)];
for n = 1:8
    dB = randn(d,1);
    if anisotropic3
        gradient_exploration_term = sigma3*abs(gradient_drift)*sqrt(dt).*dB;
    else
        gradient_exploration_term = sigma3*vecnorm(gradient_drift,2,1)*sqrt(dt).*dB;
    end
    quiver(X(1,1),X(2,1),gradient_exploration_term(1,1),gradient_exploration_term(2,1), 'LineWidth', 1.6, "color", 0.5*co(5,:)); hold on
    hold on
end

% contribution through global best drift term
drift_term = -lambda2*(X-y_alpha*ones(1,N))*dt;
quiver(X(1,1),X(2,1),drift_term(1,1),drift_term(2,1), 'LineWidth', 1.6, "color", co(1,:)); hold on

% contribution through local best drift term
X_history_0 = [X_history0, X(:,1)];
[~,ind_local_best] = min(E(X_history_0));
local_best_drift_term = -lambda1*(X-X_history_0(:,ind_local_best))*dt;
quiver(X(1,1),X(2,1),local_best_drift_term(1,1),local_best_drift_term(2,1), 'LineWidth', 1.6, "color", co(6,:)); hold on

% contribution through gradient drift term
gradient_drift_term = -gamma*(gradient_drift)*dt;
quiver(X(1,1),X(2,1),gradient_drift_term(1,1),gradient_drift_term(2,1), 'LineWidth', 1.6, "color", co(7,:)); hold on


% % plot Xes and X_histories
base_opacity = 0.2;
for n=1:N
    % particles
    X_plot = scatter(X(1,n), X(2,n), 50, "MarkerFaceColor", co(3,:)); hold on
    X_plot.MarkerFaceAlpha = base_opacity+(1-base_opacity)./(E(X(:,n))/min(E([X,X_history])));
    X_plot.MarkerEdgeAlpha = 0;
    
    % particle trajectories
    hs = (1:length(X_history0))+(n-1)*length(X_history0);
    X_traj_plot = plot([X_history(1,hs),X(1,n)],[X_history(2,hs),X(2,n)],'-','Color',[co(3,:),0.3],'LineWidth',2);
    
    X_history_n = [X_history(:,hs), X(:,n)];
    [~,ind_local_best] = min(E(X_history_n));
    X_plot_hist = scatter(X_history_n(1,ind_local_best), X_history_n(2,ind_local_best), 50, "MarkerFaceColor", 1*[1,1,1], "MarkerEdgeColor", 0.9*co(3,:)); hold on
    X_plot_hist.MarkerEdgeAlpha = base_opacity+(1-base_opacity)./(E(X_history_n(:,ind_local_best))/min(E([X,X_history])));
    X_plot_hist.MarkerFaceAlpha = 0;
    X_plot_hist.LineWidth = 1.8;
end

% y_alpha plot
p_y_alpha = scatter(y_alpha(1), y_alpha(2), 58, "MarkerFaceColor", 1*[1,1,1], "MarkerEdgeColor", co(2,:)); hold on
p_y_alpha.LineWidth = 1.8;
p_y_alpha.MarkerFaceAlpha = 0;

ylim([-0.8,1.6]);
xlim([-1.2,3.3]);
set(gca,'xtick',[]); set(gca,'ytick',[]);
set(groot,'defaultAxesTickLabelInterpreter','latex'); 

X_plot_ = scatter(-10, -10, 50, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:)); hold on
base_opacity_history = 0.2;
X_history_plot = scatter(-10, -10, 50, "MarkerFaceColor", 1*[1,1,1], "MarkerEdgeColor", 0.9*co(3,:)); hold on
X_history_plot.MarkerEdgeAlpha = 1;
X_history_plot.LineWidth = 1.8;
p_y_alpha = plot(-10, -10, 'o', 'MarkerSize', 7.2, 'LineWidth', 1.8, "color", [1,1,1], 'MarkerEdgeColor', co(2,:));
legend([p_x_star, X_plot_, X_history_plot, p_y_alpha], 'Global minimizer $x^*$', 'Particles'' positions $X_t^i$', 'Local (historical) best positions $Y_t^i$', 'Consensus point $y_{\alpha}(\widehat\rho^N_{Y,t})$','Location','southeast','Interpreter','latex','FontSize',12)

ax = gca;
ax.FontSize = 15;


%% Save Image
if pdfexport
    if anisotropic1
        print(f,[main_folder(),'/EnergyBasedCBOmemorygradientAnalysis/images_videos/CBOmemorygradientDynamicsIllustration_anisotropic'],'-dpdf');
    else
        print(f,[main_folder(),'/EnergyBasedCBOmemorygradientAnalysis/images_videos/CBOmemorygradientDynamicsIllustration_isotropic'],'-dpdf');
    end
end
