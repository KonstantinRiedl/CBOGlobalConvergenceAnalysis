% An illustration of the CBO dynamics
%
% This script visualizes the terms steering the dynamics of CBO.
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
% (E is a function mapping columnwise from R^{d\times N} to R)^N
objectivefunction = 'Rastrigin';
[E, ~, ~, ~, ~] = objective_function(objectivefunction, d, 'CBO');


%% Parameters of CBO Algorithm

% discrete time size
dt = 0.5;
 
% number of exemplary particles
N = 2; % 2 or 3
% number of other particles
N_other = 14;

% lambda (parameter of consensus drift term)
lambda = 1;
% type of diffusion
anisotropic = 0;
% sigma (parameter of exploration term)
if anisotropic==0
    sigma = 0.16;
else
    sigma = 0.24;
end

%% Plotting

% % 
% two exemplary V's
if N==2
    V = [-0.8,2.6;-0.2,1];
elseif N==3
    V = [-1,-1.2,2;-0.2,1.8,1.8];
else
    error('Wrong N.')
end
V_other = [[0.21;0.11], randn(2,N_other-1)+[1;0.8]];
load([main_folder(),'/EnergyBasedCBOAnalysis/visualizations/CBODynamicsIllustration_V_other.mat'])

% % 
% v_star
v_star = [0; 0];
p_v_star = plot(v_star(1), v_star(2), '*', 'MarkerSize', 12, 'LineWidth', 1.8, "color", co(5,:)); hold on

% v_alpha
alpha = 7;
v_alpha = compute_valpha(E,alpha,[V,V_other]);


% direction through diffusion term
for n = 1:25
    dB = randn(d,N);
    if anisotropic
        exploration_term = sigma*abs(V-v_alpha*ones(1,N))*sqrt(dt).*dB;
    else
        exploration_term = sigma*vecnorm(V-v_alpha*ones(1,N),2,1)*sqrt(dt).*dB;
    end
    for i=1:N
        quiver(V(1,i),V(2,i),exploration_term(1,i),exploration_term(2,i), 'LineWidth', 1.6, "color", co(5,:)); hold on
    end
    hold on
end

% direction through consensus drift
drift_term = -lambda*(V-v_alpha*ones(1,N))*dt;
for i=1:N
    quiver(V(1,i),V(2,i),drift_term(1,i),drift_term(2,i), 'LineWidth', 1.6, "color", co(1,:)); hold on
end


base_opacity = 0.2;
for n=1:N
    V_plot = scatter(V(1,n), V(2,n), 50, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:)); hold on
    V_plot.MarkerFaceAlpha = base_opacity+(1-base_opacity)./(E(V(:,n))/min(E([V, V_other])));
    V_plot.MarkerEdgeAlpha = 0;
end

for n = 1:N_other
    V_other_plot = scatter(V_other(1,n), V_other(2,n), 50, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:)); hold on
    V_other_plot.MarkerFaceAlpha = base_opacity+(1-base_opacity)./(E(V_other(:,n))/min(E([V, V_other])));
    V_other_plot.MarkerEdgeAlpha = 0;
end

% v_star plot
p_v_alpha = plot(v_alpha(1), v_alpha(2), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));

ylim([-0.8,1.6]);
xlim([-1.2,3.3]);
set(gca,'xtick',[]); set(gca,'ytick',[]);
set(groot,'defaultAxesTickLabelInterpreter','latex'); 

V_plot = scatter(-10, -10, 50, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:)); hold on
%legend([p_v_star, V_plot], 'Global minimizer $v^*$', 'Particles $V_t^i$','Location','southeast','Interpreter','latex','FontSize',12)
legend([p_v_star, V_plot, p_v_alpha], 'Global minimizer $v^*$', 'Particles $V_t^i$', 'Consensus point $v_{\alpha}(\widehat\rho^N_t)$','Location','southeast','Interpreter','latex','FontSize',12)

ax = gca;
ax.FontSize = 15;


%% Save Image
if pdfexport
    if anisotropic
        print(f,[main_folder(),'/EnergyBasedCBOAnalysis/images_videos/CBODynamicsIllustration_anisotropic'],'-dpdf');
    else
        print(f,[main_folder(),'/EnergyBasedCBOAnalysis/images_videos/CBODynamicsIllustration_isotropic'],'-dpdf');
    end
end
