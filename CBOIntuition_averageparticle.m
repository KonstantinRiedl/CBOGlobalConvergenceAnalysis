% CBO intuition
%
% This script illustrates the intuition behind our novel analysis approach
% for CBO. "CBO always performs a gradient descent of the squared Euclidean
% distance to the global minimizer"
% Such plot is used in Figure 1(b).
%

%%
clear; clc; close all;

co = set_color();

%% Settings for Easy Handling and Notes
% 
% decice if time steps require pressing some arbitrary key
manual_steps = 0;
% plotting empirical expectation of the particles
show_expectation = 0;
% use pre-set CBO setting (overrides manually chosen parameters)
pre_setparameters = 0;

% 3d plot
spatial_plot = 0;

% save plot
pdfexport = 0;


%% Energy Function E

% % dimension of the ambient space
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
% lopsided W-shaped function in 1d
objectivefunction = 'Rastrigin';
[E, parametersE, parametersCBO, parametersInitialization] = objective_function(objectivefunction,d);

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
zrange_plot = parametersE(:,3)';
xrange = 100*xrange_plot;
yrange = 100*yrange_plot;

% global minimizer
vstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of CBO Algorithm
 
% time horizon
T = 10;

% discrete time size
dt = 0.01;
 
% number of particles
N = 3200;

% lambda (parameter of consensus drift term)
lambda = 1;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 0.01;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^15;

%% Initialization
V0mean = [8;8];
V0std = 20;


%% Exemplary Particle
Vex = [[-2;4],[4.5;1.5],[-1.5;-1.5]];
[~, NUM_EX] = size(Vex);
N = N+NUM_EX;


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


%% Data Generation

NUM_RUNS = 100;
Expectation_trajectories = zeros(NUM_RUNS,d,1+T/dt);
Vex_trajectories = zeros(NUM_RUNS,d,NUM_EX,1+T/dt);
for r = 1:NUM_RUNS
    
    [Vex_trajectories(r,:,1:NUM_EX,:), Expectation_trajectories(r,:,:)] = CBO_trajectories(E,parametersCBO,V0mean,V0std,Vex);
    
end


%% Plotting

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% % color setting
opacity_sampletrajectories = 0.4;
grayish_averagetrajectory = 0.8;


% % plot setting
f = figure('Position', [1200 800 600 500]);
%title('Consensus Based Optimization','Interpreter','latex','FontSize',16)  

% % plotting energy function E
[X,Y] = meshgrid(xrange_plot(1):.002:xrange_plot(2),yrange_plot(1):.002:yrange_plot(2));
XY = [X(:)';Y(:)'];
Z = E(XY);
Z = reshape(Z,size(X));

Eplot = surf(X,Y,Z,'FaceAlpha',0.25); % 0.5 und 0.25
Eplot.EdgeColor = 'None';
hold on

contour(X,Y,Z,20);
hold on

if spatial_plot
    view(-25,12.5)
else
    view(2)
end

xlim(xrange_plot)
ylim(yrange_plot)
zlim([zrange_plot(1),zrange_plot(2)+0.01])

xticks([-2.5 0 2.5 5])
yticks([-2.5 0 2.5 5])

% way of plotting of all points
if spatial_plot
    F = @(x) E(x);
else
    F = @(x) zrange_plot(2)*ones(size(sum(x.*x))); 
end
%F = @(x) 0*zeros(size(sum(x.*x)));

% % plot global minimizer of energy function E
vstarplot = plot3(vstar(1), vstar(2), F(vstar), '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:));
hold on

for r = 1:NUM_RUNS
    
    if show_expectation
        Expectation_trajectory = reshape(Expectation_trajectories(r,:,:), [d,1+T/dt]);
        Expectation_trajectory_plot = plot3(Expectation_trajectory(1,:),Expectation_trajectory(2,:), F(Expectation_trajectory), '-', 'Linewidth', 2, "color", [co(2,:),opacity_sampletrajectories]);
        hold on
    end
    
    for e = 1:NUM_EX
        Vex_trajectory = reshape(Vex_trajectories(r,:,e,:), [d,1+T/dt]);
        Vex_trajectory_plot = plot3(Vex_trajectory(1,:),Vex_trajectory(2,:), F(Vex_trajectory), '-', 'Linewidth', 2, "color", [co(3,:),opacity_sampletrajectories]);
        hold on
    end
    
end

if show_expectation
    Expectation_mean_trajectory = reshape(mean(Expectation_trajectories,1), [d,1+T/dt]);
    Expectation_mean_trajectory_plot = plot3(Expectation_mean_trajectory(1,:),Expectation_mean_trajectory(2,:), F(Expectation_mean_trajectory), '-', 'Linewidth', 2.5, "color", grayish_averagetrajectory*co(2,:));
    hold on
end

for e = 1:NUM_EX
    Vex_mean_trajectory = reshape(mean(Vex_trajectories(:,:,e,:),1), [d,1+T/dt]);
    Vex_mean_trajectory_plot = plot3(Vex_mean_trajectory(1,:),Vex_mean_trajectory(2,:), F(Vex_mean_trajectory), '-', 'Linewidth', 2.5, "color", grayish_averagetrajectory*co(3,:));
end

% add initial positions
Vex_0 = plot3(Vex(1,:), Vex(2,:), F(Vex)+0.01, '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", grayish_averagetrajectory*co(3,:));
hold on

% % replot global minimizer of energy function E
vstarplot = plot3(vstar(1), vstar(2), F(vstar)+0.01, '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:), 'MarkerEdgeColor', co(5,:), 'MarkerFaceColor', co(5,:));
hold on

if show_expectation
    legend([vstarplot, Vex_0, Vex_trajectory_plot, Vex_mean_trajectory_plot, Expectation_trajectory_plot, Expectation_mean_trajectory_plot], ...
        'Global minimizer $v^*$', ...
        'Initial positions of fixed particles',...
        'Sample trajectories of each fixed particle', ...
        'Average trajectory of each fixed particle', ...
        'Sample trajectories of the average particle $\textbf{E}\overline{V}$', ...
        'Average trajectory of the average particle','Location','northeast','Interpreter','latex','FontSize',16)
else
    legend([vstarplot, Vex_0, Vex_trajectory_plot, Vex_mean_trajectory_plot], ...
        'Global minimizer $v^*$', ...
        'Initial positions of fixed particles',...
        'Sample trajectories for each fixed particle', ...
        'Mean trajectory for each fixed particle', 'Interpreter','latex','FontSize',16)
end

ax = gca;
ax.FontSize = 14;


%% Save Image
if pdfexport
    disp('Needs to be saved manually to obtain high resolution.')
    disp('(File -> Export Setup -> Rendering -> Resolution: 2400dpi; Star for v* needs to be added manually.)')
    %print(f,['images_videos/CBOIntuition_',objectivefunction],'-dpdf');

    filename = ['CBOIntuition_',objectivefunction,'N',num2str(N),'sigma',num2str(100*sigma),'div100'];
    save(['images_videos/',filename,'_param'], 'objectivefunction', 'E', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std', 'Vex', 'NUM_RUNS')

    disp('Filename when saved in higher resolution:')
    disp(filename)
    saveas(f,['images_videos/',filename,'.jpg']);
end


%% slightly modified CBO Function
function [Vex_trajectory,Expectation_trajectory] = CBO_trajectories(E,parametersCBO,V0mean,V0std,Vex)

% get parameters
[d,~] = size(Vex);
T = parametersCBO('T');
dt = parametersCBO('dt');
N = parametersCBO('N');

% storage for trajectories
s_Vex = size(Vex);
NUM_EX = s_Vex(2);

Expectation_trajectory = zeros(d,1+T/dt);
Vex_trajectory = zeros(d,NUM_EX,1+T/dt);

%initialization
V0 = V0mean+V0std*randn(d,N-NUM_EX);
V = [Vex,V0];

Expectation_trajectory(:,1) = sum(V,2)/N;
Vex_trajectory(:,1:NUM_EX,1) = V(:,1:NUM_EX);

% CBO
for k = 1:T/dt
    
    [V,~] = CBO_iteration(E,parametersCBO,V);
    
    Expectation = sum(V,2)/N;
    Expectation_trajectory(:,k+1) = Expectation;
    Vex_trajectory(:,1:NUM_EX,k+1) = V(:,1:NUM_EX);
    
end

end

