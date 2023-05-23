% CH Scheme intuition
%
% This script illustrates the dynamcis of the consensus hopping (CH)
% scheme, when optimizing nonconvex objective functions. 
% Such plot is used in Figure 2(a) in "Gradient is All You Need?"
%

%%
clear; clc; close all;

co = set_color();

%% Settings for Easy Handling and Notes
% 3d plot
spatial_plot = 0;

% save plot
pdfexport = 0;


%% Energy Function E

% % dimension of the ambient space
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'GrandCanyon3noisy';
[E, grad_E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'CBO');

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
zrange_plot = parametersE(:,3)';
xrange = 100*xrange_plot;
yrange = 100*yrange_plot;

% global minimizer
vstar = zeros(d,1);


%% Parameters of CBO Algorithm
 
% discrete time horizon
K = 250; % ConsensusHopping: 100

% number of particles
N = 200; % 40, 200

% sigma (parameter of exploration term)
sigma = 0.7; % 0.4, 0.6, 0.7

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 100; % 10, 100, 10^15


%% Initialization
V0mean = [8;8]; %[5;-1];
V0std = sigma;


%% Set containers
parametersCH = containers.Map({'K', 'N', 'alpha', 'sigma'},...
                               {  K,   N,   alpha,   sigma});
parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                          {  V0mean,   V0std});


%% Data Generation

NUM_RUNS = 50;
Consensuspoint_Average_trajectories = zeros(NUM_RUNS,d,K);
for r = 1:NUM_RUNS
    
    Consensuspoint_Average_trajectories(r,:,:) = CH_trajectories(E, parametersCH, V0mean, V0std);
    
end


%% Plotting

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% plot setting
f = figure('Position', [1200 800 600 500]);
%title('Consensus Based Optimization','Interpreter','latex','FontSize',16)  

% plotting energy function E
[X,Y] = meshgrid(xrange_plot(1):.01:xrange_plot(2),yrange_plot(1):.01:yrange_plot(2));
XY = [X(:)';Y(:)'];
Z = E(XY);
Z = reshape(Z,size(X));

Eplot = surf(X,Y,Z,'FaceAlpha',0.25); % 0.5 und 0.25
Eplot.EdgeColor = 'None';
hold on

contour(X,Y,Z,22);
hold on

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

if spatial_plot
    view(-25,12.5)
else
    view(2)
end

xlim(xrange_plot)
ylim(yrange_plot)
if spatial_plot
    zlim([zrange_plot(1),zrange_plot(2)+0.01])
else
    zlim([zrange_plot(1),zrange_plot(2)+10])
end

xticks([-2.5 0 2.5 5])
yticks([-2.5 0 2.5 5])

if strcmp(objectivefunction,'GrandCanyon2') || strcmp(objectivefunction,'GrandCanyon2noisy')
   
    xticks([-2.5 0 2.5 5])
    yticks([-2.5 0 2.5 5])

elseif strcmp(objectivefunction,'GrandCanyon3') || strcmp(objectivefunction,'GrandCanyon3noisy')
   
    xticks([-2 0 2 4 6 8])
    yticks([-2 0 2 4 6 8])

end

if spatial_plot
    zlim([zrange_plot(1),zrange_plot(2)+0.01])
else
    zlim([zrange_plot(1),zrange_plot(2)+10])
end

% plot trajectories
% opacity_sampletrajectories = 0.6;
% for r = 1:NUM_RUNS
%     
% 	Consensuspoint_trajectory = reshape(Consensuspoint_trajectories(r,:,:), [d,K]);
%  	Consensuspoint_trajectory_plot = plot3(Consensuspoint_trajectory(1,:),Consensuspoint_trajectory(2,:), F(Consensuspoint_trajectory), '-', 'Linewidth', 2, "color", [co(2,:),opacity_sampletrajectories]);
%  	hold on
%     
% end

% plot points
opacity_sampletrajectories = 0.6;
c1 = co(1,:);
c2 = co(2,:);
color_map = [linspace(c1(1), c2(1), K)', linspace(c1(2), c2(2), K)', linspace(c1(3), c2(3), K)']; %winter(K);
number_plot_points = K;
for ii = 1:number_plot_points
    
    timestamps_plot = floor(linspace(1,K,number_plot_points));
    i = timestamps_plot(ii);
	Consensuspoints_Averages_i = reshape(Consensuspoint_Average_trajectories(:,:,i), [NUM_RUNS,d])'; 
    Consensuspoint_trajectory_plot = scatter3(Consensuspoints_Averages_i(1,:),Consensuspoints_Averages_i(2,:), F(Consensuspoints_Averages_i), 'MarkerFaceColor', color_map(i,:), 'MarkerFaceAlpha', opacity_sampletrajectories, 'MarkerEdgeAlpha',0);
    hold on
    
end

%Consensuspoint_mean_trajectory = reshape(mean(Consensuspoint_trajectories(:,:,:),1), [d,K]);
%Consensuspoint_mean_trajectory_plot = plot3(Consensuspoint_mean_trajectory(1,:),Consensuspoint_mean_trajectory(2,:), F(Consensuspoint_mean_trajectory), '-', 'Linewidth', 2.5, "color", grayish_averagetrajectory*co(2,:));

% % replot global minimizer of energy function E
vstarplot = plot3(vstar(1), vstar(2), F(vstar)+0.01, '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:), 'MarkerEdgeColor', co(5,:), 'MarkerFaceColor', co(5,:));
hold on

legend([vstarplot, Consensuspoint_trajectory_plot], ...
        'Global minimizer $x^*$', ...
        'Sample locations of consensus point', 'Interpreter','latex','FontSize',16, 'Location','northwest')

% legend([vstarplot, Consensuspoint_trajectory_plot, Consensuspoint_mean_trajectory_plot], ...
%         'Global minimizer $x^*$', ...
%         'Sample trajectories for consensus point', ...
%         'Mean trajectory for consensus point', 'Interpreter','latex','FontSize',16)

ax = gca;
ax.FontSize = 14;


%% Save Image
if pdfexport
    disp('Needs to be saved manually to obtain high resolution.')
    disp('(File -> Export Setup -> Rendering -> Resolution: 2400dpi; Star for v* needs to be added manually.)')
    %print(f,['EnergyBasedCBOAnalysis/images_videos/CBOIntuition_averageconsensuspoint_',objectivefunction],'-dpdf');

    if anisotropic
        filename = ['CBOIntuition_averageconsensuspoint_',objectivefunction,'_anisotropicN',num2str(N),'sigma',num2str(100*sigma),'div100'];
    else
        filename = ['CBOIntuition_averageconsensuspoint_',objectivefunction,'_isotropicN',num2str(N),'sigma',num2str(100*sigma),'div100'];
    end
    save([main_folder(),'/EnergyBasedCBOAnalysis/images_videos/',filename,'_param'], 'objectivefunction', 'E', 'anisotropic', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std', 'Vex', 'NUM_RUNS')

    disp('Filename when saved in higher resolution:')
    disp(filename)
    saveas(f,[main_folder(),'/EnergyBasedCBOAnalysis/images_videos/',filename,'.jpg']);
end


%% modified CH Function to include trajectory
function [CH_trajectory] = CH_trajectories(E, parametersCH, V0mean, V0std)

% get parameters
d = size(V0mean,1);
K = parametersCH('K');
N = parametersCH('N');
alpha = parametersCH('alpha');


% storage for trajectories
CH_trajectory = zeros(d,K);

%initialization
V0 = V0mean + V0std*randn(d,N);
V = V0;


% % CH
for k = 1:K

    % % CH iteration
    % compute current consensus point v_alpha
    v_alpha = compute_valpha(E, alpha, V);
    CH_trajectory(:,k) = v_alpha;

    % sample randomly N points around consensus point
    V = v_alpha + parametersCH('sigma')*randn(d,N);
    
end

end

