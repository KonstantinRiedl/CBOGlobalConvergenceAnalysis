% CBO illustrative
%
% This script illustrates the optimization procedure of CBO for 2d
% objective functions.
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
spatial_plot = 1;

% save video
savevideo = 0;


%% Energy Function E

% % dimension of the ambient space
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, parametersE, parametersCBO, parametersInitialization] = objective_function(objectivefunction, d, 'CBO');

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
T = 4;

% discrete time size
dt = 0.02;
 
% number of particles
N = 25;

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
alpha = 10^15;


%% Initialization
V0mean = [6;6];
V0std = 8;


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


%% Plotting

% % plot setting
figure('Position', [1200 800 500 400]);
set(gcf,'color','w');
%title('Consensus Based Optimization','Interpreter','latex','FontSize',16)  
% % plotting energy function E
[X,Y] = meshgrid(xrange_plot(1):.1:xrange_plot(2),yrange_plot(1):.1:yrange_plot(2));
XY = [X(:)';Y(:)'];
Z = E(XY);
Z = reshape(Z,size(X));

Eplot = surf(X,Y,Z,'FaceAlpha',0.5);
Eplot.EdgeColor = 'None';
hold on

contour(X,Y,Z,20);

if spatial_plot
    view(-25,12.5)
else
    view(2)
end

xlim(xrange_plot)
ylim(yrange_plot)
zlim(zrange_plot)
if strcmp(objectivefunction,'Rastrigin')
	xticks([-2.5 0 2.5 5])
    yticks([-2.5 0 2.5 5])
	zticks([0 10 20 30 40 50])
end

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

%title('Setting','Interpreter','latex','FontSize',16)
if spatial_plot
    legend([vstarplot], 'Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',13)
else
    legend([vstarplot], 'Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',13)
end

if savevideo
    frame(1) = getframe(gcf);
end
    
pause(dt)
if manual_steps
    pause()
end


%% CBO Algorithm
%initialization
V0 = V0mean+V0std*randn(d,N);
V = V0;

% plot initial setting
fprintf("t=0\n")
%title(sprintf("CBO at time $t=0$"),'Interpreter','latex','FontSize',16)

V_plot = scatter3(V0(1,:), V0(2,:), F(V0), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
hold on
if spatial_plot
    legend([vstarplot, V_plot],'Global minimizer $v^*$','Particles $V_0^i$','Location','northwest','Interpreter','latex','FontSize',13)
else
    legend([vstarplot, V_plot],'Global minimizer $v^*$','Particles $V_0^i$','Location','northwest','Interpreter','latex','FontSize',13)
end

if savevideo
    frame(2) = getframe(gcf);
end

%%
% CBO
for k = 1:T/dt
    
    pause(dt)
    if manual_steps
        pause()
    end

    t = k*dt;
    fprintf("t=%d\n", t)
    %title(sprintf("CBO at time $t=%d$",t),'Interpreter','latex','FontSize',16)
    
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
    V_plot = scatter3(V(1,:), V(2,:), F(V), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
    % plotting of expectation
    Expectation = sum(V,2)/N;
    if show_expectation
        Expectation_plot = plot3(Expectation(1), Expectation(2), F(Expectation), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));
    end
    % plotting of consensus point
    valpha_plot = plot3(v_alpha(1), v_alpha(2), F(v_alpha), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));
    
    if show_expectation
        if spatial_plot
            legend([vstarplot, V_plot, Expectation_plot, valpha_plot], 'Global minimizer $v^*$', 'Particles $V_t^i$', 'Average particle $\textbf{E}\overline{V}_t$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
            %legend([vstarplot, V_plot, Expectation_plot, valpha_plot], 'Global minimizer $v^*$', 'Particles $V_t^i$', 'Average particle $\textbf{E}\overline{V}_t$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Position',[0.175 0.675 0.1 0.2],'Interpreter','latex','FontSize',13))
        else
            legend([vstarplot, V_plot, Expectation_plot, valpha_plot], 'Global minimizer $v^*$', 'Particles $V_t^i$', 'Average particle $\textbf{E}\overline{V}_t$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
        end
    else
        if spatial_plot
            legend([vstarplot, V_plot, valpha_plot], 'Global minimizer $v^*$', 'Particles $V_t^i$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
            %legend([vstarplot, V_plot, valpha_plot], 'Global minimizer $v^*$', 'Particles $V_t^i$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Position',[0.175 0.675 0.1 0.2],'Interpreter','latex','FontSize',13))
        else
            legend([vstarplot, V_plot, valpha_plot], 'Global minimizer $v^*$', 'Particles $V_t^i$', 'Consensus point $v_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
        end
    end
    
    if savevideo
        frame(k+2) = getframe(gcf);
    end
    
end
v_alpha = compute_valpha(E,alpha,V);
fprintf("global minimizer (numerically): [%d;%d]\n", vstar)
fprintf("final consensus point         : [%d;%d]\n", v_alpha)

%% Save Video
if savevideo
    if anisotropic
        video = VideoWriter(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/CBOIllustrative_',objectivefunction,'_anisotropic'],'MPEG-4');
    else
        video = VideoWriter(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/CBOIllustrative_',objectivefunction,'_isotropic'],'MPEG-4');
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
        save(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/CBOIllustrative_',objectivefunction,'_anisotropic_param'], 'objectivefunction', 'E', 'anisotropic', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std')
    else
        save(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/CBOIllustrative_',objectivefunction,'_isotropic_param'], 'objectivefunction', 'E', 'anisotropic', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std')
    end
end
