% CBOMemoryGradient illustrative
%
% This script illustrates the optimization procedure of CBO with memory 
% effects and gradient information for 2d objective functions.
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
% use pre-set PSO setting (overrides manually chosen parameters)
pre_setparameters = 0;

% 3d plot
spatial_plot = 0;

% save video
savevideo = 0;


%% Energy Function E

% % dimension of the ambient space
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'Rastrigin';
[E, grad_E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'CBOMemoryGradient');

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
zrange_plot = parametersE(:,3)';
xrange = 100*xrange_plot;
yrange = 100*yrange_plot;


% global minimizer
xstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of PSO Algorithm

% time horizon
T = 8;

% discrete time size
dt = 0.1;
 
% number of particles
N = 10;

%
kappa = 1/dt;

% memory
memory = 1; % 0 or 1
% lambda1, sigma1, kappa and beta have no effect for memory=0.

% lambda1 (drift towards global and in-time best parameter)(drift towards in-time best parameter)
lambda1 = 1;
% lambda2 (drift towards in-time best parameter)
lambda2 = 0;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% type of diffusion for noise term 1
anisotropic1 = 1;
% sigma (parameter of noise term 1)
sigma1 = sqrt(0.4);
% type of diffusion for noise term 2
anisotropic2 = 1;
% sigma (parameter of noise term 2)
sigma2 = lambda2*sigma1;

% alpha (weight in Gibbs measure for global and in-time best position computation)
alpha = 10^15;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta


%% Initialization
X0mean = 4*ones(d,1);
X0std = 8;


parametersCBOmemorygradient = containers.Map({'T', 'dt', 'N', 'kappa', 'memory', 'lambda1', 'lambda2', 'gamma', 'learning_rate', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'alpha', 'beta'},...
                                             {  T,   dt,   N,   kappa,   memory,   lambda1,   lambda2,   gamma,   learning_rate,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   alpha,   beta});
parametersInitialization = containers.Map({'X0mean', 'X0std'},...
                                          {  X0mean,   X0std});


%% Plotting

% % plot setting
figure('Position', [1200 800 500 400]);
set(gcf,'color','w');
%title('Consensus-Based Optimization with Memory Effects','Interpreter','latex','FontSize',16)  
% % plotting energy function E
[X_g,Y_g] = meshgrid(xrange_plot(1):.1:xrange_plot(2),yrange_plot(1):.1:yrange_plot(2));
XY = [X_g(:)';Y_g(:)'];
Z_g = E(XY);
Z_g = reshape(Z_g,size(X_g));

Eplot = surf(X_g,Y_g,Z_g,'FaceAlpha',0.5);
Eplot.EdgeColor = 'None';
hold on

contour(X_g,Y_g,Z_g,20);

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
xstarplot = plot3(xstar(1), xstar(2), F(xstar), '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:));
hold on

%title('Setting','Interpreter','latex','FontSize',16)
if spatial_plot
    legend([xstarplot], 'Global minimizer $x^*$','Location','northwest','Interpreter','latex','FontSize',13)
else
    legend([xstarplot], 'Global minimizer $x^*$','Location','northwest','Interpreter','latex','FontSize',13)
end

if savevideo
    frame(1) = getframe(gcf);
end
    
pause(dt)
if manual_steps
    pause()
end


%% CBOmemory Algorithm
%initialization
X0 = X0mean+X0std*randn(d,N);
Y0 = X0;
X = X0;
Y = Y0;

% plot initial setting
fprintf("t=0\n")
%title(sprintf("CBOmemory at time $t=0$"),'Interpreter','latex','FontSize',16)

X_plot = scatter3(X0(1,:), X0(2,:), F(X0), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
Y_plot = scatter3(Y0(1,:), Y0(2,:), F(Y0), 20, "MarkerEdgeColor", co(3,:));
hold on
if spatial_plot
    legend([xstarplot, X_plot],'Global minimizer $x^*$','Particles $X_0^i$','Location','northwest','Interpreter','latex','FontSize',13)
else
    legend([xstarplot, X_plot],'Global minimizer $x^*$','Particles $X_0^i$','Location','northwest','Interpreter','latex','FontSize',13)
end

if savevideo
    frame(2) = getframe(gcf);
end

%%
% CBOmemorygradient
for k = 1:T/dt
    
    pause(dt)
    if manual_steps
        pause()
    end

    t = k*dt;
    fprintf("t=%d\n", t)
    %title(sprintf("CBOmemory at time $t=%d$",t),'Interpreter','latex','FontSize',16)
    
    % % CBOmemory iteration
    % compute global and in-time best position y_alpha
    y_alpha = compute_yalpha(E, alpha, Y);
    
    % position updates of one iteration of PSO
    [X, Y] = CBOmemorygradient_update(E, grad_E, parametersCBOmemorygradient, y_alpha, X, Y);
    
    % % Visualization of the way PSO optimizes non-convex functions
    % remove all old plotting objects
    delete(X_plot)
    delete(Y_plot)
    if k~=1
        delete(valpha_plot)
        if show_expectation
            delete(Expectation_plot)
        end
    end
    
    % plotting of particles
    X_plot = scatter3(X(1,:), X(2,:), F(X), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
    Y_plot = scatter3(Y(1,:), Y(2,:), F(Y), 20, "MarkerEdgeColor", co(3,:));
    % plotting of expectation
    Expectation = sum(X,2)/N;
    if show_expectation
        Expectation_plot = plot3(Expectation(1), Expectation(2), F(Expectation), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));
    end
    % plotting of consensus point
    valpha_plot = plot3(y_alpha(1), y_alpha(2), F(y_alpha), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));
    
    if show_expectation
        if spatial_plot
            legend([xstarplot, X_plot, Y_plot, Expectation_plot, valpha_plot], 'Global minimizer $x^*$', 'Particles $X_t^i$', 'Particles'' best $Y_t^i$', 'Average particle $\textbf{E}\overline{X}_t$', 'Consensus point $y_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
            %legend([vstarplot, V_plot, Y_plot, Expectation_plot, valpha_plot], 'Global minimizer $x^*$', 'Particles $X_t^i$', 'Particles'' best $Y_t^i$', 'Average particle $\textbf{E}\overline{X}_t$', 'Consensus point $y_{\alpha}(\widehat\rho_t^N)$','Position',[0.175 0.675 0.1 0.2],'Interpreter','latex','FontSize',13))
        else
            legend([xstarplot, X_plot, Y_plot, Expectation_plot, valpha_plot], 'Global minimizer $x^*$', 'Particles $X_t^i$', 'Particles'' best $Y_t^i$', 'Average particle $\textbf{E}\overline{X}_t$', 'Consensus point $y_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
        end
    else
        if spatial_plot
            legend([xstarplot, X_plot, Y_plot, valpha_plot], 'Global minimizer $x^*$', 'Particles $X_t^i$', 'Particles'' best $Y_t^i$', 'Consensus point $y_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
            %legend([vstarplot, V_plot, Y_plot, valpha_plot], 'Global minimizer $x^*$', 'Particles $X_t^i$', 'Particles'' best $Y_t^i$', 'Consensus point $y_{\alpha}(\widehat\rho_t^N)$','Position',[0.175 0.675 0.1 0.2],'Interpreter','latex','FontSize',13))
        else
            legend([xstarplot, X_plot, Y_plot, valpha_plot], 'Global minimizer $x^*$', 'Particles $X_t^i$', 'Particles'' best $Y_t^i$', 'Consensus point $y_{\alpha}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
        end
    end
    
    if savevideo
        frame(k+2) = getframe(gcf);
    end
    
end
y_alpha = compute_yalpha(E, alpha, Y);
fprintf("global minimizer (numerically): [%d;%d]\n", xstar)
fprintf("final consensus point         : [%d;%d]\n", y_alpha)

%% Save Video
if savevideo
    video = VideoWriter(['EnergyBasedCBOmemoryAnalysis/images_videos/CBOIllustrative_',objectivefunction],'MPEG-4');
    video.FrameRate = 8;
    open(video);
    writeVideo(video,frame(1));
    writeVideo(video,frame(2));
    for k = 1:T/dt
        writeVideo(video,frame(k+2));
    end
    close(video);
end
