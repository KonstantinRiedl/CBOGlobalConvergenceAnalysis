% Auxiliary script for plotting the initial setting and the dynamics 
% associated with the script CBO Illustrative
% (in particular, for the latex export the star is plotted manually)
% Such plot is used in Figure 1(a).
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

% save initial plot (and only run initial plot)
pdfexport = 0;

% save video
savevideo = 0;


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
T = 4;

% discrete time size
dt = 0.02;
 
% number of particles
N = 3200; 

% lambda (parameter of consensus drift term)
lambda = 1;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 0.1;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^15;


%% Initialization
V0mean = [8;8];
V0std = 20;


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


%% Plotting

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% % color setting
grayish_averagetrajectory = 0.8;

% % plot setting
f = figure('Position', [1200 800 500 400]);
%title('Consensus Based Optimization','Interpreter','latex','FontSize',16)  

% % plotting energy function E
[X,Y] = meshgrid(xrange_plot(1):.002:xrange_plot(2),yrange_plot(1):.002:yrange_plot(2));
XY = [X(:)';Y(:)'];
Z = E(XY);
Z = reshape(Z,size(X));

Eplot = surf(X,Y,Z,'FaceAlpha',0.4);
Eplot.EdgeColor = 'None';
hold on

contour(X,Y,Z,20);

view(-25,12.5) % do NOT change


xlim(xrange_plot)
ylim(yrange_plot)
zlim(zrange_plot)
if strcmp(objectivefunction,'Rastrigin')
	xticks([-2.5 0 2.5 5])
    yticks([-2.5 0 2.5 5])
	zticks([0 10 20 30 40 50])
end


% way of plotting of all points
F = @(x) E(x);
%F = @(x) 0*zeros(size(sum(x.*x)));

% % plot global minimizer of energy function E
vstarplot = plot3(vstar(1), vstar(2), F(vstar), '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:));

hold on

if ~pdfexport
    title('Setting','Interpreter','latex','FontSize',16)
end
legend([vstarplot], 'Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',13)

if savevideo
    frame(1) = getframe(gcf);
end

if manual_steps && ~pdfexport
    pause()
end

% plotting vstar manually for high-resolution pdf export
if pdfexport
    %%%
    % | nach oben
    plot3([0,0], [0,0], 2*0.55*[0,1], '-', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:),'HandleVisibility','off');
    % | nach unten
    plot3(2*0.158*[0,-1], 2*0.158*[0,-2.145], [0,0], '-', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:),'HandleVisibility','off');
    % - links nach rechts
    plot3(2*0.03*[-2.145,2.145], 2*0.03*[1,-1], [0,0], '-', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:),'HandleVisibility','off');
    % / unten
    plot3(1.87*0.084*[-2.18,0], 1.85*0.087*[-3.02,0], [0,0], '-', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:),'HandleVisibility','off');
    % \ unten
    plot3(2*0.073*[-1,0], 2*0.073*[-3.92,0], [0,0], '-', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:),'HandleVisibility','off');
    % \ oben
    plot3(1.85*0.065*[0,-1], [0,0], 1.85*0.065*[0,7.6], '-', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:),'HandleVisibility','off');
    % / oben
    plot3(2*0.06*[0,1], [0,0], 2*0.06*[0,6.375], '-', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:),'HandleVisibility','off');
    hold on

    contour(X,Y,Z,20,'HandleVisibility','off');
end



%% CBO Algorithm
%initialization
V0 = V0mean+V0std*randn(d,N-3);

% plot initial setting
if ~pdfexport
    fprintf("t=0\n")
end
if ~pdfexport
    title(sprintf("CBO at time t=0"),'Interpreter','latex','FontSize',16)
end

V_plot = scatter3(V0(1,:), V0(2,:), F(V0), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
hold on

Vex = [[-2;4],[4.5;1.5],[-1.5;-1.5]];
[~, NUM_EX] = size(Vex);
Vex_plot = plot3(Vex(1,:), Vex(2,:), F(Vex), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", grayish_averagetrajectory*co(3,:));
hold on

%
V = [V0,Vex];
[V,v_alpha] = CBO_iteration(E,parametersCBO,V);
% plotting of consensus point
valpha_plot = plot3(v_alpha(1), v_alpha(2), F(v_alpha), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));
    
legend([vstarplot, Vex_plot, V_plot, valpha_plot],'Global minimizer $v^*$','Initial positions of fixed particles', 'Random particles', 'Initial consensus point $v_{\alpha}(\widehat\rho_0^N)$','Location','northwest','Interpreter','latex','FontSize',13)

ax = gca;
ax.FontSize = 11;


%% Save Image
if pdfexport
    disp('Needs to be saved manually to obtain high resolution.')
    disp('(File -> Export Setup -> Rendering -> Resolution: 2400dpi; Star for v* needs to be added manually.)')
    %print(f,['images_videos/CBOIntuition_',objectivefunction],'-dpdf');

    filename = ['CBOIntuitionObjectiveFunction_',objectivefunction,'N',num2str(N)];
    save(['images_videos/',filename,'_param'], 'objectivefunction', 'E', 'vstar', 'd', 'N', 'alpha', 'V0mean', 'V0std', 'Vex')

    disp('Filename when saved in higher resolution:')
    disp(filename)
    saveas(f,['images_videos/',filename,'.jpg']);
end

if savevideo
    frame(2) = getframe(gcf);
end


%%
if ~pdfexport
    % CBO
    for k = 1:T/dt

        pause(dt)
        if manual_steps
            pause()
        end

        t = k*dt;
        fprintf("t=%d\n", t)
        title(sprintf("CBO at time t=%d",t),'Interpreter','latex','FontSize',16)

        % % CBO iteration
        [V,v_alpha] = CBO_iteration(E,parametersCBO,V);

        % % Visualization of the way CBO optimizes non-convex functions
        % remove all old plotting objects
        delete(V_plot)
        delete(Vex_plot)
        delete(valpha_plot)
        if show_expectation
            delete(Expectation_plot)
        end
        
        % plotting of particles
        F_val = F(V);
        V_plot = scatter3(V(1,NUM_EX+1:end), V(2,NUM_EX+1:end), F_val(:,NUM_EX+1:end), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
        Vex = V(:,1:NUM_EX);
        Vex_plot = plot3(Vex(1,:), Vex(2,:), F(Vex), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", grayish_averagetrajectory*co(3,:));
        % plotting of expectation
        Expectation = sum(V,2)/N;
        if show_expectation
            Expectation_plot = plot3(Expectation(1), Expectation(2), F(Expectation), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));
        end
        % plotting of consensus point
        valpha_plot = plot3(v_alpha(1), v_alpha(2), F(v_alpha), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));


        if show_expectation
            legend([vstarplot, Vex_plot, V_plot, Expectation_plot, valpha_plot], 'Global minimizer $v^*$','Initial position for each fixed particle', 'Initial positions of random particles', 'Average particle $\textbf{E}\overline{V}_t$', 'Consensus point $v_{\alpha}^{\mathcal{E}}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
        else
            legend([vstarplot, Vex_plot, V_plot, valpha_plot], 'Global minimizer $v^*$','Initial position for each fixed particle', 'Initial positions of random particles', 'Consensus point $v_{\alpha}^{\mathcal{E}}(\widehat\rho_t^N)$','Location','northwest','Interpreter','latex','FontSize',13)
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
        video = VideoWriter(['images_videos/CBOIllustrative_',objectivefunction],'MPEG-4');
        open(video);
        video.FrameRate = 8;
        writeVideo(video,frame(1));
        writeVideo(video,frame(2));
        for k = 1:T/dt
            writeVideo(video,frame(k+2));
        end
        close(video);
        % save parameters
        save(['images_videos/CBOIllustrative_',objectivefunction,'_param'], 'objectivefunction', 'E', 'vstar', 'd', 'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma', 'V0mean', 'V0std')
    end
end
