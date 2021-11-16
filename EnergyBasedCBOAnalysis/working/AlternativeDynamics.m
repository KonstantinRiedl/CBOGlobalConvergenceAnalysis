clear; clc; close all;

co = set_color();

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultLegendInterpreter','latex')
set(0,'DefaultAxesTickLabelInterpreter','latex')
set(0,'DefaultLegendFontSize',20)
set(0,'DefaultTextFontSize',20)
set(0,'DefaultAxesFontSize',16)
set(0,'DefaultLineLineWidth',2)


%% Settings for Easy Handling and Notes
% 
% energy function has to be chosen manually
% decice if time steps require pressing some arbitrary key
manual_steps = 0;
% plotting error metric during or after iterations
errorplots_meanwhile = 1;
% plotting empirical expectation of the particles
show_expectation = 1;

% 
plotforlatexexport = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Specific Script parameters

epsilon = 1;
indicator_in_exploration = 0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Energy Function E

% % dimension of the ambient space
d = 1; % at the moment only 1d due to the convex envelope computation

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, parametersE, ~, parametersInitialization] = objective_function(objectivefunction, d, 'CBO');


% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
xrange = 100*xrange_plot;
yrange = 100*yrange_plot;


%% Parameters of CBO Algorithm
 
% discrete time size
dt = 0.01;
 
% number of particles
N = 200;
 
% lambda (parameter of consensus drift term)
lambda = 4;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 0.5;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 1;
 
 
%% Initialization
V0 = 4+0.25*randn(d,N-N/200);
V00 = -1+0.25*randn(d,N/200);
V = [V0 V00];

V1=V;
V2=V;

%% Convex Envelope Ec of E

% % computation of the convex hull of the energy function E
% we exploit the relationship between convex functions and their epigraphs
convhull_dom = linspace(xrange(1), xrange(2), 10^6);
Ec = [convhull_dom; E(convhull_dom)]';
[indices,~] = convhull(Ec);

convhull_x = Ec(indices,1); convhull_x(end) = [];
convhull_y = Ec(indices,2); convhull_y(end) = [];


%% Error Metrics

% J
J = NaN(1,1/dt);
J1 = NaN(1,1/dt);
J2 = NaN(1,1/dt);


%% Plotting

% % plot setting
if errorplots_meanwhile
    figure('Position', [1200 800 1100 1000])
    subplot(3,2,1)
else
    figure('Position', [1200 800 500 1000])
end
title('Consensus Based Optimization')  


for i=1:2:6
    subplot(3,2,i)
    % % plotting convex envelope of E
    plot(convhull_x, convhull_y, "color", co(6,:), 'LineWidth', 2, 'LineStyle', '--')
    hold on
    % % plotting energy function E
    fplot(E, xrange, "color", co(1,:), 'LineWidth', 2);
    xlim(xrange_plot)
    ylim(yrange_plot)
    hold on
    % % plot global minimizer of energy function E
    vstar = 0; %fminbnd(E,xrange_plot(1),xrange_plot(2));
    plot(vstar, E(vstar), '*', 'MarkerSize', 10, "color", co(5,:));
    hold on
end

%% CBO Algorithm
for k = 1:1/dt
    t = k*dt;
    subplot(3,2,1)
    title(sprintf("CBO at time t=%d",t))
    fprintf("t=%d\n", t)
    
    % compute current consensus point v_alpha
    v_alpha = compute_valpha(E,alpha,V);
    v1_alpha = compute_valpha(E,alpha,V1);
    v2_alpha = compute_valpha(E,alpha,V2);
    
    % Brownian motion for exploration term
    dB = randn(d,N);
    
    % % particle iteration step (according to SDE)
    % consensus drift and exploration term
    V = V - lambda*(V-v_alpha*ones(1,N))*dt + sigma*abs(V-v_alpha*ones(1,N))*sqrt(dt).*dB;
    if ~indicator_in_exploration
        V1 = V1 - lambda*(V1-v1_alpha*ones(1,N)).*((E(V1)-E(v_alpha))>0)*dt + sigma*abs(V1-v1_alpha*ones(1,N))*sqrt(dt).*dB;
        V2 = V2 - lambda*(V2-v2_alpha*ones(1,N)).*((E(V2)-E(vstar))>epsilon)*dt + sigma*abs(V2-v2_alpha*ones(1,N))*sqrt(dt).*dB;
    else
        V1 = V1 - (lambda*(V1-v1_alpha*ones(1,N))*dt + sigma*abs(V1-v1_alpha*ones(1,N))*sqrt(dt).*dB).*((E(V1)-E(v_alpha))>0);
        V2 = V2 - (lambda*(V2-v2_alpha*ones(1,N))*dt + sigma*abs(V2-v2_alpha*ones(1,N))*sqrt(dt).*dB).*((E(V2)-E(vstar))>epsilon);
    end
    % gradient drift term
    h = 10^-3;
    gradE = zeros(d,N);
    for i = 1:d
        dV = h*zeros(d,N);
        dV(i,:) = ones(1,N);
        gradE = (E(V+h*dV)-E(V-h*dV))/(2*h);
        gradE1 = (E(V1+h*dV)-E(V1-h*dV))/(2*h);
        gradE2 = (E(V2+h*dV)-E(V2-h*dV))/(2*h);
    end
    V = V - gamma*learning_rate*gradE*dt;
    V1 = V1 - gamma*learning_rate*gradE1*dt;
    V2 = V2 - gamma*learning_rate*gradE2*dt;
    
    % plotting of particles
    if errorplots_meanwhile
        subplot(3,2,1)
    end
    V_plot = scatter(V, E(V), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
    subplot(3,2,1)
    valpha_plot = plot(v_alpha, E(v_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
    
    if errorplots_meanwhile
        subplot(3,2,3)
    end
    V1_plot = scatter(V1, E(V1), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
    subplot(3,2,3)
    v1alpha_plot = plot(v1_alpha, E(v1_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
    
    if errorplots_meanwhile
        subplot(3,2,5)
    end
    V2_plot = scatter(V2, E(V2), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
    subplot(3,2,5)
    v2alpha_plot = plot(v2_alpha, E(v2_alpha), '.', 'MarkerSize', 20, "color", co(2,:));

    
    % % Computation of Error Metrics
    % Functional J
    J(k) = sum(interp1(convhull_x,convhull_y,V,'linear')-E(vstar))/N;
    J1(k) = sum(interp1(convhull_x,convhull_y,V1,'linear')-E(vstar))/N;
    J2(k) = sum(interp1(convhull_x,convhull_y,V2,'linear')-E(vstar))/N;
    
    
    % Variance
    Expectation = sum(V)/N;
    if show_expectation
        delete(valpha_plot)
        subplot(3,2,1)
        valpha_plot = plot(v_alpha, E(v_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
        delete(v1alpha_plot)
        subplot(3,2,3)
        v1alpha_plot = plot(v1_alpha, E(v1_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
        delete(v2alpha_plot)
        subplot(3,2,5)
        v2alpha_plot = plot(v2_alpha, E(v2_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
    end
    Variance(k) = sum((V-Expectation).^2)/N;
    VarianceStar(k) = sum((V-vstar).^2)/N;
    
    % normalization (for plotting convenience)
    if k==1
        normal_J = J(1);
        normal_J1 = J1(1);
        normal_J2 = J2(1);
    end
    J(k) = J(k)/normal_J;
    J1(k) = J1(k)/normal_J1;
    J2(k) = J2(k)/normal_J2;
    
    % plotting of error metrics (meanwhile)
    if errorplots_meanwhile
        subplot(1,2,2)
        plot_errormetric(dt,J,J1,J2)
    end

    pause(dt)
    if manual_steps
        pause()
    end
    
    % plotting
    delete(valpha_plot)
    delete(v1alpha_plot)
    delete(v2alpha_plot)
    if t~=1
        delete(V_plot)
        if t~=-1
            delete(V_plot)
            delete(V1_plot)
            delete(V2_plot)
        end
    else
        subplot(3,2,1)
        valpha_plot_final = plot(v_alpha, E(v_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
        subplot(3,2,3)
        v1alpha_plot_final = plot(v1_alpha, E(v1_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
        subplot(3,2,5)
        v2alpha_plot_final = plot(v2_alpha, E(v2_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
    end
    
end
v_alpha = compute_valpha(E,alpha,V);
fprintf("global minimizer (numerically): %d\n", vstar)
fprintf("final consensus point         : %d\n", v_alpha)
fprintf("final consensus point (mod1)  : %d\n", v1_alpha)
fprintf("final consensus point (mod2)  : %d\n", v2_alpha)


% plotting of error metrics (afterwards)
if ~errorplots_meanwhile
    figure('Position', [1700 800 500 1000])
    plot_errormetric(dt,J,J1,J2)
end

if plotforlatexexport
    disp('Not availble yet.')
    %matlab2tikz('myfile.tex');
end

%% Computation of consensus point v_alpha
function v_alpha = compute_valpha(E,alpha,V)

Es = E(V);   
    
Emin = min(Es);
v_alpha = sum((V.*exp(-alpha*(Es-Emin))),2);
v_alpha = v_alpha/sum(exp(-alpha*(Es-Emin)));

end


%% error metric plotting routine
function plot_errormetric(dt,J,J1,J2)
co = set_color();
xlim([0,1])
ylim([0,2])
xlabel('$t$')

plot(dt:dt:1,J, "color", co(1,:), 'LineWidth', 2, 'LineStyle', '-.')
hold on;
plot(dt:dt:1,J1, "color", co(2,:), 'LineWidth', 2, 'LineStyle', '--')
hold on;
plot(dt:dt:1,J2, "color", co(3,:), 'LineWidth', 2, 'LineStyle', ':')
hold on;

legend('$\mathcal{J}(\rho_t)$','$\mathcal{J}1(\rho_t)$','$\mathcal{J}2(\rho_t)$')
end

%% Auxiliary Functions
function co = set_color()
co = [0         0.4470    0.7410
      0.8500    0.3250    0.0980
      0.9290    0.6940    0.1250
      0.4940    0.1840    0.5560
      0.4660    0.6740    0.1880
      0.3010    0.7450    0.9330
      0.6350    0.0780    0.1840
      0.6667    0.3333    0
      0.8000    0.8000    0.8000];
set(groot,'defaultAxesColorOrder',co)
end



