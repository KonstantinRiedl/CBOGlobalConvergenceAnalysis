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
% error metrics to be plotted
Metrics = "all"; % all, J or V
% the error metrics J and V are normalized by their initial value
% the global minimizer shall be at x=0, i.e., vstar=0
% plotting error metric during or after iterations
errorplots_meanwhile = 1;
% plotting empirical expectation of the particles
show_expectation = 1;

% 
plotforlatexexport = 1;


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
N = 20000;
 
% lambda (parameter of consensus drift term)
lambda = 4;
% gamma (parameter of gradient drift term)
gamma = 4;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 0.5;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 40;
 
 
%% Initialization
V0 = 4+0.1*randn(d,N-N/200);
V00 = -1+0.1*randn(d,N/200);
V = [V0 V00];
gm = gmdistribution([4;-1],0.1,[1-1/200;1/200]);
[V,~] = random(gm,N); V=V';

%% Convex Envelope Ec of E

% % computation of the convex hull of the energy function E
% we exploit the relationship between convex functions and their epigraphs
convhull_dom = linspace(xrange(1), xrange(2), 10^6);
Ec = [convhull_dom; E(convhull_dom)]';
[indices,~] = convhull(Ec);

convhull_x = Ec(indices,1); convhull_x(end) = [];
convhull_y = Ec(indices,2); convhull_y(end) = [];


%% Error Metrics

% Variance
Variance = NaN(1,1/dt);
% Variance around Minimizer v*
VarianceStar = NaN(1,1/dt);
% J with E
J_E = NaN(1,1/dt);
% weighted J
J_a = NaN(1,1/dt);
% weighted J
J_a2 = NaN(1,1/dt);
% J
J = NaN(1,1/dt);


%% Set D(delta)

% Volume of the set D(delta)
delta = 0.1;
volDdelta = NaN(1,1/dt);


%% Plotting

% % plot setting
if errorplots_meanwhile
    figure('Position', [1200 800 1100 800])
    subplot(2,2,1)
else
    figure('Position', [1200 800 500 400])
end
title('Consensus Based Optimization')  


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


%% CBO Algorithm
for k = 1:1/dt
    t = k*dt;
    subplot(2,2,1)
    title(sprintf("CBO at time t=%d",t))
    fprintf("t=%d\n", t)
    
    % compute current consensus point v_alpha
    v_alpha = compute_valpha(E,alpha,V);
    
    % Brownian motion for exploration term
    dB = randn(d,N);
    
    % % particle iteration step (according to SDE)
    % consensus drift and exploration term
    V = V - lambda*(V-v_alpha*ones(1,N))*dt + sigma*abs(V-v_alpha*ones(1,N))*sqrt(dt).*dB;
    % gradient drift term
    h = 10^-3;
    gradE = zeros(d,N);
    for i = 1:d
        dV = h*zeros(d,N);
        dV(i,:) = ones(1,N);
        gradE = (E(V+h*dV)-E(V-h*dV))/(2*h);
    end
    V = V - gamma*learning_rate*gradE*dt;
    
    % plotting of particles
    if errorplots_meanwhile
        subplot(2,2,1)
    end
    V_plot = scatter(V, E(V), 20, "MarkerFaceColor", co(3,:), "MarkerEdgeColor", co(3,:));
    valpha_plot = plot(v_alpha, E(v_alpha), '.', 'MarkerSize', 20, "color", co(2,:));

    
    % % Computation of Error Metrics
    % Functional J
    J(k) = sum(interp1(convhull_x,convhull_y,V,'linear')-E(vstar))/N;
    J_a(k) = sum((interp1(convhull_x,convhull_y,V,'linear')-E(vstar)).*exp(-alpha*E(V)))/(sum(exp(-alpha*E(V))));
    J_a2(k) = sum((interp1(convhull_x,convhull_y,V,'linear')-E(vstar)).*exp(-alpha*E(V)))/N;
    J_E(k) = sum(E(V)-E(vstar))/N;
    
    % Variance
    Expectation = sum(V)/N;
    if show_expectation
        Expectation_plot = plot(Expectation, E(Expectation), '.', 'MarkerSize', 20, "color", 0.9*co(3,:));
        delete(valpha_plot)
        valpha_plot = plot(v_alpha, E(v_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
    end
    Variance(k) = sum((V-Expectation).^2)/N;
    VarianceStar(k) = sum((V-vstar).^2)/N;
    
    % normalization (for plotting convenience)
    if k==1
        normal_J = J(1);
        normal_J_a = J_a(1);
        normal_J_a2 = J_a2(1);
        normal_J_E = J_E(1);
        normal_V = Variance(1);
        normal_Vstar = VarianceStar(1);
    end
    J(k) = J(k)/normal_J;
    J_a(k) = J_a(k)/normal_J_a;
    J_a2(k) = J_a2(k)/normal_J_a2;
    J_E(k) = J_E(k)/normal_J_E;
    Variance(k) = Variance(k)/normal_V;
    VarianceStar(k) = VarianceStar(k)/normal_Vstar;
    
    volDdelta(k) = sum((E(V)-E(vstar))<delta)/N;
    
    % plotting of error metrics (meanwhile)
    if errorplots_meanwhile
        subplot(2,2,2)
        plot_errormetric(Metrics,dt,J,J_a,J_a2,J_E,Variance,VarianceStar)
    end
    
    % plotting of volume of D(delta)
    if errorplots_meanwhile
        subplot(2,2,4)
        plot_volDdelta(dt,volDdelta)
    end

    pause(dt)
    if manual_steps
        pause()
    end
    
    % plotting
    delete(valpha_plot)
    if show_expectation
        delete(Expectation_plot)
    end
    if t~=1
        delete(V_plot)
        if t~=-1
            delete(V_plot)
        end
    else
        subplot(2,2,1)
        valpha_plot_final = plot(v_alpha, E(v_alpha), '.', 'MarkerSize', 20, "color", co(2,:));
    end
    
end
v_alpha = compute_valpha(E,alpha,V);
fprintf("global minimizer (numerically): %d\n", vstar)
fprintf("final consensus point         : %d\n", v_alpha)


% plotting of error metrics (afterwards)
if ~errorplots_meanwhile
    subplot(2,2,1)
    %figure('Position', [1700 800 500 400])
    plot_errormetric(Metrics,dt,J,J_a,J_a2,J_E,Variance,VarianceStar)
    subplot(2,2,4)
    plot_volDdelta(dt,volDdelta)
end

if plotforlatexexport
    disp('Not availble yet.')
    %matlab2tikz('myfile.tex');
end


%% error metric plotting routine
function plot_errormetric(Metrics,dt,J,J_a,J_a2,J_E,Variance,VarianceStar)
co = set_color();
xlim([0,1])
ylim([0,2])
xlabel('$t$')
ylabel('normalized error')
if or(Metrics=="J",Metrics=="all")
    plot(dt:dt:1,J, "color", co(1,:), 'LineWidth', 2, 'LineStyle', '--')
    hold on;
    plot(dt:dt:1,J_a, "color", co(3,:), 'LineWidth', 2, 'LineStyle', '-.')
    hold on;
    %plot(dt:dt:1,J_a2, "color", co(2,:), 'LineWidth', 2, 'LineStyle', ':')
    %hold on;
    plot(dt:dt:1,J_E, "color", co(4,:), 'LineWidth', 2, 'LineStyle', ':')
end
if Metrics=="all"
    hold on;
end
if or(Metrics=="V",Metrics=="all")
    plot(dt:dt:1,Variance, "color", co(6,:), 'LineWidth', 2)
    hold on;
    plot(dt:dt:1,VarianceStar, "color", co(7,:), 'LineWidth', 2)    
end

if Metrics=="J"
    %legend('$\mathcal{J}=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$','$\mathcal{J}^\alpha=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_{\alpha,t}(v)$','$\widetilde{\mathcal{J}}^\alpha=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\omega_{\mathcal{E}}^\alpha(v)\,\mathrm{d}\rho_{t}(v)$','$\mathcal{J}_\mathcal{E}=\int(\mathcal{E}(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$')
    legend('$\mathcal{J}(\rho_t)=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$','$\mathcal{J}^\alpha(\rho_t)=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_{\alpha,t}(v)$','$\mathcal{J}_\mathcal{E}(\rho_t)=\int(\mathcal{E}(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$')
elseif Metrics=="V"
    legend('$\mathrm{Var}(\rho_t)$','$\mathrm{Var}^*(\rho_t)$')
else
    %legend('$\mathcal{J}=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$','$\mathcal{J}^\alpha=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_{\alpha,t}(v)$','$\widetilde{\mathcal{J}}^\alpha=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\omega_{\mathcal{E}}^\alpha(v)\,\mathrm{d}\rho_{t}(v)$','$\mathcal{J}_\mathcal{E}=\int(\mathcal{E}(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$','$V$','$V^*$')
    legend('$\mathcal{J}(\rho_t)=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$','$\mathcal{J}^\alpha(\rho_t)=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_{\alpha,t}(v)$','$\mathcal{J}_\mathcal{E}(\rho_t)=\int(\mathcal{E}(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$','$\mathrm{Var}(\rho_t)$','$\mathrm{Var}^*(\rho_t)$')
    %legend('$\mathcal{J}=\int(\mathcal{E}^c(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$','$\mathcal{J}_\mathcal{E}=\int(\mathcal{E}(v)-\underline{\mathcal{E}})\,\mathrm{d}\rho_t(v)$','$V$')
end
end

function plot_volDdelta(dt,volDdelta)
co = set_color();
subplot(2,2,4)
plot(dt:dt:1,volDdelta, "color", co(1,:), 'LineWidth', 2, 'LineStyle', '-')
xlabel('$t$')
xlim([0,1])
ylim([0,1])
legend('$\mathrm{Prob}_{\rho_t}D(\delta)$')
end


