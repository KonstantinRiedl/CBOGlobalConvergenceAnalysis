% Energy landscape detection
%
% This script uses the memory effects to eventually detect the local and 
% gloabl minimizer landscape of the objective function by clustering the
% positions the memory of each particle has seen over time.
%

%%
clear; clc; close all;

co = set_color();


%% Settings for Easy Handling and Notes
% 
% use pre-set CBOMemoryGradient setting (overrides manually chosen parameters)
pre_setparameters = 0;

% 3d plot
spatial_plot = 0;


%% Energy Function E

% % dimension of the ambient space
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'Rastrigin';
[E, grad_E, parametersE, parametersCBOmemorygradient, parametersInitialization] = objective_function(objectivefunction, d, 'CBO');

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
zrange_plot = parametersE(:,3)';
xrange = 100*xrange_plot;
yrange = 100*yrange_plot;

% global minimizer
xstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of CBOmemorygradient Algorithm

% time horizon
T = 40;

% discrete time size
dt = 0.1;
 
% number of particles
N = 200;

%
kappa = 1/dt;

% memory
memory = 1; % 0 or 1
% lambda1, sigma1, kappa and beta have no effect for memory=0.

% lambda1 (drift towards global and in-time best parameter)(drift towards in-time best parameter)
lambda1 = 1;
% lambda2 (drift towards in-time best parameter)
lambda2 = 0.4;
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
alpha = 10000;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta



%% Initialization
X0mean = 4*ones(d,1);
X0std = 4;


%% Use pre-set setting
if pre_setparameters==1
    T = parametersCBOmemorygradient('T');
    dt = parametersCBOmemorygradient('dt');
    N = parametersCBOmemorygradient('N');
    memory = parametersCBOmemorygradient('memory');
    lambda1 = parametersCBOmemorygradient('lambda1');
    lambda2 = parametersCBOmemorygradient('lambda2');
    anisotropic1 = parametersCBOmemorygradient('anisotropic1');
    sigma1 = parametersCBOmemorygradient('sigma1');
    anisotropic2 = parametersCBOmemorygradient('anisotropic2');
    sigma2 = parametersCBOmemorygradient('sigma2');
    gamma = parametersCBOmemorygradient('gamma');
    learning_rate = parametersCBOmemorygradient('learning_rate');
    alpha = parametersCBOmemorygradient('alpha');
    beta = parametersCBOmemorygradient('beta');
    X0mean = parametersInitialization('X0mean');
    X0std = parametersInitialization('X0std');
else
    parametersCBOmemorygradient = containers.Map({'T', 'dt', 'N', 'memory', 'kappa', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'gamma', 'learning_rate', 'alpha', 'beta'},...
                                   {  T,   dt,   N,   memory,   kappa,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   gamma,   learning_rate,   alpha,  beta});
    parametersInitialization = containers.Map({'X0mean', 'X0std'},...
                                              {  X0mean,   X0std});
end

%% Plotting

if d == 2
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
end


%% CBOmemorygradient Algorithm
%initialization
X0 = X0mean+X0std*randn(d,N);
X = X0;
Y = X;

Y_all = zeros(d,0);


k_clusters = 100;

% CBOmemorygradient
for k = 1:T/dt
    
    % % CBOmemorygradient iteration
    % compute current consensus point y_alpha
    y_alpha = compute_yalpha(E, alpha, Y);

    % position updates of one iteration of CBOmemorygradient
    [X, Y] = CBOmemorygradient_update(E, grad_E, parametersCBOmemorygradient, y_alpha, X, Y);
    
    if mod(k,1)==0
        Y_all = [Y_all, Y];
    end
    
end

kmeans_indices = kmeans(Y_all',k_clusters)';

minimizers = zeros(d,k_clusters);
for k = 1:k_clusters
    
    minimizers(:,k) = compute_yalpha(E, 10^3, Y_all(:, boolean(kmeans_indices==k)));
    
end

if d == 2
    scatter3(minimizers(1,:), minimizers(2,:), F(minimizers), 'filled')
end


