% CBOMemoryGradient numerical example
%
% This script tests CBOMemoryGradient numerically and outputs the 
% approximation to the global minimizer.
%

%%
clear; clc; close all;

co = set_color();


%% Settings for Easy Handling and Notes
%
% use pre-set CBOMemorygradient setting (overrides manually chosen parameters)
pre_setparameters = 0;


%% Energy Function E

% % dimension of the ambient space
d = 4;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'Rastrigin';
[E, grad_E, parametersE, parametersCBOmemorygradient, parametersInitialization] = objective_function(objectivefunction, d, 'CBO');

% global minimizer
xstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of CBOMemoryGradient Algorithm

% time horizon
T = 20;

% discrete time size
dt = 0.02;

% number of particles
N = 100;

% memory
memory = 1; % 0 or 1
% lambda2, sigma2, kappa and beta have no effect for memory=0.

%
kappa = 1/dt;

% lambda1 (drift towards global and in-time best (consensus) parameter)
lambda1 = 1;
% lambda2 (drift towards in-time best parameter)
lambda2 = 0;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% exploration/noise 1 type
anisotropic1 = 1;
% sigma (exploration/noise parameter 1)
sigma1 = sqrt(1.6);
% exploration/noise 2 type
anisotropic2 = 1;
% sigma (exploration/noise parameter 2)
sigma2 = lambda2*sigma1;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 1000;
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


%% CBOMemoryGradient Algorithm
%initialization
X0 = X0mean+X0std*randn(d,N);

% CBOmemoryGradient
[xstar_app] = CBOmemorygradient(E, grad_E, parametersCBOmemorygradient, X0);

fmtxstar     = ['global minimizer (numerically): [', repmat('%g, ', 1, numel(xstar)-1), '%g]\n'];
fprintf(fmtxstar, xstar)
fprintf('          with objective value: %f\n', E(xstar))

fmtvstar_app = ['final approximated minimizer  : [', repmat('%g, ', 1, numel(xstar_app)-1), '%g]\n'];
fprintf(fmtvstar_app, xstar_app)
fprintf('          with objective value: %f\n', E(xstar_app))
if E(xstar_app)<0.8 
    fprintf('************** CBO   successful **************\n')
else
    fprintf('************** CBO UNsuccessful **************\n')
end

