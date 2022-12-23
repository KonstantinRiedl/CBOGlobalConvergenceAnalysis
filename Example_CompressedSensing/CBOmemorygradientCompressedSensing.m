% CBOmemorygradient compressed sensing
%
% This script tests CBOmemorygradient numerically for a compressed sensing
% problem
%

%%
clear; clc; close all;

co = set_color();


%% Energy Function E

% % dimension of the ambient space
d = 200;
% % sparsity of signal
s = 8;

% % energy function E
% original signal / global minimizer
xstar = zeros(d,1);
xstar(randsample(d,s)) = sign(randn(s,1)).*(0.5+rand(s,1));

% number of measurements and noise-to-signal ratio
M = 5*s;
noise_to_signal_ratio = 0; % noise level in measurements (0 to 1)

% E (E is a function mapping columnwise from R^{d\times N} to R^N)
p = 1;
[E, grad_E, A, Y] = objective_function_compressed_sensing(xstar, M, p, noise_to_signal_ratio);


%% Parameters of CBOMemoryGradient Algorithm

% time horizon
T = 20;

% discrete time size
dt = 0.02;

% number of particles
N = 100;

% memory
memory = 0; % 0 or 1
% lambda2, sigma2, kappa and beta have no effect for memory=0.

% kappa
kappa = 1/dt;

% lambda1 (drift towards global and in-time best (consensus) parameter)
lambda1 = 1;
% lambda2 (drift towards in-time best parameter)
lambda2 = 0;
% gamma (parameter of gradient drift term)
gamma = 1;
learning_rate = 1;
% exploration/noise 1 type
anisotropic1 = 1;
% sigma (exploration/noise parameter 1)
sigma1 = sqrt(1.6);
% exploration/noise 2 type
anisotropic2 = 1;
% sigma (exploration/noise parameter 2)
sigma2 = lambda2*sigma1;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^2;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta


%% Initialization
X0mean = 0*ones(d,1);
X0std = 2;


parametersCBOmemorygradient = containers.Map({'T', 'dt', 'N', 'memory', 'kappa', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'gamma', 'learning_rate', 'alpha', 'beta'},...
                                             {  T,   dt,   N,   memory,   kappa,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   gamma,   learning_rate,   alpha,  beta});
parametersInitialization = containers.Map({'X0mean', 'X0std'},...
                                          {  X0mean,   X0std});


%% CBOMemoryGradient Algorithm
%initialization
X0 = X0mean + X0std*randn(d,N);

% CBOMemoryGradient
[xstar_app] = CBOmemorygradient(E, grad_E, parametersCBOmemorygradient, X0);

% post-processing
xstar_app(abs(xstar_app)<0.01) = 0; % hard threshold
A_support = A(:,xstar_app~=0);
xstar_app_support = (A_support'*A_support) \ A_support'*Y;
xstar_app(xstar_app~=0) = xstar_app_support;


% evaluate if recovery was successful
%norm(A*xstar_app - Y)/norm(xstar)
norm(xstar_app - xstar)/norm(xstar)
X = [xstar'; xstar_app'];
X(:,abs(xstar'+xstar_app')>0)


