% CBO numerical example
%
% This script tests CBO numerically and outputs the approximation to the
% global minimizer.
%

%%
clear; clc; close all;

co = set_color();


%% Settings for Easy Handling and Notes
%
% use pre-set CBO setting (overrides manually chosen parameters)
pre_setparameters = 0;


%% Energy Function E

% % dimension of the ambient space
d = 4;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'Rastrigin';
[E, grad_E, parametersE, parametersCBO, parametersInitialization] = objective_function(objectivefunction, d, 'CBO');

% global minimizer
vstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of CBO Algorithm

% time horizon
T = 10;

% discrete time size
dt = 0.01;
 
% number of particles
N = 100;

% lambda (parameter of consensus drift term)
lambda = 1;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% type of diffusion
anisotropic = 1;
% sigma (parameter of exploration term)
sigma = sqrt(1.6);

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 1000;


%% Initialization
V0mean = 4*ones(d,1);
V0std = 4;


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


%% CBO Algorithm
%initialization
V0 = V0mean+V0std*randn(d,N);
V = V0;

% CBO
[vstar_app] = CBO(E, grad_E, parametersCBO, V0);

fmtvstar     = ['global minimizer (numerically): [', repmat('%g, ', 1, numel(vstar)-1), '%g]\n'];
fprintf(fmtvstar, vstar)
fprintf('          with objective value: %f\n', E(vstar))

fmtvstar_app = ['final approximated minimizer  : [', repmat('%g, ', 1, numel(vstar_app)-1), '%g]\n'];
fprintf(fmtvstar_app, vstar_app)
fprintf('          with objective value: %f\n', E(vstar_app))
if E(vstar_app)<0.8 
    fprintf('************** CBO   successful **************\n')
else
    fprintf('************** CBO UNsuccessful **************\n')
end

