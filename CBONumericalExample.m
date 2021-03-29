% CBO Numerical Example
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
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
% lopsided W-shaped function in 1d
objectivefunction = 'Rastrigin';
[E, parametersE, parametersCBO, parametersInitialization] = objective_function(objectivefunction,d);

% global minimizer
vstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Parameters of CBO Algorithm

% time horizon
T = 4;

% discrete time size
dt = 0.02;
 
% number of particles
N = 100;

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
V0mean = [4;4];
V0std = 8;


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


%% CBO Algorithm
%initialization
V0 = V0mean+V0std*randn(d,N);
V = V0;
% CBO
[vstar_app] = CBO(E,parametersCBO,V0);

fprintf("global minimizer (numerically): [%d;%d]\n", vstar)
fprintf("final approximated minimizer  : [%d;%d]\n", vstar_app)



