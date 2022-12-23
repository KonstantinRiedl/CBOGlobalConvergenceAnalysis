% MNIST Classification with CBOMemoryGradient
%
% This script implements the framework for training a neural network 
% classifier (shallow NN, deep NN, CNN) for MNIST classification using CBO
% with memory effects.
%

%%
clear; clc; close all;

co = set_color();


%% MNIST Dataset

image_size = [28, 28]; % the standard size of MNIST is [28, 28]

[train_data, train_label, test_data, test_label] = load_MNIST(image_size);


%% Neural Network Architecture

% type of the neural network
NNtype = 'CNN'; % fully_connected or CNN

% architecture of the neural network
if strcmp(NNtype, 'fully_connected')
    architecture = ['d'];
    neurons =      [ 10];
    %architecture = ['d', 'd', 'd', 'd'];
    %neurons =      [ 20,  10,  10,  10];
elseif strcmp(NNtype, 'CNN')
    architecture = ['c', 'p', 'c', 'p', 'd'];
    neurons =      [  4,   0,   3,   0,  10 ; % #filters (for 'c') and #weights (for 'd')
                      5,   2    5,   2,   0]; % size of kernel for 'c' and 'p'
else
    error('NN architecture type not known.')
end

% % parameter dimension aka dimension of the optimization space
[d, ~, ~] = number_of_weightsbiases(NNtype, architecture, neurons, image_size);

NN_architecture = containers.Map({'NNtype', 'architecture', 'neurons', 'd'},...
                                 {  NNtype,   architecture,   neurons,   d});


%% Parameters of CBOMemoryGradient Algorithm

% number of epochs
epochs = 100;

% discrete time size
dt = 0.1;
 
% number of particles
N = 100;
% particle reduction strategy (for N)
particle_reduction = 0;

% memory
memory = 0; % 0 or 1
% lambda1, sigma1, kappa and beta have no effect for memory=0.

%
kappa = 1/dt;

% % CBOMemoryGradient parameters
% lambda1 (drift towards in-time best parameter)
lambda1 = 0;
% lambda2 (drift towards global and in-time best (consensus) parameter)
lambda2 = 1;

% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;

% exploration/noise 1 type
anisotropic1 = 1;
% sigma (exploration/noise parameter 1)
sigma1 = lambda1*sqrt(0.4);
% exploration/noise 2 type
anisotropic2 = 1;
% sigma (exploration/noise parameter 2)
sigma2 = sqrt(0.4);
% parameter cooling strategy (for sigma and alpha)
parameter_cooling = 1;

% alpha
alpha = 50;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta


parametersCBOmemorygradient = containers.Map({'epochs', 'dt', 'N', 'memory', 'kappa', 'alpha', 'beta', 'lambda1', 'lambda2', 'gamma', 'learning_rate', 'anisotropic1', 'anisotropic2', 'sigma1', 'sigma2', 'particle_reduction', 'parameter_cooling'},...
                                             {  epochs,   dt,   N,   memory,   kappa,   alpha,   beta,   lambda1,   lambda2,   gamma,   learning_rate,   anisotropic1,   anisotropic2,   sigma1,   sigma2,   particle_reduction,   parameter_cooling});


% mini batch size in the number of particles
batch_size_N = 100; % ensure that batch_size_E divides N

% batch size used for the evaluation of the objective
batch_size_E = 60; % ensure that batch_size_N divides size(train_data,2)

% full or partial update
full_or_partial_XY_update = 'full';

parametersbatch = containers.Map({'batch_size_N', 'batch_size_E', 'full_or_partial_XY_update'},...
                                 {  batch_size_N,   batch_size_E,   full_or_partial_XY_update});


%% Initialization

X0mean = zeros(d,1);
X0std = 1;

parametersInitialization = containers.Map({'X0mean', 'X0std'},...
                                          {  X0mean,   X0std});


%% CBOmemorygradient Algorithm

% CBOmemorygradient
[xstar_app, performance_tracking] = CBOmemorygradientmachinelearning(parametersCBOmemorygradient, parametersbatch, parametersInitialization, train_data, train_label, test_data, test_label, image_size, NN_architecture);


