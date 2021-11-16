% MNIST Classification with CBO
%
% This script implements the framework for training a neural network 
% classifier (shallow NN, deep NN, CNN) for MNIST classification using CBO.
%

%%
clear; clc; close all;

co = set_color();


%% MNIST Dataset

image_size = [28, 28]; % the standard size of MNIST is [28, 28]

[train_data, train_label, test_data, test_label] = load_MNIST(image_size);


%% Neural Network Architecture

% type of the neural network
NNtype = 'fully_connected'; % fully_connected or CNN

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


%% Parameters of CBO Algorithm

% number of epochs
epochs = 4;

% discrete time size
dt = 0.1;
 
% number of particles
N = 100;
% particle reduction strategy (for N)
particle_reduction = 0;


% % CBO parameters
% lambda (parameter of consensus drift term)
lambda = 1;

% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;

% type of diffusion
anisotropic = 1;
% sigma (parameter of exploration term)
sigma = sqrt(0.4);
% parameter cooling strategy (for sigma and alpha)
parameter_cooling = 1;

% alpha
alpha = 50;


parametersCBO = containers.Map({'epochs', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma', 'particle_reduction', 'parameter_cooling'},...
                               {  epochs,   dt,   N,   alpha,   lambda,   gamma,   learning_rate,   anisotropic,   sigma,   particle_reduction,   parameter_cooling});


% mini batch size in the number of particles
batch_size_N = 10; % ensure that batch_size_E divides N

% batch size used for the evaluation of the objective
batch_size_E = 60; % ensure that batch_size_N divides size(train_data,2)

% full or partial update
full_or_partial_V_update = 'full';

parametersbatch = containers.Map({'batch_size_N', 'batch_size_E', 'full_or_partial_V_update'},...
                                 {  batch_size_N,   batch_size_E,   full_or_partial_V_update});


%% Initialization

%initialization
V0mean = zeros(d,1);
V0std = 1;

parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                          {  V0mean,   V0std});


%% CBO Algorithm

% CBO
[vstar_app, performance_tracking] = CBOmachinelearning(parametersCBO, parametersbatch, parametersInitialization, train_data, train_label, test_data, test_label, image_size, NN_architecture);


