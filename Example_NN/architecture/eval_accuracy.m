% This function evaluates the accuracy of a set of parameters of the NN
% on a given sample (data, label).
% 
% 
% accuracy = eval_accuracy(x, data, image_size, label, NNtype, architecture, neurons)
% 
% input:    x             = array with different sets of parameters of size (d x N_batch)
%           data          = data as array of size (image_size(1)*image_size(2) x n_batch)
%           image_size    = dimension of the image
%                           (a 28x28 image has image_size = [28, 28])
%           label         = labels as array of size (1 x n_batch)
%           NNtype        = type of the neural network  (fully_connected or CNN)
%           architecture  = list of types of layers ('d': dense, 'c': convolutional, 'p': pooling)
%           neurons       = describtion of layers of NN
%           alg_time      = optional flag 'testing_time' if NN is evaluated
%                           during testing phase
%           train_data    = testing data if alg_time=='testing_time'
%           
% output:   accuracy      = accuracy for each parameter set,
%                           i.e., of size (1 x 1 x N_batch)
%

function [accuracy] = eval_accuracy(x, data, image_size, label, NNtype, architecture, neurons, alg_time, train_data)

if nargin==7
    % evaluate accuracy at training time
    z_out = NN(x, data, image_size, NNtype, architecture, neurons);
else
    % evaluate accuracy at testing time
    if ~strcmp(alg_time, 'testing_time')
        error('eval_accuracy input error.')
    end
    z_out = NN(x, data, image_size, NNtype, architecture, neurons, 'testing_time', train_data);
end
[~, results] = max(z_out, [], 1);
results = results-1;

accuracy = sum(results==label,2)/size(label,2);

clear x data image_size label neurons architecture results

end