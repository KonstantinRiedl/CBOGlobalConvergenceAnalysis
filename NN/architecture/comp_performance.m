% This function computes and displays the current performance of the NN 
% on a given sample (data, label).
% 
% 
% [objective_value, train_accuracy, test_accuracy] = comp_performance(E, V, train_data, test_data, image_size, train_label, test_label, NNtype, architecture, neurons)
% 
% input:    E               = objective function E (as anonymous function)
%           V               = array with different sets of parameters of size (d x N_batch)
%           train_data      = training data as array of size (image_size(1)*image_size(2) x n_batch_train)
%           test_data       = test data as array of size (image_size(1)*image_size(2) x n_batch_test)
%           image_size      = dimension of the image
%                             (a 28x28 image has image_size = [28, 28])
%           train_label     = training labels as array of size (1 x n_batch_train)
%           test_label      = test labels as array of size (1 x n_batch_test)
%           NNtype          = type of the neural network  (fully_connected or CNN)
%           architecture    = list of types of layers ('d': dense, 'c': convolutional, 'p': pooling)
%           neurons         = describtion of layers of NN
%           alg_state       = dictionary with number of current and total
%                             batch(es) and epoch(s)
%           verbose         = whether performance measures should be
%                             displayed
%           worker          = number of worker (for HPC testing)
%           
% output:   train_accuracy  = accuracy on training set
%           test_accuracy   = accuracy on test set
%           objective_value = value of objective function
%

function [train_accuracy, test_accuracy, objective_value] = comp_performance(E, V, train_data, test_data, image_size, train_label, test_label, NNtype, architecture, neurons, alg_state, verbose, worker)

E_train = @(x) E(x, train_data, train_label);
v_alpha = compute_yalpha(E_train, 10^15, V);

objective_value = E_train(v_alpha);
train_accuracy = eval_accuracy(v_alpha, train_data, image_size, train_label, NNtype, architecture, neurons);
test_accuracy  = eval_accuracy(v_alpha, test_data, image_size, test_label, NNtype, architecture, neurons, 'testing_time', train_data);

% displaying performance
if verbose
    disp_epoch = ['Epoch ', sprintf('%03d', alg_state('epoch')), ' of ', sprintf('%03d', alg_state('epochs')), '; '];
    disp_batch = ['Batch ', sprintf('%04d', alg_state('batch')), ' of ', sprintf('%04d', alg_state('training_batches_per_epoch')), ': '];
    disp_obj_val = ['Objective value: ', sprintf('%01.4f', objective_value), '; '];
    disp_train_acc = ['Training accuracy: ', sprintf('%01.4f', train_accuracy), '; '];
    disp_test_acc  = ['Test accuracy: ', sprintf('%01.4f', test_accuracy), '.'];
    
    disp_alg_state = [disp_epoch];
    if nargin==13
        disp_worker = ['Worker ', sprintf('%02d', worker)];
        disp_alg_state = [disp_worker, '. ', disp_alg_state];
    end
    if alg_state('batch')>0
        disp_alg_state = [disp_alg_state, disp_batch];
    else
        disp_alg_state = [disp_alg_state, '##################: '];
    end
      
    disp_performance = [disp_obj_val, disp_train_acc, disp_test_acc];
    
    disp([disp_alg_state, disp_performance])

end

end