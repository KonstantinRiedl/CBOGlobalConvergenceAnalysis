% This function implements the forward pass of the neural network, i.e.,
% the flow of the data through the network.
% 
% 
% [z_out] = NN(x, data, image_size, NNtype, architecture, neurons)
% 
% input:    x             = array with different sets of parameters of size (d x N_batch)
%           data          = data as array of size (image_size(1)*image_size(2) x n_batch)
%           NNtype        = type of the neural network  (fully_connected or CNN)
%           architecture  = list of types of layers ('d': dense, 'c': convolutional, 'p': pooling)
%           neurons       = describtion of layers of NN
%           image_size    = dimension of the image
%                           (a 28x28 image has image_size = [28, 28])
%           alg_time      = optional flag 'testing_time' if NN is evaluated
%                           during testing phase
%           train_data    = testing data if alg_time=='testing_time'
%           
% output:   z_out         = likelihood per class for each parameter set and
%                           each data sample, i.e., of size (10 x n_batch x N_batch)
%

function [z_out] = NN(x, data, image_size, NNtype, architecture, neurons, alg_time, train_data)

N_batch = size(x,2);
n_batch = size(data,2);
[~, nparam_layer, d_layer] = number_of_weightsbiases(NNtype, architecture, neurons, image_size);
nparam_layer = reshape(nparam_layer, [1, size(nparam_layer,1)*size(nparam_layer,2)]);

layers = size(architecture, 2);

z_out = zeros(neurons(1,end), n_batch, N_batch);

% fixate the current particle and iterate over them
for i = 1:N_batch
    
    z = data;
    if nargin==8
        % evaluate NN at testing time
        if ~strcmp(alg_time, 'testing_time')
            error('NN input error.')
        end
        z_train = train_data;
    end
    if strcmp(NNtype, 'CNN')
        z = reshape(z, image_size(1), image_size(2), 1, 1, []);
        if nargin==8
            % evaluate NN at testing time
            z_train = reshape(z_train, image_size(1), image_size(2), 1, 1, []);
        end
    end
    
    for layer = 1:layers
        
        % retrieve weights or kernel and biases of current layer from x
        indices_weightskernel = sum(nparam_layer(1:2*layer-2)) + (1:nparam_layer(2*layer-1));
        indices_bias = sum(nparam_layer(1:2*layer-1)) + (1:nparam_layer(2*layer));
        
        if strcmp(architecture(layer), 'd')
            
        	weights = reshape(x(indices_weightskernel, i), neurons(1,layer), []);
            bias = x(indices_bias, i);
            
            z = reshape(z, [], n_batch);
            
            % W*z + biases
            z = weights*z + bias;
            
            % activation
            z = ReLU(z);
            
            % batch normalization
            if nargin==6
                % batch normalization at training time
                z = (z-mean(z,[2]))./sqrt(std(z,0,[2]).^2+10^-4);
            else
                % batch normalization at testing time
                z_train = reshape(z_train, [], n_batch);
                z_train = weights*z_train + bias;
                z_train = ReLU(z_train);
                z = (z-mean(z_train,[2]))./sqrt(std(z_train,0,[2]).^2+10^-4);
                z_train = (z_train-mean(z_train,[2]))./sqrt(std(z_train,0,[2]).^2+10^-4);
            end

        elseif strcmp(architecture(layer), 'c')

            kernels = reshape(x(indices_weightskernel, i), neurons(1,layer), neurons(2,layer), neurons(2,layer));
            bias = x(indices_bias, i);
            
            z_c = zeros([d_layer(2,layer+1),d_layer(3,layer+1),size(z,3),neurons(1,layer),n_batch]);
            for k = 1:size(kernels, 1)
                kernel = reshape(kernels(k,:,:), neurons(2,layer)*[1,1]);
                z_c(:,:,:,k,:) = convn(z, kernel, 'valid') + bias(k);
                z_c = ReLU(z_c);
            end

            z = reshape(z_c, [size(z_c,1),size(z_c,2),size(z_c,3)*size(z_c,4),1,size(z_c,5)]);
            
            % batch normalization
            if nargin==6
                % batch normalization at training time
                z = (z-mean(z,[1,2,5]))./sqrt(std(z,0,[1,2,5]).^2+10^-4);
            else
                % batch normalization at testing time
                z_c_train = zeros([d_layer(2,layer+1),d_layer(3,layer+1),size(z_train,3),neurons(1,layer),n_batch]);
                for k = 1:size(kernels, 1)
                    kernel = reshape(kernels(k,:,:), neurons(2,layer)*[1,1]);
                    z_c_train(:,:,:,k,:) = convn(z_train, kernel, 'valid') + bias(k);
                    z_c_train = ReLU(z_c_train);
                end

                z_train = reshape(z_c_train, [size(z_c_train,1),size(z_c_train,2),size(z_c_train,3)*size(z_c_train,4),1,size(z_c_train,5)]);
                
                z = (z-mean(z_train,[1,2,5]))./sqrt(std(z_train,0,[1,2,5]).^2+10^-4);
                z_train = (z_train-mean(z_train,[1,2,5]))./sqrt(std(z_train,0,[1,2,5]).^2+10^-4);
            end
            
        elseif strcmp(architecture(layer), 'p')

            pooling_kernel = [neurons(2,layer),neurons(2,layer)];
            z = sepblockfun(z, pooling_kernel, 'max');
            
            % batch normalization
            if nargin==6
                % batch normalization at training time
                z = (z-mean(z,[1,2,5]))./sqrt(std(z,0,[1,2,5]).^2+10^-4);
            else
                % batch normalization at testing time
                z_train = sepblockfun(z_train, pooling_kernel, 'max');
                z = (z-mean(z_train,[1,2,5]))./sqrt(std(z_train,0,[1,2,5]).^2+10^-4);
                z_train = (z_train-mean(z_train,[1,2,5]))./sqrt(std(z_train,0,[1,2,5]).^2+10^-4);
            end

        else
            error('NN architecture error: layer type not known.')
        end
    
    end
    
    z_out(:,:,i) = z;
    
end

z_out = softmax(z_out);

clear x data image_size neurons architecture
clear N_batch n_batch nparam_layer neurons_plusinput layers
clear z

end

function [z] = ReLU(z)

z = max(z,0);

end

function [z] = softmax(z)

exp_z = exp(z-max(z,[],1));
z = exp_z./sum(exp_z, 1);

clear exp_z

end