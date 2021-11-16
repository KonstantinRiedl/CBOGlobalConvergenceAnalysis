% This function computes the total number of weights and biases of the NN
% as well as the number of weights and biases per layer.
% 
% 
% [nparam, nparam_layer, d_layer] = number_of_weightsbiases(NNtype, architecture, neurons, image_size)
%
% input:    NNtype        = fully_connected or CNN
%           architecture  = list of types of layers
%           neurons       = list of numbers of neurons in the layers of NN
%           image_size    = dimension of the image
%                           (a 28x28 image has image_size = [28, 28])
%           
% output:   nparam        = total number of parameters (weights and biases)
%           nparam_layer  = array of size (2, #layers) with number of
%                           weights and biases per layer
%           d_layer       = dimensions of outputs of layers
%                           (incluing input layer)
%                           for CNN: [#filters; 2d feature map size]
%

function [nparam, nparam_layer, d_layer] = number_of_weightsbiases(NNtype, architecture, neurons, image_size)

if size(architecture, 2) ~= size(neurons, 2)
    error('NN architecture error: the architecture differs from the distribution of neurons.')
end

if strcmp(NNtype, 'fully_connected')
    
    if size(neurons, 1) ~= 1
        error('NN neuron error: for fully_connected NN only dense layers are allowed.')
    end
    if min(architecture=='d') ~= 1
        error('NN architecture error: for fully_connected NN only dense layers are allowed.')
    end
    
    nparam_layer = zeros(2, size(neurons, 2));
    d_layer = zeros(1, 1+size(neurons, 2));
    current_d = image_size(1)*image_size(2);
    d_layer(1) = current_d;
    %disp(['at start      , initial dimension: ', num2str(current_d)]);
    neurons_plusinput = [current_d, neurons];
    
    layers = size(architecture, 2);
    for layer = 1:layers
        
        % number of weights in layer l
        nparam_layer(1, layer) = neurons_plusinput(layer)*neurons_plusinput(layer+1);
        % number of biases in layer l
        nparam_layer(2, layer) = neurons_plusinput(layer+1);
        
        current_d = neurons_plusinput(layer+1);
        d_layer(layer+1) = current_d;
        %disp(['after layer: ', num2str(layer), ', current dimension: ', num2str(current_d)]);
        
    end
    
    if current_d ~= 10
        error('NN architecture error: the final layer has to have 10 neurons.')
    end
    
    nparam = sum(sum(nparam_layer));
    
elseif strcmp(NNtype, 'CNN')
    
    if size(neurons, 1) ~= 2
        error('NN neuron error: for CNN features and kernel size has to be specified.')
    end
    
    nparam_layer = zeros(2, size(neurons, 2));
    d_layer = zeros(3, 1+size(neurons, 2));
    current_d = [1, image_size];
    d_layer(:, 1) = current_d;
    %disp(['at start      , initial dimension: ', num2str(current_d(1)), ' feature maps of size ', num2str(current_d(2:end))]);
    
    layers = size(architecture, 2);
    for layer = 1:layers
        
        if strcmp(architecture(layer), 'd')
            nparam_layer(:, layer) = neurons(1,layer)*[prod(current_d), 1];
            current_d = [neurons(1,layer), 1, 1];
        elseif strcmp(architecture(layer), 'c')
            nparam_layer(:, layer) = neurons(1,layer)*[neurons(2,layer)*neurons(2,layer); 1];
            current_d = [current_d(1)*neurons(1,layer), current_d(2:end) - neurons(2,layer) + 1];
        elseif strcmp(architecture(layer), 'p')
            current_d = [current_d(1), current_d(2:end)/2];
        else
            error('NN architecture error: layer type not known.')
        end
        d_layer(:, layer+1) = current_d;
        
        %disp(['after layer: ', num2str(layer), ', current dimension: ', num2str(current_d(1)), ' feature maps of size ', num2str(current_d(2:end))]);

    end
    
    if current_d(1) ~= 10
        error('NN architecture error: the final layer has to have 10 neurons.')
    end

    nparam = sum(sum(nparam_layer));
    
else
    error('NN architecture type not known.')
end

clear architecture neurons image_size neurons_plusinput layer

end