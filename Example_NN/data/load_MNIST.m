% This function loads the MNIST dataset and optionally scales the images.
% 
% 
% [train_data, train_label, test_data, test_label, image_size] = load_MNIST(image_size)
% 
% input:    image_size    = desired dimension of the images
%                           (the unresized size is 28x28)
%           
% output:   train_data    = training data as array of size (image_size, number of training samples)
%           train_label   = training labels as array of size (1, number of training samples)
%           test_data     = test data as array of size (image_size, number of test samples)
%           test_label    = test labels as array of size (1, number of test samples)
%

function [train_data, train_label, test_data, test_label] = load_MNIST(image_size)

mnist_train = table2array(readtable('mnist_train.csv'));
train_data  = mnist_train(:,2:end)'./255.;
train_label = mnist_train(:,1);
train_label = train_label';

mnist_test = table2array(readtable('mnist_test.csv'));
test_data = mnist_test(:,2:end)'./255.;
test_label = mnist_test(:,1);
test_label = test_label';


if nargin==0
    image_size = [28, 28];
end

% perform a permutation on the complete dataset
data = [train_data, test_data];
labels = [train_label, test_label];

permutation = randperm(length(labels));
data = data(:,permutation);
labels = labels(permutation);

train_data = data(:,1:length(train_label));
test_data = data(:,length(train_label)+1:end);
train_label = labels(1:length(train_label));
test_label = labels(length(train_label)+1:end);

if image_size(1) ~= 28 || image_size(2) ~= 28
    train_data = reshape(train_data, [28,28,size(train_data,2)]);
    test_data = reshape(test_data, [28,28,size(test_data,2)]);

    % resize all images
    train_data = imresize(train_data, image_size);
    test_data = imresize(test_data, image_size);

    train_data = reshape(train_data, [image_size(1)*image_size(2), size(train_data,3)]);
    test_data = reshape(test_data, [image_size(1)*image_size(2), size(test_data,3)]);
end

clear mnist_train mnist_test
clear permutation data labels image_size

end

% imshow(permute(reshape(train_data(:,1),image_size),[2,1]))