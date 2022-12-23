% Consensus-based optimization with memory for Machine Learning (CBOmemorygradient)
%
% This function performs PSO for machine learning applications.
% 
% 
% [xstar_approx, performance_tracking] = CBOmemorygradientmachinelearning(parametersCBOmemorygradient, parametersbatch, parametersInitialization, train_data, train_label, test_data, test_label, image_size, NN_architecture, worker)
% 
% input:    E                                = objective function E (as anonymous function)
%           parametersCBOmemorygradient      = suitable parameters for PSO
%                                            = [epochs, dt, N, memory, kappa, lambda1, lambda2, gamma, learning_rate, anisotropic1, sigma1, anisotropic2, sigma2, alpha, beta]
%               - epochs                     = number of epochs
%               - dt                         = time step size
%               - N                          = number of particles
%               - kappa                      = scaling parameter (usually 1/dt)
%               - gamma*learning_rate        = gradient drift coefficient
%               - lambda1                    = drift towards in-time best parameter
%               - lambda2                    = drift towards global and in-time best parameter
%               - anisotropic1               = noise 1 type
%               - sigma1                     = noise parameter 1
%               - anisotropic2               = noise 2 type
%               - sigma2                     = noise parameter 2
%               - alpha                      = weight/temperature parameter
%               - beta                       = regularization parameter for sigmoid
%           parametersbatch                  = suitable batch sizes for PSO
%                                            = [batch_size_N, batch_size_E, full_or_partial_XY_update]
%           parametersInitialization         = suitable initializaiton for PSO
%                                            = [X0mean, X0std, V0mean, V0std]
%           train_data, train_label, test_data, test_label, image_size = training and testing data
%           NN_architecture                  = architecture of NN
%           X0                               = initial positions of the particles
%           Y0                               = initial best positions of the particles
%           
% output:   xstar_approx                     = approximation to xstar
%           performance_tracking             = saved performance metrics of training porcess
%

function [xstar_approx, performance_tracking] = CBOmemorygradientmachinelearning(parametersCBOmemorygradient, parametersbatch, parametersInitialization, train_data, train_label, test_data, test_label, image_size, NN_architecture, worker)

% get parameters
epochs = parametersCBOmemorygradient('epochs');
sigma1_initial = parametersCBOmemorygradient('sigma1');
sigma2_initial = parametersCBOmemorygradient('sigma2');
alpha_initial = parametersCBOmemorygradient('alpha');
batch_size_N = parametersbatch('batch_size_N');
batch_size_E = parametersbatch('batch_size_E');
full_or_partial_XY_update = parametersbatch('full_or_partial_XY_update');

NNtype = NN_architecture('NNtype');
architecture = NN_architecture('architecture');
neurons = NN_architecture('neurons');
d = NN_architecture('d');


% initialization
X0 = parametersInitialization('X0mean') + parametersInitialization('X0std')*randn(d,parametersCBOmemorygradient('N'));
X = X0;
Y = X;
y_alpha_on_batch_old = zeros(d,1);

% % definition of the risk (objective function E)
% predicted and true label
predicted_label = @(z, data) NN(z, data, image_size, NNtype, architecture, neurons);
true_label_1hot = @(z, label) reshape(repmat((label==(0:9)'),[1, size(z,2)]), [neurons(1,end), size(label,2), size(z,2)]);
% categorical crossentropy loss function
categ_CE_loss = @(z, data, label) -1/neurons(1,end)*reshape(sum(true_label_1hot(z, label).*log(predicted_label(z, data)), 1), [size(data, 2), size(z, 2)]);
% risk (objective function E)
E = @(z, data, label) 1/size(data, 2)*sum(categ_CE_loss(z, data, label),1);


% performance tracking during training
training_batches_per_epoch = size(train_data,2)/batch_size_E;
recording_sample_size = 10000;
recording_frequency = 500; % ensure that recording_frequency divides training_batches_per_epoch
performance_tracking = NaN(3, epochs+1, training_batches_per_epoch/recording_frequency);


% compute, display and save initial performance
rand_indices_train = randsample(size(train_data,2),recording_sample_size)';
rand_indices_test = randsample(size(test_data,2),recording_sample_size)';
alg_state = containers.Map({'epoch', 'epochs', 'batch', 'training_batches_per_epoch'},...
                           {      0,   epochs,       0,                            0});
if nargin==10
    [train_accuracy, test_accuracy, objective_value] = comp_performance(E, Y, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1, worker);
else
    [train_accuracy, test_accuracy, objective_value] = comp_performance(E, Y, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1);
end
performance_tracking(1,1,end) = train_accuracy;
performance_tracking(2,1,end) = test_accuracy;
performance_tracking(3,1,end) = objective_value;

% % CBOmemorygradient
for epoch = 1:epochs
    
    % employ optional particle reduction strategy
    particle_reduction = parametersCBOmemorygradient('particle_reduction');
    if particle_reduction
        varianceY_before = norm(var(Y,0,2));
    end
    
    for batch = 1:training_batches_per_epoch
        
        % % definition of objective function E on current training batch
        % indices of current training batch
        indices_t_b = (batch-1)*batch_size_E+(1:batch_size_E);
        batch_data = train_data(:,indices_t_b);
        batch_label = train_label(indices_t_b);
        % objective function E on current training batch
        E_train_batch = @(x) E(x, batch_data, batch_label);
        objective_function_Y = E_train_batch(Y);
        
        % % update of particles' positions 
        particle_batches = parametersCBOmemorygradient('N')/batch_size_N;
        permutation = randperm(parametersCBOmemorygradient('N'));
        X = X(:,permutation);
        Y = Y(:,permutation);
        objective_function_Y = objective_function_Y(:,permutation);
        for particle_batch = 1:particle_batches
            
            % indices of current particle batch
            indices_p_b = (particle_batch-1)*batch_size_N+(1:batch_size_N);
            X_particle_batch = X(:,indices_p_b);
            Y_particle_batch = Y(:,indices_p_b);
            objective_function_Y_batch = objective_function_Y(:,indices_p_b);
            
            
            % % CBOmemorygradient iteration
            % compute current consensus point y_alpha
            y_alpha_on_batch = compute_yalpha(E_train_batch, parametersCBOmemorygradient('alpha'), Y_particle_batch, objective_function_Y_batch);

            % position updates of one iteration of CBOmemorygradient
            if strcmp(full_or_partial_XY_update, 'partial')
                [X_particle_batch, Y_particle_batch, objective_function_Y_batch] = CBOmemorygradient_update(E_train_batch, @(x) nan, parametersCBOmemorygradient, y_alpha_on_batch, X_particle_batch, Y_particle_batch, objective_function_Y_batch);
                X(:,indices_p_b) = X_particle_batch;
                Y(:,indices_p_b) = Y_particle_batch;
                objective_function_Y(:,indices_p_b) = objective_function_Y_batch;
            elseif strcmp(full_or_partial_XY_update, 'full')
                [X, Y, objective_function_Y] = CBOmemorygradient_update(E_train_batch, @(x) nan, parametersCBOmemorygradient, y_alpha_on_batch, X, Y, objective_function_Y);
            else
                error('full_or_partial_XY_update type not known.')
            end
            
            % random Brownian motion if y_alpha_on_batch is not changing
            if norm(y_alpha_on_batch_old-y_alpha_on_batch, 'inf') < 10^-5
                dB = randn(d,parametersCBOmemorygradient('N'));
                sigma = max(parametersCBOmemorygradient('sigma1'), parametersCBOmemorygradient('sigma2'));
                dt = parametersCBOmemorygradient('dt');
                X = X + sigma*sqrt(dt).*dB;
                clear dB dt
            end
            y_alpha_on_batch_old = y_alpha_on_batch;
            clear valpha_on_batch
            
        end
        
        % compute, display and save current performance
        if mod(batch,recording_frequency)==0
            rand_indices_train = randsample(size(train_data,2),recording_sample_size)';
            rand_indices_test = randsample(size(test_data,2),recording_sample_size)';
            alg_state = containers.Map({'epoch', 'epochs', 'batch', 'training_batches_per_epoch'},...
                                       {  epoch,   epochs,   batch,  training_batches_per_epoch});
            if nargin==10
                [train_accuracy, test_accuracy, objective_value] = comp_performance(E, Y, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1, worker);
            else
                [train_accuracy, test_accuracy, objective_value] = comp_performance(E, Y, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1);
            end
            performance_tracking(1,epoch+1,batch/recording_frequency) = train_accuracy;
            performance_tracking(2,epoch+1,batch/recording_frequency) = test_accuracy;
            performance_tracking(3,epoch+1,batch/recording_frequency) = objective_value;
            
            clear E_train rand_indices_train rand_indices_test
        end
        
    end
    
    % employ optional particle reduction strategy (PRS)
    speedup_PRS = 0.2;
    if particle_reduction
        varianceV = norm(var(Y,0,2));
        reduction_factor = min(max(1-speedup_PRS,(speedup_PRS*varianceV+(1-speedup_PRS)*varianceY_before)/varianceY_before),1);
        if reduction_factor<1
            parametersCBOmemorygradient('N') = ceil(parametersCBOmemorygradient('N')/batch_size_N*reduction_factor)*batch_size_N;
            X = X(:,randsample(size(X,2), parametersCBOmemorygradient('N'))');
            Y = Y(:,randsample(size(Y,2), parametersCBOmemorygradient('N'))');
        end
    end
    
    
    % employ optional and cooling strategy (CS) 
    parameter_cooling = parametersCBOmemorygradient('parameter_cooling');
    if parameter_cooling
        parametersCBOmemorygradient('sigma1') = parametersCBOmemorygradient('sigma1')*log2(epoch+1)/log2(epoch+2);
        parametersCBOmemorygradient('sigma2') = parametersCBOmemorygradient('sigma2')*log2(epoch+1)/log2(epoch+2);
        parametersCBOmemorygradient('alpha') = parametersCBOmemorygradient('alpha')*2;
    end
    
    
    % saving results and parameters
    if strcmp(NNtype, 'fully_connected') && length(architecture)==1
        NNtype_save = 'ShallowNN';
    elseif strcmp(NNtype, 'fully_connected') && length(architecture)>1
        NNtype_save = 'DeepNN';
    elseif strcmp(NNtype, 'CNN')
        NNtype_save = 'CNN';
    else
        error('NNtype does not exist')
    end
    if nargin==10
        filename = ['CBOandPSO/NN/results/CBOmemorygradient/', NNtype_save, '/', 'CBOMNIST_worker', num2str(worker), '_memory_', num2str(parametersgradient('memory')), '_N_', num2str(parametersCBOmemorygradient('N')), '_nN_', num2str(batch_size_N), '_lambda1_', num2str(100*round(parametersCBOmemorygradient('lambda1'),2)),'e-2', '_sigma2**2_', num2str(1000*round(sigma2_initial^2,3)),'e-3', '_sigma1**2_', num2str(1000*round(sigma1_initial^2,3)),'e-3', '_alpha_', num2str(alpha_initial), '_beta_', num2str(parametersCBOmemorygradient('beta')), '_parametercooling_', num2str(parameter_cooling), '_', num2str(epochs), 'epochs', '_preliminary'];
    else
        filename = ['CBOandPSO/NN/results/CBOmemorygradient/', NNtype_save, '/', 'CBOMNIST', '_memory_', num2str(parametersCBOmemorygradient('memory')), '_N_', num2str(parametersCBOmemorygradient('N')), '_nN_', num2str(batch_size_N), '_lambda1_', num2str(100*round(parametersCBOmemorygradient('lambda1'),2)),'e-2', '_sigma2**2_', num2str(1000*round(sigma2_initial^2,3)),'e-3', '_sigma1**2_', num2str(1000*round(sigma1_initial^2,3)),'e-3', '_alpha_', num2str(alpha_initial), '_beta_', num2str(parametersCBOmemorygradient('beta')), '_parametercooling_', num2str(parameter_cooling), '_', num2str(epochs), 'epochs', '_preliminary'];
    end
    parsave_CBOmemorygradient(filename, y_alpha_on_batch_old, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, parametersCBOmemorygradient('dt'), parametersCBOmemorygradient('N'), parametersCBOmemorygradient('memory'), parametersCBOmemorygradient('kappa'), parametersCBOmemorygradient('alpha'), parametersCBOmemorygradient('beta'), parametersCBOmemorygradient('lambda1'), parametersCBOmemorygradient('lambda2'), parametersCBOmemorygradient('gamma'), parametersCBOmemorygradient('learning_rate'), parametersCBOmemorygradient('anisotropic1'), parametersCBOmemorygradient('sigma1'), parametersCBOmemorygradient('anisotropic2'), parametersCBOmemorygradient('sigma2'), particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_XY_update, parametersInitialization('X0mean'), parametersInitialization('X0std'))
    
end
  
xstar_approx = nan;
%E_train = @(x) E(x, train_data, train_label);
%vstar_approx = compute_valpha(E_train, parametersCBO('alpha'), V);

% saving final results and parameters
if strcmp(NNtype, 'fully_connected') && length(architecture)==1
    NNtype_save = 'ShallowNN';
elseif strcmp(NNtype, 'fully_connected') && length(architecture)>1
    NNtype_save = 'DeepNN';
elseif strcmp(NNtype, 'CNN')
    NNtype_save = 'CNN';
else
    error('NNtype does not exist')
end
if nargin==10
    filename = ['CBOandPSO/NN/results/CBOmemorygradient/', NNtype_save, '/', 'CBOMNIST_worker', num2str(worker), '_memory_', num2str(parametersCBOmemorygradient('memory')), '_N_', num2str(parametersCBOmemorygradient('N')), '_nN_', num2str(batch_size_N), '_lambda1_', num2str(100*round(parametersCBOmemorygradient('lambda1'),2)),'e-2', '_sigma2**2_', num2str(1000*round(sigma2_initial^2,3)),'e-3', '_sigma1**2_', num2str(1000*round(sigma1_initial^2,3)),'e-3', '_alpha_', num2str(alpha_initial), '_beta_', num2str(parametersCBOmemorygradient('beta')), '_parametercooling_', num2str(parameter_cooling), '_', num2str(epochs), 'epochs', '_final'];
else
    filename = ['CBOandPSO/NN/results/CBOmemorygradient/', NNtype_save, '/', 'CBOMNIST', '_memory_', num2str(parametersCBOmemorygradient('memory')), '_N_', num2str(parametersCBOmemorygradient('N')), '_nN_', num2str(batch_size_N), '_lambda1_', num2str(100*round(parametersCBOmemorygradient('lambda1'),2)),'e-2', '_sigma2**2_', num2str(1000*round(sigma2_initial^2,3)),'e-3', '_sigma1**2_', num2str(1000*round(sigma1_initial^2,3)),'e-3', '_alpha_', num2str(alpha_initial), '_beta_', num2str(parametersCBOmemorygradient('beta')), '_parametercooling_', num2str(parameter_cooling), '_', num2str(epochs), 'epochs', '_final'];
end
parsave_CBOmemorygradient(filename, xstar_approx, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, parametersCBOmemorygradient('dt'), parametersCBOmemorygradient('N'), parametersCBOmemorygradient('memory'), parametersCBOmemorygradient('kappa'), parametersCBOmemorygradient('alpha'), parametersCBOmemorygradient('beta'), parametersCBOmemorygradient('lambda1'), parametersCBOmemorygradient('lambda2'), parametersCBOmemorygradient('gamma'), parametersCBOmemorygradient('learning_rate'), parametersCBOmemorygradient('anisotropic1'), parametersCBOmemorygradient('sigma1'), parametersCBOmemorygradient('anisotropic2'), parametersCBOmemorygradient('sigma2'), particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_XY_update, parametersInitialization('X0mean'), parametersInitialization('X0std'))


end