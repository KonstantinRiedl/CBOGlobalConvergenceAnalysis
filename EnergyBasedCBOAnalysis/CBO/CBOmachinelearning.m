function [vstar_approx, performance_tracking] = CBOmachinelearning(parametersCBO, parametersbatch, parametersInitialization, train_data, train_label, test_data, test_label, image_size, NN_architecture, worker)

% get parameters
epochs = parametersCBO('epochs');
batch_size_N = parametersbatch('batch_size_N');
batch_size_E = parametersbatch('batch_size_E');
full_or_partial_V_update = parametersbatch('full_or_partial_V_update');

NNtype = NN_architecture('NNtype');
architecture = NN_architecture('architecture');
neurons = NN_architecture('neurons');
d = NN_architecture('d');


% initialization
V0 = parametersInitialization('V0mean') + parametersInitialization('V0std')*randn(d,parametersCBO('N'));
V = V0;
v_alpha_on_batch_old = zeros(d,1);


% % definition of the risk (objective function E)
% predicted and true label
predicted_label = @(x, data) NN(x, data, image_size, NNtype, architecture, neurons);
true_label_1hot = @(x, label) reshape(repmat((label==(0:9)'),[1, size(x,2)]), [neurons(1,end), size(label,2), size(x,2)]);
% categorical crossentropy loss function
categ_CE_loss = @(x, data, label) -1/neurons(1,end)*reshape(sum(true_label_1hot(x, label).*log(predicted_label(x, data)), 1), [size(data, 2), size(x, 2)]);
% risk (objective function E)
E = @(x, data, label) 1/size(data, 2)*sum(categ_CE_loss(x, data, label),1);


% performance tracking during training
training_batches_per_epoch = size(train_data,2)/batch_size_E;
recording_sample_size = 10000;
recording_frequency = 100; % ensure that recording_frequency divides training_batches_per_epoch
performance_tracking = NaN(3, epochs+1, training_batches_per_epoch/recording_frequency);


% compute, display and save initial performance
rand_indices_train = randsample(size(train_data,2),recording_sample_size)';
rand_indices_test = randsample(size(test_data,2),recording_sample_size)';
alg_state = containers.Map({'epoch', 'epochs', 'batch', 'training_batches_per_epoch'},...
                           {      0,   epochs,       0,                            0});
if nargin==10
    [train_accuracy, test_accuracy, objective_value] = comp_performance(E, V, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1, worker);
else
    [train_accuracy, test_accuracy, objective_value] = comp_performance(E, V, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1);
end
performance_tracking(1,1,end) = train_accuracy;
performance_tracking(2,1,end) = test_accuracy;
performance_tracking(3,1,end) = objective_value;

% % CBO
for epoch = 1:epochs
    
    % employ optional particle reduction strategy
    particle_reduction = parametersCBO('particle_reduction');
    if particle_reduction
        varianceV_before = norm(var(V,0,2));
    end
    
    for batch = 1:training_batches_per_epoch
        
        % % definition of objective function E on current training batch
        % indices of current training batch
        indices_t_b = (batch-1)*batch_size_E+(1:batch_size_E);
        batch_data = train_data(:,indices_t_b);
        batch_label = train_label(indices_t_b);
        % objective function E on current training batch
        E_train_batch = @(x) E(x, batch_data, batch_label);
        
        
        % % update of particles' positions 
        particle_batches = parametersCBO('N')/batch_size_N;
        permutation = randperm(parametersCBO('N'));
        V = V(:,permutation);
        for particle_batch = 1:particle_batches
            
            % indices of current particle batch
            indices_p_b = (particle_batch-1)*batch_size_N+(1:batch_size_N);
            V_particle_batch = V(:,indices_p_b);
            
            
            % % CBO iteration
            % compute current consensus point v_alpha
            v_alpha_on_batch = compute_valpha(E_train_batch, parametersCBO('alpha'), V_particle_batch);

            % position updates of one iteration of CBO
            if strcmp(full_or_partial_V_update, 'partial')
                V_particle_batch = CBO_update(E_train_batch, parametersCBO, v_alpha_on_batch, V_particle_batch);
                V(:,indices_p_b) = V_particle_batch;
            elseif strcmp(full_or_partial_V_update, 'full')
                V = CBO_update(E_train_batch, parametersCBO, v_alpha_on_batch, V);
            else
                error('full_or_partial_V_update type not known.')
            end
            
            % random Brownian motion if v_alpha_on_batch is not changing
            if norm(v_alpha_on_batch_old-v_alpha_on_batch, 'inf') < 10^-5
                dB = randn(d,parametersCBO('N'));
                dt = parametersCBO('dt');
                V = V + parametersCBO('sigma')*sqrt(dt).*dB;
                clear dB dt
            end
            v_alpha_on_batch_old = v_alpha_on_batch;
            clear valpha_on_batch
            
        end
        
        % compute, display and save current performance
        if mod(batch,recording_frequency)==0
            rand_indices_train = randsample(size(train_data,2),recording_sample_size)';
            rand_indices_test = randsample(size(test_data,2),recording_sample_size)';
            alg_state = containers.Map({'epoch', 'epochs', 'batch', 'training_batches_per_epoch'},...
                                       {  epoch,   epochs,   batch,  training_batches_per_epoch});
            if nargin==10
                [train_accuracy, test_accuracy, objective_value] = comp_performance(E, V, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1, worker);
            else
                [train_accuracy, test_accuracy, objective_value] = comp_performance(E, V, train_data(:,rand_indices_train), test_data(:,rand_indices_test), image_size, train_label(:,rand_indices_train), test_label(:,rand_indices_test), NNtype, architecture, neurons, alg_state, 1);
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
        varianceV = norm(var(V,0,2));
        reduction_factor = min(max(1-speedup_PRS,(speedup_PRS*varianceV+(1-speedup_PRS)*varianceV_before)/varianceV_before),1);
        if reduction_factor<1
            parametersCBO('N') = ceil(parametersCBO('N')/batch_size_N*reduction_factor)*batch_size_N;
            V = V(:,randsample(size(V,2), parametersCBO('N'))');
        end
    end
    
    % employ optional and cooling strategy (CS) 
    parameter_cooling = parametersCBO('parameter_cooling');
    if parameter_cooling
        parametersCBO('sigma') = parametersCBO('sigma')*log2(epoch+1)/log2(epoch+2);
        parametersCBO('alpha') = parametersCBO('alpha')*2;
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
        filename = ['CBOandPSO/NN/results/CBO/', NNtype_save, '/', 'CBOMNIST_worker', num2str(worker), '_', datestr(datetime('today')), 'N', num2str(parametersCBO('N')),'_preliminary'];
    else
        filename = ['CBOandPSO/NN/results/CBO/', NNtype_save, '/', 'CBOMNIST', '_', datestr(datetime('today')), 'N', num2str(parametersCBO('N')), '_preliminary'];
    end
    parsave_CBO(filename, v_alpha_on_batch_old, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, parametersCBO('dt'), parametersCBO('N'), parametersCBO('alpha'), parametersCBO('lambda'), parametersCBO('gamma'), parametersCBO('learning_rate'), parametersCBO('anisotropic'), parametersCBO('sigma'), particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_V_update, parametersInitialization('V0mean'), parametersInitialization('V0std'))
    
end
  
E_train = @(x) E(x, train_data, train_label);
vstar_approx = compute_valpha(E_train, parametersCBO('alpha'), V);

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
    filename = ['CBOandPSO/NN/results/CBO/', NNtype_save, '/', 'CBOMNIST_worker', num2str(worker), '_', datestr(datetime('today')), 'N', num2str(parametersCBO('N')), '_final'];
else
    filename = ['CBOandPSO/NN/results/CBO/', NNtype_save, '/', 'CBOMNIST', '_', datestr(datetime('today')), 'N', num2str(parametersCBO('N')), '_final'];
end
parsave_CBO(filename, vstar_approx, performance_tracking, image_size, NN_architecture, NNtype, architecture, neurons, d, epochs, parametersCBO('dt'), parametersCBO('N'), parametersCBO('alpha'), parametersCBO('lambda'), parametersCBO('gamma'), parametersCBO('learning_rate'), parametersCBO('anisotropic'), parametersCBO('sigma'), particle_reduction, parameter_cooling, batch_size_N, batch_size_E, full_or_partial_V_update, parametersInitialization('V0mean'), parametersInitialization('V0std'))


end