% This function implements the plotting routine for performance (testing 
% accuracy and training loss) plots and can be used to compare different
% parameter settings
% 
% Note: plotting parameters (#architectures, #parameters, #epochs,
% batches_per_epoch) have to be specified in the function as well as the
% data to be visualized
% 
% 
% plot_loss_and_testing_accuracy_CBOmemory()
%           
% output:   plot
%

function plot_loss_and_testing_accuracy_CBOmemory()

zoom = 0;

number_of_architectures = 2;
number_of_parametersettings = 3; % needs to be consistent with the number of loaded settings

N_nN = 'N_100_nN_10'; % N_100_nN_10, N_100_nN_100
testing_or_training_accuracy = 'testing'; % training, testing
averaged_results = 1; % averaged results might not be available for all settings
number_of_samples = 4;

epochs_plot = 100;
batches_per_epoch = 1; % needs to be a common divisor (2,5,10) of the individual batches_per_epoch if any

performance_tracking_all = zeros(number_of_architectures,number_of_parametersettings,3,epochs_plot+1,batches_per_epoch);


% % % % load (accumulated/averaged) shallow NN data
% % darkest line: memory: 1, lambda1: 0.4
if ~averaged_results
    load([main_folder(),'/Example_NN/results/CBOmemorygradient/ShallowNN/CBOMNIST_memory_1_', N_nN, '_lambda1_40e-2_sigma2**2_400e-3_sigma1**2_64e-3_alpha_50_beta_-1_parametercooling_1_100epochs_1.mat'])
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,1,:,1:(epochs+1),:) = performance_tracking;
else
    for l=1:number_of_samples
        load([main_folder(),'/Example_NN/results/CBOmemorygradient/ShallowNN/CBOMNIST_memory_1_', N_nN, '_lambda1_40e-2_sigma2**2_400e-3_sigma1**2_64e-3_alpha_50_beta_-1_parametercooling_1_100epochs_', num2str(l),'.mat'])
        performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
        performance_tracking_all(2,1,:,1:(epochs+1),:) = performance_tracking_all(2,1,:,1:(epochs+1),:) + reshape(performance_tracking,[1,1,size(performance_tracking)])/number_of_samples;
    end
end


% % middle line: memory: 1, lambda1: 0
if ~averaged_results
    load([main_folder(),'/Example_NN/results/CBOmemorygradient/ShallowNN/CBOMNIST_memory_1_', N_nN, '_lambda1_0e-2_sigma2**2_400e-3_sigma1**2_0e-3_alpha_50_beta_-1_parametercooling_1_100epochs_1.mat'])
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,2,:,1:(epochs+1),:) = performance_tracking;
else
    for l=1:number_of_samples
        load([main_folder(),'/Example_NN/results/CBOmemorygradient/ShallowNN/CBOMNIST_memory_1_', N_nN, '_lambda1_0e-2_sigma2**2_400e-3_sigma1**2_0e-3_alpha_50_beta_-1_parametercooling_1_100epochs_', num2str(l),'.mat'])
        performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
        performance_tracking_all(2,2,:,1:(epochs+1),:) = performance_tracking_all(2,2,:,1:(epochs+1),:) + reshape(performance_tracking,[1,1,size(performance_tracking)])/number_of_samples;
    end
end

% % brightest line: memory: 0 (computed via CBO)
if ~averaged_results
    load([main_folder(),'/Example_NN/results/CBO/ShallowNN/CBOMNIST_', N_nN, '_sigma**2_0,4_alpha_50_parametercooling_1_100epochs_1.mat']) % from CBO folder
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(2,3,:,1:(epochs+1),:) = performance_tracking;
else
    for l=1:number_of_samples
        load([main_folder(),'/Example_NN/results/CBO/ShallowNN/CBOMNIST_', N_nN, '_sigma**2_0,4_alpha_50_parametercooling_1_100epochs_', num2str(l),'.mat'])
        performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
        performance_tracking_all(2,3,:,1:(epochs+1),:) = performance_tracking_all(2,3,:,1:(epochs+1),:) + reshape(performance_tracking,[1,1,size(performance_tracking)])/number_of_samples;
    end
end



% % % % load (accumulated/averaged) CNN data
% % darkest line: memory: 1, lambda1: 0.4
if ~averaged_results
    load([main_folder(),'/Example_NN/results/CBOmemorygradient/CNN/CBOMNIST_memory_1_', N_nN, '_lambda1_40e-2_sigma2**2_400e-3_sigma1**2_64e-3_alpha_50_beta_-1_parametercooling_1_100epochs_1.mat'])
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,1,:,1:(epochs+1),:) = performance_tracking;
else
    for l=1:number_of_samples
        load([main_folder(),'/Example_NN/results/CBOmemorygradient/CNN/CBOMNIST_memory_1_', N_nN, '_lambda1_40e-2_sigma2**2_400e-3_sigma1**2_64e-3_alpha_50_beta_-1_parametercooling_1_100epochs_', num2str(l),'.mat'])
        performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
        performance_tracking_all(1,1,:,1:(epochs+1),:) = performance_tracking_all(1,1,:,1:(epochs+1),:) + reshape(performance_tracking,[1,1,size(performance_tracking)])/number_of_samples;
    end
end

% % % middle line: memory: 1, lambda1: 0
if ~averaged_results
    load([main_folder(),'/Example_NN/results/CBOmemorygradient/CNN/CBOMNIST_memory_1_', N_nN, '_lambda1_0e-2_sigma2**2_400e-3_sigma1**2_0e-3_alpha_50_beta_-1_parametercooling_1_100epochs_1.mat'])
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,2,:,1:(epochs+1),:) = performance_tracking;
else
    for l=1:number_of_samples
        load([main_folder(),'/Example_NN/results/CBOmemorygradient/CNN/CBOMNIST_memory_1_', N_nN, '_lambda1_0e-2_sigma2**2_400e-3_sigma1**2_0e-3_alpha_50_beta_-1_parametercooling_1_100epochs_', num2str(l),'.mat'])
        performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
        performance_tracking_all(1,2,:,1:(epochs+1),:) = performance_tracking_all(1,2,:,1:(epochs+1),:) + reshape(performance_tracking,[1,1,size(performance_tracking)])/number_of_samples;
    end
end

% % brightest line: memory: 0 (computed via CBO)
if ~averaged_results
    load([main_folder(),'/Example_NN/results/CBO/CNN/CBOMNIST_', N_nN, '_sigma**2_0,4_alpha_50_parametercooling_1_100epochs_1.mat']) % from CBO folder
    performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
    performance_tracking_all(1,3,:,1:(epochs+1),:) = performance_tracking;
else
    for l=1:number_of_samples
        load([main_folder(),'/Example_NN/results/CBO/CNN/CBOMNIST_', N_nN, '_sigma**2_0,4_alpha_50_parametercooling_1_100epochs_', num2str(l),'.mat'])
        performance_tracking = performance_tracking(:,:,linspace(size(performance_tracking,3)/batches_per_epoch,size(performance_tracking,3),batches_per_epoch));
        performance_tracking_all(1,3,:,1:(epochs+1),:) = performance_tracking_all(1,3,:,1:(epochs+1),:) + reshape(performance_tracking,[1,1,size(performance_tracking)])/number_of_samples;;
    end
end



%performance_tracking_all = performance_tracking_all(1:number_of_architectures,1:number_of_parametersettings,1:3,1:(epochs_plot+1),1:10);
performance_tracking_all(performance_tracking_all==0) = nan;


%% plotting
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

figure('Position', [1200 800 600 350]);
co = set_color();

loss_color = co(1,:);
accuracy_color = co(2,:);

for k = 1:number_of_architectures
    
    for l = 1:number_of_parametersettings
    
        performance_tracking = reshape(performance_tracking_all(k,l,:,:,:), [3,size(performance_tracking_all(k,l,:,:,:),4),batches_per_epoch]);

        [~, epochs_, batch_size_E_batches] = size(performance_tracking); epochs_ = epochs_-1;

        performance_tracking_k = [performance_tracking(:,1,end), reshape(permute(performance_tracking(:,2:end,:),[1 3 2]), [3, epochs_*batch_size_E_batches])];

        epoch_batch_discretization = 0:1/batch_size_E_batches:epochs_;

        if k==1
            line_style = '-';
        elseif k==2
            line_style = '--';
        elseif k==3
            line_style = '-.';
        else
            line_style = ':';
        end
        
        if l==1
            line_opacity = 1;
        elseif l==2
            line_opacity = 0.6;
        elseif l==3
            line_opacity = 0.2;
        else
            error('Change line_opacities.')
        end

        yyaxis right
        % plot testing or training accuracy
        if strcmp(testing_or_training_accuracy, 'testing')
            testing_or_training_index = 2;
        elseif strcmp(testing_or_training_accuracy, 'training')
            testing_or_training_index = 1;
        else
            error('testing_or_training_accuracy not known.')
        end
        kk((k-1)*3+1) = plot(epoch_batch_discretization, performance_tracking_k(testing_or_training_index,:), 'LineWidth', 2, 'LineStyle', line_style, 'Color', [accuracy_color, line_opacity], 'Marker', 'none'); hold on
        %plot(epoch_discretization, performance_tracking_epoch(1,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(1,:)); hold on

        xlabel('number of epochs','Interpreter','latex','FontSize',15)
        if strcmp(testing_or_training_accuracy, 'testing')
            ylabel('testing accuracy','Interpreter','latex','FontSize',15)
        elseif strcmp(testing_or_training_accuracy, 'training')
            ylabel('training accuracy','Interpreter','latex','FontSize',15)
        end
        ylim([0.6 1])
        yticks(0:0.04:1)
        %yticks(0:0.005:1)
        if epochs_plot<=20
            xlim([0 epochs_plot])
            xticks(0:1:epochs_plot)
        else
            xlim([0 epochs_plot])
            xticks(0:10:epochs_plot)
        end
        ax = gca;
        ax.FontSize = 13;
        ax.YColor = accuracy_color;
        box on

        yyaxis left
        % plot training loss
        kk((k-1)*3+2) = plot(epoch_batch_discretization, performance_tracking_k(3,:), 'LineWidth', 2, 'LineStyle', line_style, 'Color', [loss_color, line_opacity], 'Marker', 'none'); hold on
        %plot(epoch_discretization, performance_tracking_epoch(2,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(2,:)); hold on

        ylabel('crossentropy risk~$\mathcal{E}$','Interpreter','latex','FontSize',15)
        ylim([0 0.2])
        yticks(0:0.02:1)
        ax = gca;
        ax.YColor = loss_color;


        % plot for legend
        %(k-1)*number_of_architectures+3
        kk((k-1)*3+3) = plot(epoch_batch_discretization, 100*ones(size(epoch_batch_discretization)), 'LineWidth', 2, 'LineStyle', line_style, 'Color', 'black', 'Marker', 'none'); hold on
        
        yyaxis right
    end
end
grid on
ax = gca;
ax.GridColor = [0.2, 0.2, 0.2];

legend(kk([3 6]), 'Convolutional neural network','Shallow neural network','Location','southeast','Interpreter','latex','FontSize',15)

if zoom && strcmp(N_nN, 'N_100_nN_10')

    leg = legend;
    set(leg,'visible','off')

    % accuracy zoom plots
    yyaxis left
    ylabel('')
    ax = gca;
    ax.YColor = 'black';
    ylim([0.999 0.9991])
    yyaxis right
    %ylim([0.964 0.974])
    %yticks(0:0.002:1)
    ylim([0.893 0.903])
    yticks(0.001:0.002:1)
    ylabel(' ')
    
    % risk zoom plots
    %yyaxis right
    %ylabel('')
    %ax = gca;
    %ax.YColor = 'black';
    %ylim([-600 -599])
    %yyaxis left
    %ylim([0.06 0.065])
    %%ylim([0.041 0.046])
    %yticks(0:0.001:1)
    %ylabel('')
    
    
    grid off
    xlim([95 100])
    xticks(95:100)
    xlabel('')

    ax = gca;
    ax.FontSize = 25;
    set(gca,'linewidth',1.1)

elseif zoom && strcmp(N_nN, 'N_100_nN_100')

    leg = legend;
    set(leg,'visible','off')

    % accuracy zoom plots
    yyaxis left
    ylabel('')
    ax = gca;
    ax.YColor = 'black';
    ylim([0.999 0.9991])
    yyaxis right
    %ylim([0.954 0.964])
    %yticks(0:0.002:1)
    ylim([0.881 0.891])
    yticks(0.001:0.002:1)
    ylabel(' ')
    
    % risk zoom plots
    %yyaxis right
    %ylabel('')
    %ax = gca;
    %ax.YColor = 'black';
    %ylim([-600 -599])
    %yyaxis left
    %ylim([0.063 0.068])
    %%ylim([0.044 0.049])
    %yticks(0:0.001:1)
    %ylabel('')
    
    
    grid off
    xlim([95 100])
    xticks(95:100)
    xlabel('')

    ax = gca;
    ax.FontSize = 25;
    set(gca,'linewidth',1.1)

end

end


