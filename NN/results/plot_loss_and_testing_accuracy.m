function plot_loss_and_testing_accuracy()

number_of_architectures = 2;
epochs_plot = 10;

performance_tracking_all = nan(number_of_architectures,3,21,epochs_plot);


% load CNN data
load('/Users/Konstantin/MATLAB/Research/CBOandPSO/NN/results/CBO/CNN/CBOMNIST_31-Oct-2021_10epochs.mat')
performance_tracking_all(1,:,:,:) = performance_tracking;
% load shallow NN data
load('/Users/Konstantin/MATLAB/Research/CBOandPSO/NN/results/CBO/ShallowNN/CBOMNIST_shallow_N1000_sigmas0,4_alpha50_parametercooling1.mat')
performance_tracking_all(2,:,:,:) = performance_tracking;



% plotting

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

figure('Position', [1200 800 600 350]);
co = set_color();

loss_color = co(1,:);
accuracy_color = co(2,:);

for k=1:number_of_architectures
    
    performance_tracking = reshape(performance_tracking_all(k,:,:,:), [3,21,epochs_plot]);
    
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
    
    yyaxis right
    % plot training accuracy 
    kk((k-1)*3+1) = plot(epoch_batch_discretization, performance_tracking_k(1,:), 'LineWidth', 2, 'LineStyle', line_style, 'Color', accuracy_color); hold on
    %plot(epoch_discretization, performance_tracking_epoch(1,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(1,:)); hold on

    xlabel('number of epochs','Interpreter','latex','FontSize',15)
    ylabel('testing accuracy','Interpreter','latex','FontSize',15)
    ylim([0.6 1])
    yticks(0:0.04:1)
    xlim([0 epochs_plot])
    xticks(0:1:epochs_plot)
    ax = gca;
    ax.FontSize = 13;
    ax.YColor = accuracy_color;
    box on

    yyaxis left
    % plot testing accuracy
    kk((k-1)*3+2) = plot(epoch_batch_discretization, performance_tracking_k(3,:), 'LineWidth', 2, 'LineStyle', line_style, 'Color', loss_color); hold on
    %plot(epoch_discretization, performance_tracking_epoch(2,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(2,:)); hold on

    ylabel('crossentropy risk~$\mathcal{E}$','Interpreter','latex','FontSize',15)
    ylim([0 0.2])
    yticks(0:0.02:1)
    ax = gca;
    ax.YColor = loss_color;
    
    
    % plot for legend
    %(k-1)*number_of_architectures+3
    kk((k-1)*3+3) = plot(epoch_batch_discretization, 100*ones(size(epoch_batch_discretization)), 'LineWidth', 2, 'LineStyle', line_style, 'Color', 'black'); hold on
    
end

legend(kk([3 6]), 'Convolutional neural network','Shallow neural network','Location','southeast','Interpreter','latex','FontSize',15)

end


