% This function implements the plotting routine for performance (testing 
% accuracy and training accuracy) plots
% 
% 
% plot_training_testing_accuracy(performance_tracking)
% 
% input:    performance_tracking = tensor with saved performance measures
%               - training accuracy = performance_tracking(1,:,:)
%               - testing accuracy  = performance_tracking(2,:,:)
%               - testing loss      = performance_tracking(3,:,:)
%           
% output:   plot
%

function plot_training_testing_accuracy(performance_tracking)

[~, epochs, batch_size_E_batches] = size(performance_tracking); epochs = epochs-1;

performance_tracking_all = [performance_tracking(:,1,end), reshape(permute(performance_tracking(:,2:end,:),[1 3 2]), [3, epochs*batch_size_E_batches])];


epoch_batch_discretization = 0:1/batch_size_E_batches:epochs;

% plotting
figure('Position', [1200 800 600 350]);
co = set_color();

% plot training accuracy 
training_accuracy_plot = plot(epoch_batch_discretization, performance_tracking_all(1,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(1,:)); hold on
%plot(epoch_discretization, performance_tracking_epoch(1,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(1,:)); hold on

% plot testing accuracy
testing_accuracy_plot = plot(epoch_batch_discretization, performance_tracking_all(2,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(2,:)); hold on
%plot(epoch_discretization, performance_tracking_epoch(2,:), 'LineStyle', '-', 'LineWidth', 2, 'Color', co(2,:)); hold on

xlabel('number of epochs')
ylabel('accuracy')
ylim([0.4 1])
yticks(0:0.05:1)
xlim([0 10])
xticks(0:1:10)
legend([training_accuracy_plot, testing_accuracy_plot], {'Training accuracy','Testing accuracy'},'Location','southeast') %,'FontSize',16

end