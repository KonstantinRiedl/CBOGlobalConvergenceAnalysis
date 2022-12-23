% This function implements the plotting routine for phase transition plots
% to compare CBOgradient and standard CBO
% 
% Note: plotting parameters (objectivefunction, d, alpha, p, N)
% have to be specified in the function
% 
% 
% plot_phase_transition_diagram_CBOgradient_Figure6()
%           
% output:   plot
%

function plot_phase_transition_diagram_CBOgradient_Figure6()

objectivefunction = 'CompressedSensing';
d = 50;
alpha = 100;
p = 0.5;
N = 100;


load(['phasediagram_gammaM_', objectivefunction, '_', 'p', num2str(10*p),'div10_', 'd', num2str(d) , '_anisotropic11_alpha', num2str(alpha), '_beta-1_N', num2str(N),'_sigma10div100']);

indices_M = 1:1:length([1:1:40]);

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultLegendInterpreter','latex')

imagesc(flipud(pt_diagram_success(indices_M,1)))
%c = colorbar;
%set(c,'TickLabelInterpreter','latex')
%c.FontSize = 13;
set(gca, 'clim', [0 1]);
xlabel(['$\mathrm{CBO}$'],'interpreter','latex')
xticks(1)
xticklabels(['\newline'])
%xlabel(['$\mathrm{CBOGradient,}\ \', 'lambda_3', '$'],'interpreter','latex')
%xticks(1:length(pt_diagram_x_values))
%xticklabels(pt_diagram_x_values)
%xtickangle(0)
ylabel(['$', 'm', '$'],'interpreter','latex')
yticks(1:1:length(indices_M))
yticklabels('\, '+string(flipud(pt_diagram_y_values(indices_M(1:1:length(indices_M)))')'))
gc = gca;
labels = string(gc.YAxis.TickLabels); % extract
labels(2:5:end) = nan; % remove every other one
labels(3:5:end) = nan; % remove every other one
labels(4:5:end) = nan; % remove every other one
labels(5:5:end) = nan; % remove every other one
gc.YAxis.TickLabels = labels; % set
gc.FontSize = 13;


%%

figure()

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultLegendInterpreter','latex')

imagesc(flipud(pt_diagram_success(indices_M,:)))
c = colorbar;
set(c,'TickLabelInterpreter','latex')
c.FontSize = 13;
set(gca, 'clim', [0 1]);
xlabel(['$\mathrm{CBOGradient,}\ \', 'lambda_3', '$'],'interpreter','latex')
xticks(1:length(pt_diagram_x_values))
xticklabels(pt_diagram_x_values)
xtickangle(0)
%ylabel(['$', pt_diagram_y, '$'],'interpreter','latex')
%yticks(1:5:length(indices_M))
%yticklabels(flipud(pt_diagram_y_values(indices_M(5:5:length(indices_M)))')')
ylabel([],'interpreter','latex')
yticks(1:1:length(indices_M))
yticklabels([])
gc = gca;
labels = string(gc.XAxis.TickLabels); % extract
labels(2:5:end) = nan; % remove every other one
labels(3:5:end) = nan; % remove every other one
labels(4:5:end) = nan; % remove every other one
labels(5:5:end) = nan; % remove every other one
gc.XAxis.TickLabels = labels; % set
gc.FontSize = 13;

end
