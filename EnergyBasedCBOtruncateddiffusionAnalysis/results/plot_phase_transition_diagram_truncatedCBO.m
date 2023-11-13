% This function implements the plotting routine for phase transition plots
% to compare truncated CBO and standard CBO
% 
% Note: plotting parameters (objectivefunction, d, alpha) have to be 
% specified in the function
% 
% 
% plot_phase_transition_diagram_truncatedCBO()
%           
% output:   plot
%

function plot_phase_transition_diagram_truncatedCBO()

objectivefunction = 'Ackley';
d = 4;
alpha = 100000;

load(['phasediagram_Msigma_', objectivefunction, 'd', num2str(d), '_anisotropic0_alpha', num2str(alpha), '_N100.mat'])

indices_y = 1:1:length(pt_diagram_y_values);


%%

f = figure;
f.Position = [440   378   560   420];

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultLegendInterpreter','latex')


imagesc(flipud(pt_diagram(indices_y,1:end-1)))
c = colorbar;
c.FontSize = 13;
c.Limits = [0,1];
c.Location = 'westoutside';
set(c,'TickLabelInterpreter','latex')
colorbar( 'off' )
xlabel(['$\mathrm{truncated CBO,}\ ', 'M', '$'],'interpreter','latex')
xticks(1:length(pt_diagram_x_values))
xticklabels(pt_diagram_x_values)
xtickangle(0)
ylabel(['$\', pt_diagram_y, '$'],'interpreter','latex')
yticks(indices_y)
yticklabels(flipud(pt_diagram_y_values(indices_y)')')
ax = gca;
labels = string(ax.XAxis.TickLabels); % extract
labels(2:8:end) = nan; % remove every other one
labels(3:8:end) = nan; % remove every other one
labels(4:8:end) = nan; % remove every other one
labels(5:8:end) = nan; % remove every other one
labels(5:8:end) = nan; % remove every other one
labels(6:8:end) = nan; % remove every other one
labels(7:8:end) = nan; % remove every other one
labels(8:8:end) = nan; % remove every other one
ax.XAxis.TickLabels = labels; % set
ax.FontSize = 13;
ax = gca;
labels = string(ax.YAxis.TickLabels); % extract
labels(2:10:end) = nan; % remove every other one
labels(3:10:end) = nan; % remove every other one
labels(4:10:end) = nan; % remove every other one
labels(5:10:end) = nan; % remove every other one
labels(6:10:end) = nan; % remove every other one
labels(7:10:end) = nan; % remove every other one
labels(8:10:end) = nan; % remove every other one
labels(9:10:end) = nan; % remove every other one
labels(10:10:end) = nan; % remove every other one
ax.YAxis.TickLabels = labels; % set
ax.FontSize = 13;
set(gca,'TickLabelInterpreter','latex')



%%

f = figure;
f.Position = [440   378   50   420];

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultLegendInterpreter','latex')
set(gca,'TickLabelInterpreter','latex')

imagesc(flipud(pt_diagram(indices_y,end)))
c = colorbar;
c.FontSize = 13;
c.Limits = [0,1];
c.Location = 'eastoutside';
set(c,'TickLabelInterpreter','latex')
xlabel(['$\mathrm{CBO}$'],'interpreter','latex')
xticks(1)
xticklabels(['\newline'])
ylabel([],'interpreter','latex')
yticklabels([])
yticks(indices_y)

success_CBO = pt_diagram(indices_y,4)';
%%

f = figure;
f.Position = [440   378   560   420];

set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultLegendInterpreter','latex')
set(gca,'TickLabelInterpreter','latex')

imagesc(flipud([0,1]))
c = colorbar;
c.FontSize = 13;
c.Limits = [0,1];
c.Location = 'eastoutside';
set(c,'TickLabelInterpreter','latex')
xlabel(['$\mathrm{just for colorbar}$'],'interpreter','latex')
xticks(1)
xticklabels(['\newline'])
ylabel([],'interpreter','latex')
yticklabels([])
yticks(indices_y)
ax = gca;
labels = string(ax.YAxis.TickLabels); % extract
labels(2:10:end) = nan; % remove every other one
labels(3:10:end) = nan; % remove every other one
labels(4:10:end) = nan; % remove every other one
labels(5:10:end) = nan; % remove every other one
labels(6:10:end) = nan; % remove every other one
labels(7:10:end) = nan; % remove every other one
labels(8:10:end) = nan; % remove every other one
labels(9:10:end) = nan; % remove every other one
labels(10:10:end) = nan; % remove every other one
ax.YAxis.TickLabels = labels; % set
ax.FontSize = 13;

success_CBO = pt_diagram(indices_y,4)';


end
