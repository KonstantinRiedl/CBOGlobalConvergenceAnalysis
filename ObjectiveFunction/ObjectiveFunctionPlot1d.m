% Objective function plot 1d
%
% This script produces plots of the objective function E in 1d together 
% with the global minimizer and the squared Euclidean norm distance.
% Such plot is used in Figure 2(a).
%

%%
clear; clc; close all;

co = set_color();


%% Settings for Easy Handling and Notes
% save plot
pdfexport = 0;


%% Energy Function E

% % dimension of the ambient space
d = 1;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, parametersE, parametersCBO] = objective_function(objectivefunction,d);

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
xrange = 100*xrange_plot;

% global minimizer
vstar = 0; %fminbnd(E,xrange_plot(1),xrange_plot(2));



%% Plotting

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% % plot setting
f = figure('Position', [1200 800 400 400]);


% % plotting squared Euclidean distance to v^*
sqEuclidNorm = plot([xrange_plot(1):0.0001:xrange_plot(2)], [xrange_plot(1):0.0001:xrange_plot(2)].^2, "color", co(6,:), 'LineWidth', 2, 'LineStyle', '--');
hold on
% % plotting energy function E
Eplot = fplot(E, xrange_plot, "color", co(1,:), 'LineWidth', 2);
xlim(xrange_plot)
ylim(yrange_plot)
if strcmp(objectivefunction,'Rastrigin')
	xticks([-2.5 0 2.5 5])
	yticks([0 5 10 15 20 25])
end
hold on
% % plot global minimizer of energy function E
vstarplot = plot(vstar, E(vstar), '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:));

%legend([Eplot, sqEuclidNorm, vstarplot], 'Objective function $\mathcal{E}$','$v\mapsto\frac{1}{2}\|v-v^*\|_2^2$','Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',15)
legend([Eplot, sqEuclidNorm, vstarplot], 'Objective function $\mathcal{E}$','Squared norm distance from $v^*$','Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',15)


ax = gca;
ax.FontSize = 13;



%% Save Image
if pdfexport
    
    %cleanfigure;
    %matlab2tikz('myfile.tex');

    print(f,['images_videos/ObjectiveFunction_',objectivefunction],'-dpdf');

    % save parameters
    save(['images_videos/ObjectiveFunction_',objectivefunction,'_param'], 'objectivefunction', 'E', 'vstar', 'd')

end
