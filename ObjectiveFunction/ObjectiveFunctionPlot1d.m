% Objective function plot 1d
%
% This script produces plots of the objective function E in 1d together 
% with the global minimizer and the convex envelope Ec of the function E.
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
d = 1; % only 1d due to the convex envelope computation and plotting

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


%% Convex Envelope Ec of E

% % computation of the convex hull of the energy function E
% we exploit the relationship between convex functions and their epigraphs
convhull_dom = linspace(xrange(1), xrange(2), 10^6);
Ec = [convhull_dom; E(convhull_dom)]';
[indices,~] = convhull(Ec);

convhull_x = Ec(indices,1); convhull_x(end) = [];
convhull_y = Ec(indices,2); convhull_y(end) = [];



%% Plotting

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% % plot setting
f = figure('Position', [1200 800 400 400]);


% % plotting convex envelope of E
Ecplot = plot(convhull_x, convhull_y, "color", co(6,:), 'LineWidth', 2, 'LineStyle', '--');
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

legend([Eplot, Ecplot, vstarplot], 'Objective function $\mathcal{E}$','Convex envelope $\mathcal{E}^c$','Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',15)

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
