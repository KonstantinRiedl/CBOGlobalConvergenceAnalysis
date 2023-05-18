% Objective function plot 2d
%
% This script produces plots of the objective function E in 2d together 
% with the global minimizer.
%

%%
clear; clc; close all;

co = set_color();


%% Settings for Easy Handling and Notes
% save plot
pdfexport = 0;

% plot convex envelope in case of Rastrigin function
sqEuclildDist = 0;


%% Energy Function 

% % dimension of the ambient space
d = 2;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'Rastrigin';
[E, ~, parametersE, parametersCBO, parametersInitialization] = objective_function(objectivefunction, d, 'CBO');

% range of x (and x and y for plotting)
xrange_plot = parametersE(:,1)';
yrange_plot = parametersE(:,2)';
zrange_plot = parametersE(:,3)';
xrange = 100*xrange_plot;
yrange = 100*yrange_plot;

% global minimizer
vstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Plotting

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% % plot setting
f = figure('Position', [1200 800 500 400]);

% % plotting energy function E
[X,Y] = meshgrid(xrange_plot(1):.01:xrange_plot(2),yrange_plot(1):.01:yrange_plot(2));
XY = [X(:)';Y(:)'];
Z = E(XY);
Z = reshape(Z,size(X));

Eplot = surf(X,Y,Z,'FaceAlpha',0.55); % 0.5 und 0.25
Eplot.EdgeColor = 'None';
hold on

contour(X,Y,Z,22);

view(-25,12.5)
xlim(xrange_plot)
ylim(yrange_plot)
zlim(zrange_plot)


% % plot global minimizer of energy function E
%vstarplot = plot3(vstar(1), vstar(2), E(vstar), '*', 'MarkerSize', 10, 'LineWidth', 1.8, "color", co(5,:));
hold on

% % plot convex envelope
if strcmp(objectivefunction,'Rastrigin')
    Ec = @(v) sum(v.*v,1);
    ZZ = Ec(XY);
    ZZ = reshape(ZZ,size(X));
    
    if sqEuclildDist
        Ecplot = surf(X,Y,ZZ,'FaceAlpha',0.35);
        Ecplot.EdgeColor = 'None';
        hold on
    end
    
	xticks([-2.5 0 2.5 5])
    yticks([-2.5 0 2.5 5])
	zticks([0 10 20 30 40 50])
elseif strcmp(objectivefunction,'GrandCanyon2') || strcmp(objectivefunction,'GrandCanyon2noisy')
   
    xticks([-2.5 0 2.5 5])
    yticks([-2.5 0 2.5 5])
    zticks([0 10 20 30 40 50])
elseif strcmp(objectivefunction,'GrandCanyon3') || strcmp(objectivefunction,'GrandCanyon3noisy')
   
    xticks([-2 0 2 4 6 8])
    yticks([-2 0 2 4 6 8])
    zticks([0 10 20 30 40 50])
end

%legend([vstarplot], 'Global minimizer $v^*$','Location','northwest','Interpreter','latex','FontSize',13)

ax = gca;
ax.FontSize = 11;


%% Save Image
if pdfexport
    
    %cleanfigure;
    %matlab2tikz('myfile.tex');
    
    print(f,[main_folder(),'/EnergyBasedCBOAnalysis/images_videos/ObjectiveFunction2d_',objectivefunction],'-dpdf');

    % save parameters
    save([main_folder(),'/EnergyBasedCBOAnalysis/images_videos/ObjectiveFunction2d_',objectivefunction,'_param'], 'objectivefunction', 'E', 'vstar', 'd')

end
