% A lower bound for the probability mass  around v^*
%
% This script visualizes the partition used in the proof of Proposition 19.
% Wlog we may assume here that v^*=0.
% Such plot is used in Figure 3.
%

%%
clear; clc; close all;

co = set_color();

%% Settings for Easy Handling and Notes
% 
opacity = 0.75;

% save plot
pdfexport = 0;

% doublecheck
CHECK = 0;
disp('ALWAYS DOUBLECHECK p_set_2 by setting CHECK=1. Only accept gray set if black dotted point cloud coincides with it.')
% e.g. for sigma=2, v_alpha_norm=0.96, the sets do NOT coincide as the set is not connected.


%% Parameters
lambda = 1; % wlog 1
sigma = 1;                                    %%%%%%%%%%% can be modified
d = 2;
r = 1; % wlog 1
c = 0.6667; % in (0.5,1), 2/3                   %%%%%%%%%%% can be modified
if (c*(2*c-1) >= 2*(1-c)^2)
    disp('c suitable. (satisfies (2c-1)c >= d(1-c)^2)')
else
    disp('c not suitable. INCREASE c.')
end
cc = 2*c-1;

% v_alpha
v_alpha_norm = 1; % 1.1, 0.9, 1               %%%%%%%%%%% can be modified
v_alpha = v_alpha_norm*[1.5,1]/sqrt(3.25);
%v_alpha = v_alpha_norm*[1,1]/sqrt(1);


%% Plotting of Partition
f = figure('Position', [1200 800 400 400]);

% % meshgrid
if pdfexport
    x = -r:0.0001:r;
    y = -r:0.0001:r;
else
    x = -r:0.001:r;
    y = -r:0.001:r;
end
[X,Y] = meshgrid(x,y);

% % Definition of Sets

% support supp(phi_r)
support = X.^2+Y.^2 <= r^2;
%p_support = plot(X(support),Y(support),'.', 'MarkerEdgeColor', co(3,:)); hold on

% K_1
K_1 = X.^2+Y.^2 > c*r^2;
K_1 = logical(K_1.*support);
%p_K_1 = plot(X(K_1),Y(K_1),'.', 'MarkerEdgeColor',co(4,:)); hold on

% K_2
K_2 = -lambda*((X-v_alpha(1)).*X+(Y-v_alpha(2)).*Y).*(r^2-(X.^2+Y.^2)).^2 >cc*r^2*sigma^2/2*((X-v_alpha(1)).^(2)+(Y-v_alpha(2)).^(2)).*(X.^2+Y.^2);
K_2 = logical(K_2.*support);
%p_K_2 = fill(X(K_2),Y(K_2), co(2,:)); hold on



% % disjoint sets in partition
% K_1^c n support
set_1 = logical((1-K_1).*support);
% K_1 n K_2 n support
set_2 = logical(K_1.*K_2.*support);
% K_1 n K_2^c n support
set_3 = logical(K_1.*(1-K_2).*support);



% % plotting disjoint sets in partition
th = 0:pi/10^3:2*pi;
% plot K_1 n K_2^c n support
%p_set_2 = plot(X(set_3),Y(set_3),'.', 'MarkerEdgeColor',co(2,:)); hold on
x_support_unit = r*cos(th);
y_support_unit = r*sin(th);
p_set_2 = fill(x_support_unit, y_support_unit,co(2,:)); hold on
set(p_set_2,'facealpha',opacity)
set(p_set_2,'LineWidth',0.5)


% plot K_1 n K_2 n support
%p_set_3 = plot(X(set_2),Y(set_2),'.', 'MarkerEdgeColor',co(9,:)); hold on
set_2_x = X(set_2);
set_2_y = Y(set_2);
k = boundary(set_2_x,set_2_y);
fill(set_2_x(k),set_2_y(k),'w'); hold on % white baselayer
p_set_3 = fill(set_2_x(k),set_2_y(k),co(9,:));hold on
set(p_set_3,'facealpha',opacity)
set(p_set_3,'LineWidth',0.5)
if CHECK
   plot(X(set_2),Y(set_2),'.', 'MarkerEdgeColor','k'); hold on
end


% plot K_1^c n support
%p_set_1 = plot(X(set_1),Y(set_1),'.', 'MarkerEdgeColor',co(1,:)); hold on
x_K_1_unit = sqrt(c)*r*cos(th);
y_K_1_unit = sqrt(c)*r*sin(th);
fill(x_K_1_unit, y_K_1_unit,'w'); hold on % white baselayer
p_set_1 = fill(x_K_1_unit, y_K_1_unit,co(1,:)); hold on
set(p_set_1,'facealpha',opacity)
set(p_set_1,'LineWidth',0.5)



% % % % maximal set where noise term compensates drift
% set__ = lambda*((X-v_alpha(1)).*X+(Y-v_alpha(2)).*Y).*(r^2-(X.^2+Y.^2)).^2 + sigma^2/2*((X-v_alpha(1)).^2+(Y-v_alpha(2)).^2).*(2*(2*(X.^2+Y.^2)-r^2).*(X.^2+Y.^2)-2*(r^2-(X.^2+Y.^2)).^2);
% set__ = boolean((set__ >= 0).*((X.^2+Y.^2)<=r^2));
% 
% plot(X(set__),Y(set__),'.', 'MarkerEdgeColor','g'); hold on


% % % % maximal set where noise term compensates drift (case: l_infty ball)
% set__x = (lambda*((X-v_alpha(1)).*X).*(r^2-(X.^2)).^2 + sigma^2/2*((X-v_alpha(1)).^2+(Y-v_alpha(2)).^2).*(2*(2*(X.^2)-r^2).*(X.^2)-2*(r^2-(X.^2)).^2)).*(r^2-Y.^2).^4;
% set__y = (lambda*((Y-v_alpha(2)).*Y).*(r^2-(Y.^2)).^2 + sigma^2/2*((X-v_alpha(1)).^2+(Y-v_alpha(2)).^2).*(2*(2*(Y.^2)-r^2).*(Y.^2)-2*(r^2-(Y.^2)).^2)).*(r^2-X.^2).^4;
% set__ = (set__x+set__y) >= 0;
% 
% plot(X(set__),Y(set__),'.', 'MarkerEdgeColor','g'); hold on


% % 
% v_star
p_v_star = plot(0,0, '*', 'MarkerSize', 12, 'LineWidth', 1.8, "color", co(5,:));

% v_alpha
p_v_alpha = plot(v_alpha(1),v_alpha(2), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));

xlim([-r,r]); ylim([-r,r]); %xlim([-2*r,2*r]); ylim([-2*r,2*r]);
ticks = -r:0.5*r:r;
xticks(ticks); yticks(ticks);
ticks_lab = {'-$r$','-0.5$r$','0','0.5$r$','$r$'};
set(groot,'defaultAxesTickLabelInterpreter','latex'); 
xticklabels(ticks_lab); yticklabels(ticks_lab);

legend([p_v_star, p_v_alpha, p_set_1, p_set_2, p_set_3], 'Global minimizer $v^*$', 'Consensus point $v_{\alpha}(\rho_t)$','$K_1^c \cap \Omega_r$','$K_1 \cap K_2^c \cap \Omega_r$','$K_1 \cap K_2 \cap \Omega_r$','Location','southwest','Interpreter','latex','FontSize',17)

%tit = ['$\sigma =\; $', num2str(sigma), ', $\|v_{\alpha}\|_2=\; $', num2str(v_alpha_norm)];
%title(tit,'Interpreter','latex','FontSize',16)

ax = gca;
ax.FontSize = 15;

axis('equal')
axis('tight')


%% Save Image
if pdfexport
    
    filename = ['Prop19_isotropic_','v_alpha_norm',num2str(100*v_alpha_norm),'div100','sigma',num2str(100*sigma),'div100'];
    
    print(f,['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/', filename],'-dpdf');

    % save parameters
    save(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/', filename], 'sigma', 'v_alpha_norm', 'v_alpha', 'c')
    
end




