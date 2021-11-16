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
opacity = 0.75/1.5;

% save plot
pdfexport = 1;

% doublecheck
CHECK = 0;
disp('ALWAYS DOUBLECHECK p_set_2 by setting CHECK=1. Only accept gray set if black dotted point cloud coincides with it.')


%% Parameters
lambda = 1; % wlog 1
sigma = 1;                                    %%%%%%%%%%% can be modified
d = 2;
r = 1; % wlog 1
c = 0.64; % in (0.5,1), optimal choice is (sqrt(5)-1)/2, however try to choose that resolution of meshgrid fits
cc = 2*c-1;

% v_alpha
%v_alpha = v_alpha_norm*[1,1];
v_alpha = [0.9, 1.1]; % [0.9, 1.1]; [0.9, 0.8];

%% Plotting of Partition
f = figure('Position', [1200 800 400 380]);

% % meshgrid
if pdfexport
    x = -r:0.001:r;
    y = -r:0.001:r;
else
    x = -r:0.001:r;
    y = -r:0.001:r;
end
[X,Y] = meshgrid(x,y);

% % Definition of Sets

% support supp(phi_r)
support = max(X,Y).^2 <= r^2;

% K_1
K_11 = abs(X).^2 >= c*r^2;
K_11 = logical(K_11.*support);
K_12 = abs(Y).^2 >= c*r^2;
K_12 = logical(K_12.*support);


% K_2
K_21 = -lambda*(X-v_alpha(1)).*X.*(r^2-X.^2).^2 > cc*r^2*sigma^2/2*(X-v_alpha(1)).^2.*X.^2;
K_21 = logical(K_21.*support);
K_22 = -lambda*(Y-v_alpha(2)).*Y.*(r^2-Y.^2).^2 > cc*r^2*sigma^2/2*(Y-v_alpha(2)).^2.*Y.^2;
K_22 = logical(K_22.*support);



% % disjoint sets in partition
% K_1^c n support
set_11 = logical((1-K_11).*support);
set_12 = logical((1-K_12).*support);
% K_1 n K_2^c n support
set_21 = logical(K_11.*(1-K_21).*support);
set_22 = logical(K_12.*(1-K_22).*support);
% K_1 n K_2 n support
set_31 = logical(K_11.*K_21.*support);
set_32 = logical(K_12.*K_22.*support);



% % % plotting disjoint sets in partition
% % % plot K_11^c n support
%p_set_11 = plot(X(set_11),Y(set_11),'.', 'MarkerEdgeColor',co(1,:)); hold on
set_11_x = X(set_11);
set_11_y = Y(set_11);
k = boundary(set_11_x,set_11_y);
p_set_11 = fill(set_11_x(k),set_11_y(k),co(1,:));hold on
set(p_set_11,'facealpha',opacity)
set(p_set_11,'LineWidth',0.5)
% % % plot K_12^c n support
%p_set_12 = plot(X(set_12),Y(set_12),'.', 'MarkerEdgeColor',co(1,:)); hold on
set_12_x = X(set_12);
set_12_y = Y(set_12);
k = boundary(set_12_x,set_12_y);
p_set_12 = fill(set_12_x(k),set_12_y(k),co(1,:));hold on
set(p_set_12,'facealpha',opacity)
set(p_set_12,'LineWidth',0.5)


% % plot K_11 n K_21^c n support
%p_set_21 = plot(X(set_21),Y(set_21),'.', 'MarkerEdgeColor',co(2,:)); hold on
set_21neg = boolean(set_21.*(X<=0));
set_21_xneg = X(set_21neg);
set_21_yneg = Y(set_21neg);
k = boundary(set_21_xneg,set_21_yneg);
p_set_21neg = fill(set_21_xneg(k),set_21_yneg(k),co(2,:)); hold on
set(p_set_21neg,'facealpha',opacity)
set(p_set_21neg,'LineWidth',0.5)
set_21pos = boolean(set_21.*(X>=0));
set_21_xpos = X(set_21pos);
set_21_ypos = Y(set_21pos);
k = boundary(set_21_xpos,set_21_ypos);
p_set_21pos = fill(set_21_xpos(k),set_21_ypos(k),co(2,:)); hold on
set(p_set_21pos,'facealpha',opacity)
set(p_set_21pos,'LineWidth',0.5)
% % plot K_12 n K_22^c n support
%p_set_22 = plot(X(set_22),Y(set_22),'.', 'MarkerEdgeColor',co(2,:)); hold on
set_22neg = boolean(set_22.*(Y<=0));
set_22_xneg = X(set_22neg);
set_22_yneg = Y(set_22neg);
k = boundary(set_22_xneg,set_22_yneg);
p_set_22neg = fill(set_22_xneg(k),set_22_yneg(k),co(2,:)); hold on
set(p_set_22neg,'facealpha',opacity)
set(p_set_22neg,'LineWidth',0.5)
set_22pos = boolean(set_22.*(Y>=0));
set_22_xpos = X(set_22pos);
set_22_ypos = Y(set_22pos);
k = boundary(set_22_xpos,set_22_ypos);
p_set_22pos = fill(set_22_xpos(k),set_22_ypos(k),co(2,:)); hold on
set(p_set_22pos,'facealpha',opacity)
set(p_set_22pos,'LineWidth',0.5)


% % plot K_11 n K_21 n support
%p_set_31 = plot(X(set_31),Y(set_31),'.', 'MarkerEdgeColor',co(9,:)); hold on
set_31_x = X(set_31);
set_31_y = Y(set_31);
k = boundary(set_31_x,set_31_y);
p_set_31 = fill(set_31_x(k),set_31_y(k),co(9,:));hold on
set(p_set_31,'facealpha',opacity)
set(p_set_31,'LineWidth',0.5)
% plot K_12 n K_22 n support
%p_set_32 = plot(X(set_32),Y(set_32),'.', 'MarkerEdgeColor',co(9,:)); hold on
set_32_x = X(set_32);
set_32_y = Y(set_32);
k = boundary(set_32_x,set_32_y);
p_set_32 = fill(set_32_x(k),set_32_y(k),co(9,:));hold on
set(p_set_32,'facealpha',opacity)
set(p_set_32,'LineWidth',0.5)


% % % % % %  
if CHECK
   %plot(X(set_1x),Y(set_1x),'.', 'MarkerEdgeColor','k'); hold on
end
% % % % % %


% % % set where whole term >= 0 in l^infty case
% set__x = (lambda*(X-v_alpha(1)).*X.*(r^2-X.^2).^2 + sigma^2/2*(X-v_alpha(1)).^2.*(2*((2*X.^2-r^2).*X.^2)-(r^2-X.^2).^2))./(r^2-X.^2).^4;
% set__y = (lambda*(Y-v_alpha(2)).*Y.*(r^2-Y.^2).^2 + sigma^2/2*(Y-v_alpha(2)).^2.*(2*((2*Y.^2-r^2).*Y.^2)-(r^2-Y.^2).^2))./(r^2-Y.^2).^4;
% 
% set__ = (set__x+set__y) >= 0;
% 
% plot(X(set__),Y(set__),'.', 'MarkerEdgeColor','g'); hold on


% % 
% v_star
p_v_star = plot(0,0, '*', 'MarkerSize', 12, 'LineWidth', 1.8, "color", co(5,:));

% v_alpha
p_v_alpha = plot(v_alpha(1),v_alpha(2), '.', 'MarkerSize', 25, 'LineWidth', 1.8, "color", co(2,:));

xlim(1.2*[-r,r]); ylim(1.2*[-r,r]); %xlim([-2*r,2*r]); ylim([-2*r,2*r]);
ticks = -r:0.5*r:r;
xticks(ticks); yticks(ticks);
ticks_lab = {'-$r$','-0.5$r$','0','0.5$r$','$r$'};
set(groot,'defaultAxesTickLabelInterpreter','latex'); 
xticklabels(ticks_lab); yticklabels(ticks_lab);

legend([p_v_star, p_v_alpha, p_set_11, p_set_21pos, p_set_31], 'Global minimizer $v^*$', 'Consensus point $v_{\alpha}(\rho_t)$','$K_{1k}^c \cap \Omega_r$','$K_{1k} \cap K_{2k}^c \cap \Omega_r$','$K_{2k} \cap K_{2k} \cap \Omega_r$','Location','southwest','Interpreter','latex','FontSize',17)

%tit = ['$\sigma =\; $', num2str(sigma), ', $\|v_{\alpha}\|_2=\; $', num2str(v_alpha_norm)];
%title(tit,'Interpreter','latex','FontSize',16)

ax = gca;
ax.FontSize = 15;

%axis('equal')
%axis('tight')


%% Save Image
if pdfexport
    
    filename = ['Prop19_anisotropic_','v_alpha',num2str(100*v_alpha(1)),'div100', num2str(100*v_alpha(2)),'div100','sigma',num2str(100*sigma),'div100'];
    
    print(f,['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/', filename],'-dpdf');

    % save parameters
    save(['CBOandPSO/EnergyBasedCBOAnalysis/images_videos/', filename], 'sigma', 'v_alpha', 'c')
    
end



