% CBO phase transition diagrams for different parameters
%
% This script produces phase diagrams with respect to two selected
% parameters of CBO with and without truncated diffusion.
%

%%
clear; clc; close all;

co = set_color();

number_workers = 8; %25;


%% Energy Function E

% % dimension of the ambient space
d = 4;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'Rastrigin';
[E, grad_E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'CBO');

% global minimizer
vstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Settings for Easy Handling and Notes

% % parameter for comparison (with values)
% (this overwrites the one from below)
pt_diagram_x = 'M';
pt_diagram_y = 'sigma';
pt_diagram_x_values = [0:0.5:40,Inf];%[2.^[0:1:10],Inf];
pt_diagram_y_values = 0:0.05:4;
  
number_runs = 100;


%% Parameters of CBO Algorithm

% time horizon
T = 50;

% discrete time size
dt = 0.01;

% number of particles
N = 100;

% lambda (parameter of consensus drift term)
lambda = 1;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% type of diffusion
anisotropic = 0;
% sigma (parameter of exploration term)
sigma = sqrt(0.8);
% truncation parameter of CBO
M = Inf;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^5;


%% Initialization
V0mean = 1*ones(d,1);
V0std = sqrt(2000);

parametersCBOtruncateddiffusion = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma', 'M'},...
                                                 {  T,   dt,   N,   alpha,   lambda,  gamma,    learning_rate,   anisotropic,   sigma,   M});
parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                          {  V0mean,   V0std});


%% Phase Transition Diagram table
pt_diagram = zeros(number_runs, length(pt_diagram_y_values), length(pt_diagram_x_values));


%% CBO Algorithm

for i = 1:length(pt_diagram_x_values)
    
    % setting parameter of interest
    parametersCBOtruncateddiffusion(pt_diagram_x) = pt_diagram_x_values(i);
    
    for j = 1:length(pt_diagram_y_values)
        
        % setting parameter of interest
        parametersCBOtruncateddiffusion(pt_diagram_y) = pt_diagram_y_values(j);
        
        parfor (r = 1:number_runs, number_workers)
            
            %initialization
            V0 = V0mean+V0std*randn(d,parametersCBOtruncateddiffusion('N'));
            V = V0;
            
            % CBO with truncated diffusion
            [vstar_app] = CBOtruncateddiffusion(E, grad_E, parametersCBOtruncateddiffusion, V0);
            disp([i,j,r])
            
            % count successful runs
            if E(vstar_app)<0.1
                pt_diagram(r, j, i) = pt_diagram(r, j, i)+1;
            end

        end
        
    end
    
end

pt_diagram = reshape(mean(pt_diagram,1), [length(pt_diagram_y_values), length(pt_diagram_x_values)]);


%%

filename = [main_folder(),'/EnergyBasedCBOtruncateddiffusionAnalysis/results/', 'phasediagram_', pt_diagram_x, pt_diagram_y, '_', objectivefunction, 'd', num2str(d) , '_anisotropic', num2str(anisotropic), '_alpha', num2str(alpha), '_N', num2str(N), '_', datestr(datetime('now'))];

save(filename, 'pt_diagram', 'd', 'objectivefunction', 'pt_diagram_x', 'pt_diagram_y', 'pt_diagram_x_values', 'pt_diagram_y_values', 'number_runs', 'T', 'dt', 'N', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma', 'alpha', 'V0mean', 'V0std')

%% Plotting

figure()
imagesc(flipud(pt_diagram))
colorbar
caxis([0, 1]);
xlabel(['$', pt_diagram_x, '$'],'interpreter','latex')
xticks(1:length(pt_diagram_x_values))
xticklabels(pt_diagram_x_values)
ylabel(['$\', pt_diagram_y, '$'],'interpreter','latex')
yticks(1:length(pt_diagram_y_values))
yticklabels(flipud(pt_diagram_y_values')')

