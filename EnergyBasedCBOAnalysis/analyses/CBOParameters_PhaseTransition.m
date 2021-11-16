% CBO phase transition diagrams for different parameters
%
% This script produces phase diagrams with respect to two selected
% parameters of CBO.
%

%%
clear; clc; close all;

co = set_color();

number_workers = 4; %25;


%% Energy Function E

% % dimension of the ambient space
d = 8;

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R)
objectivefunction = 'RastriginNonSeparable2';
[E, parametersE, ~, ~] = objective_function(objectivefunction, d, 'PSO');

% global minimizer
vstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Settings for Easy Handling and Notes

% % parameter for comparison (with values)
% (this overwrites the one from below)
pt_diagram_x = 'alpha';
pt_diagram_y = 'N';
pt_diagram_x_values = [1,2,4,10,16,32,50,10^2,10^3,10^4,10^5,10^6];
pt_diagram_y_values = [5:20:200];


number_runs = 16; %100;


%% Parameters of PSO Algorithm

% time horizon
T = 20;

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
anisotropic = 1;
% sigma (parameter of exploration term)
sigma = 8;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 100;


%% Initialization
V0mean = 2*ones(d,1);
V0std = 4;

parametersCBO = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma'},...
                               {  T,   dt,   N,   alpha,   lambda,  gamma,    learning_rate,   anisotropic, sigma});
parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                          {  V0mean,   V0std});


%% Phase Transition Diagram table
pt_diagram = zeros(number_runs, length(pt_diagram_y_values), length(pt_diagram_x_values));


%% PSO Algorithm

for i = 1:length(pt_diagram_x_values)
    
    % setting parameter of interest
    parametersCBO(pt_diagram_x) = pt_diagram_x_values(i);
    if strcmp(pt_diagram_y, 'lambda1')
        parametersCBO('sigma1') = parametersCBO('lambda1')*parametersCBO('sigma2');
    end
    
    for j = 1:length(pt_diagram_y_values)
        
        % setting parameter of interest
        parametersCBO(pt_diagram_y) = pt_diagram_y_values(j);
        if strcmp(pt_diagram_y, 'sigma2')
            parametersCBO('sigma1') = parametersCBO('lambda1')*parametersCBO('sigma2');
        end
        
        parfor (r = 1:number_runs, number_workers)
            
            %initialization
            V0 = V0mean+V0std*randn(d,parametersCBO('N'));
            V = V0;
            
            % PSO
            [xstar_app] = CBO(E, parametersCBO, V0);

            % count successful runs
            if E(xstar_app)<0.8
                pt_diagram(r, j, i) = pt_diagram(r, j, i)+1;
            end

        end
        
    end
    
end

pt_diagram = reshape(mean(pt_diagram,1), [length(pt_diagram_y_values), length(pt_diagram_x_values)]);


%%

filename = ['CBOandPSO/EnergyBasedCBOAnalysis/results/', 'phasediagram', pt_diagram_x, pt_diagram_y, '_', datestr(datetime('today'))];

save(filename, 'pt_diagram', 'd', 'objectivefunction', 'pt_diagram_x', 'pt_diagram_y', 'pt_diagram_x_values', 'pt_diagram_y_values', 'number_runs', 'T', 'dt', 'N', 'lambda', 'gamma', 'learning_rate', 'anisotropic', 'sigma', 'alpha', 'V0mean', 'V0std')

%% Plotting

imagesc(flipud(pt_diagram))
colorbar
xlabel(['$', pt_diagram_x, '$'],'interpreter','latex')
xticks(1:length(pt_diagram_x_values))
xticklabels(pt_diagram_x_values)
ylabel(['$', pt_diagram_y, '$'],'interpreter','latex')
yticks(1:length(pt_diagram_y_values))
yticklabels(flipud(pt_diagram_y_values')')




