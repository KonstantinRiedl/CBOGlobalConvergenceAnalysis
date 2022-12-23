% CBOmemorygradient phase transition diagrams for different parameters
%
% This script produces phase diagrams with respect to two selected
% parameters of CBOmemorygradient.
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
[E, gradE, ~, ~, ~] = objective_function(objectivefunction, d, 'CBO');

% global minimizer
xstar = zeros(d,1); %fminbnd(E,xrange_plot(1),xrange_plot(2));


%% Settings for Easy Handling and Notes

% % parameter for comparison (with values)
% (this overwrites the one from below)
pt_diagram_x = 'gamma';
pt_diagram_y = 'N';
pt_diagram_x_values = 0:0.1:4;
pt_diagram_y_values = [2:2:200];


number_runs = 100;


%% Parameters of CBOMemoryGradient Algorithm

% time horizon
T = 20;

% discrete time size
dt = 0.01;

% number of particles
N = 100;

% memory
memory = 1; % 0 or 1
% lambda2, sigma2, kappa and beta have no effect for memory=0.

%
kappa = 1/dt;

% lambda1 (drift towards global and in-time best (consensus) parameter)
lambda1 = 1;
% lambda2 (drift towards in-time best parameter)
lambda2 = 0;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% exploration/noise 1 type
anisotropic1 = 1;
% sigma (exploration/noise parameter 1)
sigma1 = sqrt(1.6);
% exploration/noise 2 type
anisotropic2 = 1;
% sigma (exploration/noise parameter 2)
sigma2 = lambda2*sigma1; %requires to be changed in code below
sigma2_activated = 0;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 10^2;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta


%% Initialization
X0mean = 2*ones(d,1);
X0std = 4;

parametersCBOmemorygradient = containers.Map({'T', 'dt', 'N', 'memory', 'kappa', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'gamma', 'learning_rate', 'alpha', 'beta'},...
                                             {  T,   dt,   N,   memory,   kappa,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   gamma,   learning_rate,   alpha,  beta});
parametersInitialization = containers.Map({'X0mean', 'X0std'},...
                                          {  X0mean,   X0std});


%% Phase Transition Diagram table

pt_diagram = zeros(number_runs, length(pt_diagram_y_values), length(pt_diagram_x_values));


%% CBOMemoryGradient Algorithm

for i = 1:length(pt_diagram_x_values)
    
    % setting parameter of interest
    parametersCBOmemorygradient(pt_diagram_x) = pt_diagram_x_values(i);
    
    for j = 1:length(pt_diagram_y_values)
        
        % setting parameter of interest
        parametersCBOmemorygradient(pt_diagram_y) = pt_diagram_y_values(j);
        if strcmp(pt_diagram_y, 'lambda2') || strcmp(pt_diagram_x, 'lambda2') || strcmp(pt_diagram_y, 'sigma1') || strcmp(pt_diagram_x, 'sigma1') 
            parametersCBOmemorygradient('sigma2') = sigma2_activated*parametersCBOmemorygradient('lambda2')*parametersCBOmemorygradient('sigma1');
        end

        parfor (r = 1:number_runs, number_workers)
            
            %initialization
            X0 = X0mean+X0std*randn(d,parametersCBOmemorygradient('N'));
            
            % CBOmemory
            [xstar_app] = CBOmemorygradient(E, gradE, parametersCBOmemorygradient, X0);
            disp([i,j,r])

            % count successful runs
            if E(xstar_app)<0.8
                pt_diagram(r, j, i) = pt_diagram(r, j, i) + 1;
            end
            %pt_diagram(r, j, i) = pt_diagram(r, j, i) + E(xstar_app);

        end
        
    end
    
end

pt_diagram = reshape(mean(pt_diagram,1), [length(pt_diagram_y_values), length(pt_diagram_x_values)]);


%%

filename = [main_folder(),'/EnergyBasedCBOmemorygradientAnalysis/results/', 'phasediagram_', pt_diagram_x, pt_diagram_y, '_', objectivefunction, 'd', num2str(d) , '_anisotropic', num2str(anisotropic1), num2str(anisotropic2), '_alpha', num2str(alpha), '_beta', num2str(beta), '_N', num2str(N), '_sigma1', num2str(floor(100*sigma1)), 'div100_', 'sigma2activated', num2str(sigma2_activated), '_', datestr(datetime('now'))];


save(filename, 'pt_diagram', 'd', 'objectivefunction', 'pt_diagram_x', 'pt_diagram_y', 'pt_diagram_x_values', 'pt_diagram_y_values', 'number_runs', 'T', 'dt', 'N', 'lambda1', 'lambda2', 'gamma', 'learning_rate', 'anisotropic1', 'anisotropic2', 'sigma1', 'sigma2', 'alpha', 'beta', 'X0mean', 'X0std')

%% Plotting

imagesc(flipud(pt_diagram))
colorbar
xlabel(['$\', pt_diagram_x, '$'],'interpreter','latex')
xticks(1:length(pt_diagram_x_values))
xticklabels(pt_diagram_x_values)
ylabel(['$', pt_diagram_y, '$'],'interpreter','latex')
yticks(1:length(pt_diagram_y_values))
yticklabels(flipud(pt_diagram_y_values')')




