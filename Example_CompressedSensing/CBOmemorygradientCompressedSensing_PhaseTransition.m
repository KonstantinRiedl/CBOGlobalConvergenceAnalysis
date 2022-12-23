% CBOmemorygradient phase transition diagrams for compressed sensing
%
% This script produces phase diagrams with respect to the number of 
% required measurements and one selected parameter of CBOmemorygradient in
% a compressed sensing application. Instead of the number of measurements
% also another parameter is possible.
%

%%
clear; clc; close all;

co = set_color();

number_workers = 8; %25;


%% Energy Function E

% % dimension of the ambient space
d = 200;
% % sparsity of signal
s = 8;

% % energy function E
% original signal / global minimizer
xstar = zeros(d,1);
xstar(randsample(d,s)) = sign(randn(s,1)).*(0.5+rand(s,1));

% number of measurements and noise-to-signal ratio
M = 10*s;
noise_to_signal_ratio = 0; % noise level in measurements (0 to 1)

% E (E is a function mapping columnwise from R^{d\times N} to R^N)
objectivefunction = 'CompressedSensing';
p = 1;


%% Settings for Easy Handling and Notes

% % parameter for comparison (with values)
% (this overwrites the one from below)
pt_diagram_x = 'gamma';
pt_diagram_y = 'M';
pt_diagram_x_values = 0:0.1:4;
pt_diagram_y_values = [1:1:40];


number_runs = 100;


%% Parameters of CBOMemoryGradient Algorithm

% time horizon
T = 20;

% discrete time size
dt = 0.02;

% number of particles
N = 100;

% memory
memory = 1; % 0 or 1
% lambda2, sigma2, kappa and beta have no effect for memory=0.

% kappa
kappa = 1/dt;

% lambda1 (drift towards global and in-time best (consensus) parameter)
lambda1 = 1;
% lambda2 (drift towards in-time best parameter)
lambda2 = 0;
% gamma (parameter of gradient drift term)
gamma = 1;
learning_rate = 1; 
% exploration/noise 1 type
anisotropic1 = 1;
% sigma (exploration/noise parameter 1)
sigma1 = 0*sqrt(1.6);
% exploration/noise 2 type
anisotropic2 = 1;
% sigma (exploration/noise parameter 2)
sigma2 = lambda2*sigma1;

% alpha (weight in Gibbs measure for consensus point computation)
alpha = 100;
% beta (regularization parameter for sigmoid)
beta = -1; % 'inf' or -1 for using Heaviside function instead of S_beta


%% Initialization
X0mean = 0*ones(d,1);
X0std = 1;

parametersCBOmemorygradient = containers.Map({'T', 'dt', 'N', 'memory', 'kappa', 'lambda1', 'lambda2', 'anisotropic1', 'sigma1', 'anisotropic2', 'sigma2', 'gamma', 'learning_rate', 'alpha', 'beta'},...
                                             {  T,   dt,   N,   memory,   kappa,   lambda1,   lambda2,   anisotropic1,   sigma1,   anisotropic2,   sigma2,   gamma,   learning_rate,   alpha,  beta});
parametersInitialization = containers.Map({'X0mean', 'X0std'},...
                                          {  X0mean,   X0std});


%% Phase Transition Diagram table
pt_diagram_success = zeros(number_runs, length(pt_diagram_y_values), length(pt_diagram_x_values));
pt_diagram_error = zeros(number_runs, length(pt_diagram_y_values), length(pt_diagram_x_values));


%% CBOMemoryGradient Algorithm

for i = 1:length(pt_diagram_x_values)
    
    % setting parameter of interest
    parametersCBOmemorygradient(pt_diagram_x) = pt_diagram_x_values(i);
    
    for j = 1:length(pt_diagram_y_values)
        
        % setting parameter of interest
        if strcmp(pt_diagram_y,'M')
            M = pt_diagram_y_values(j);
        else
            parametersCBOmemorygradient(pt_diagram_y) = pt_diagram_y_values(j);
        end
        if strcmp(pt_diagram_y, 'lambda2') || strcmp(pt_diagram_x, 'lambda2') || strcmp(pt_diagram_y, 'sigma1') || strcmp(pt_diagram_x, 'sigma1') 
            parametersCBOmemorygradient('sigma2') = sigma2_activated*parametersCBOmemorygradient('lambda2')*parametersCBOmemorygradient('sigma1');
        end

        parfor (r = 1:number_runs, number_workers)
        %for r = 1:number_runs

            [E, grad_E, A, Y] = objective_function_compressed_sensing(xstar, M, p, noise_to_signal_ratio);

            %initialization
            X0 = X0mean+X0std*randn(d,parametersCBOmemorygradient('N'));
            
            % CBOMemoryGradient
            [xstar_app] = CBOmemorygradient(E, grad_E, parametersCBOmemorygradient, X0);
            
            % post-processing
            xstar_app(abs(xstar_app)<0.01) = 0; % hard threshold
            xstar_app_pp = xstar_app;
            A_support = A(:,xstar_app~=0);
            xstar_app_pp_support = (A_support'*A_support) \ A_support'*Y;
            xstar_app_pp(xstar_app_pp~=0) = xstar_app_pp_support;

            % count successful runs
            error = min(1,min(norm(xstar_app_pp - xstar),norm(xstar_app - xstar))/norm(xstar))
            fprintf('Index %d, %d, %d: %0.5e\n', i,j,r, error)
            if error < 10^-12
                pt_diagram_success(r, j, i) = pt_diagram_success(r, j, i) + 1;
            end
            pt_diagram_error(r, j, i) = pt_diagram_error(r, j, i) + error;

            fprintf('Index %d, %d, %d: %0.5e\n', i,j,r, 0)
        end
        
    end
    
end

pt_diagram_success = reshape(mean(pt_diagram_success,1), [length(pt_diagram_y_values), length(pt_diagram_x_values)]);
pt_diagram_error = reshape(mean(pt_diagram_error,1), [length(pt_diagram_y_values), length(pt_diagram_x_values)]);


%%

filename = [main_folder(),'/Example_CompressedSensing/results/', 'phasediagram_', pt_diagram_x, pt_diagram_y, '_', objectivefunction, '_', 'p', num2str(10*p),'div10_', 'd', num2str(d) , '_anisotropic', num2str(anisotropic1), num2str(anisotropic2), '_alpha', num2str(alpha), '_beta', num2str(beta), '_N', num2str(N), '_sigma1', num2str(floor(100*sigma1)), 'div100_', datestr(datetime('now'))];

save(filename, 'pt_diagram_success', 'pt_diagram_error', 'p', 'd', 'objectivefunction', 'pt_diagram_x', 'pt_diagram_y', 'pt_diagram_x_values', 'pt_diagram_y_values', 'number_runs', 'T', 'dt', 'N', 'lambda1', 'lambda2', 'gamma', 'learning_rate', 'anisotropic1', 'anisotropic2', 'sigma1', 'sigma2', 'alpha', 'beta', 'X0mean', 'X0std')


%% Plotting

imagesc(flipud(pt_diagram_success))
colorbar
xlabel(['$\', pt_diagram_x, '$'],'interpreter','latex')
xticks(1:length(pt_diagram_x_values))
xticklabels(pt_diagram_x_values)
ylabel(['$', pt_diagram_y, '$'],'interpreter','latex')
yticks(1:length(pt_diagram_y_values))
yticklabels(flipud(pt_diagram_y_values')')

