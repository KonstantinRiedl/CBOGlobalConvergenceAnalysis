% Objective-function function
%
% This function returns the objective function E as an anonymous function
% together with a set of necessary parameters (parametersE) for plotting. 
% Additionally, the function returns two further set of parameters which
% are suitable parameters for CBO (parametersCBO) as well as a suitable 
% initialization (parametersInitialization).
%
% The function E maps columnwise from R^{d\times N} to R, i.e., for a 
% matrixV in R^{d\times N} the function is applied to every column and 
% returns a row vector in R^N (matrix in R^{1\times N})
% 
% 
% [E, parametersE, parametersCBO, parametersInitialization] = objective_function(name, d)
% 
% input:    name          = name of objective function E
%                           (e.g., Rastrigin, Ackley, Wshaped)
%           d             = ambient dimension of the optimization problem
%           
% output:   E             = objective function E (as anonymous function)
%           parametersE   = necessary parameters for plotting of E
%                         = [xrange_plot,yrange_plot,zrange_plot]
%               - *range_plot = range of *coordinate as column vector
%           parametersCBO = suitable parameters for CBO
%                         = [T, dt, N, lambda, gamma, learning_rate, sigma, alpha]
%               - T       = time horizon
%               - dt      = time step size
%               - N       = number of particles
%               - lambda  = consensus drift parameter
%               - gamma   = gradient drift parameter
%               - l._r.   = learning rate associated with gradient drift
%               - sigma   = exploration/noise parameter
%               - alpha   = weight/temperature parameter alpha
%           parametersInitialization = suitable parameters for initialization
%                         = [V0mean, V0std]
%               - V0mean  = mean of initial distribution
%               - V0std   = standard deviation of initial distribution
%

function [E, parametersE, parametersCBO, parametersInitialization] = objective_function(name, d)

% standard domain parameters
xrange_plot = [-5;10];
yrange_plot = xrange_plot;
zrange_plot = [0;100];

% standard CBO parameters
if d == 1
    parametersCBO = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma'},...
                                   {  4, 0.04, 100,   10^15,        1,       0,            0.01,     0.1});
    parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                              {       3,       2});
elseif d == 2
    parametersCBO = containers.Map({'T', 'dt', 'N', 'alpha', 'lambda', 'gamma', 'learning_rate', 'sigma'},...
                                   {  4, 0.02, 100,   10^15,        1,       0,            0.01,     0.1});
    parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                              {   [4;6],      8});
else
    parametersCBO = [];
    parametersInitialization = [];
end

if strcmp(name,'Ackley')
    E = @(v) -20*exp(-0.2*sqrt(1/d*sum(v.*v,1)))-exp(1/d*sum(cos(2*pi*v),1))+exp(1)+20;
    xrange_plot = [-5;10];
    parametersE = [xrange_plot*ones(1,d), [0;20]];
    parametersCBO = parametersCBO;
    parametersInitialization = parametersInitialization;
elseif strcmp(name,'Rastrigin')
    E = @(v) sum(v.*v,1) + 2.5*sum(1-cos(2*pi*v),1);
    xrange_plot = [-2.5;5];
    parametersE = [xrange_plot*ones(1,d), (1*[0;25]+(d-1)*[0;25])]; %(1*[0;25]+(d-1)*[0;30]) for flat 2d plot
    parametersCBO = parametersCBO;
    parametersInitialization = parametersInitialization;
elseif strcmp(name,'Quadratic')
    E = @(v) sum(v.*v,1);
    xrange_plot = [-6;8];
    parametersE = [xrange_plot*ones(1,d), d^2*[0;50]];
    parametersCBO = parametersCBO;
    parametersInitialization = parametersInitialization;
elseif strcmp(name,'Himmelblau')
    error('Not implemented.')
    if d~=2
        error('Dimension Error. The Himmelblau function is only available in 2d.')
    else
        E = @(v) (v(1,:).^2+v(2,:)-11).^2+(v(1,:)+v(2,:).^2-7).^2;
        parametersE = [xrange_plot*ones(1,d), [0;100]];
        parametersCBO = parametersCBO;
        parametersInitialization = parametersInitialization;
    end
elseif strcmp(name,'Wshaped')
    if d>2
        error('Dimension Error. The Wshaped function is only available in 1d or 2d.')
    elseif d==1
        E = @(v) v.^2.*((v-4).^2+1);
        parametersE = [[-2;6],[0;60]];
        parametersCBO = parametersCBO;
    elseif d==2
        E = @(v) 0.1*(v(1,:).^2.*((v(1,:)-4).^2+1)+10*(v(2,:)).^2);
        parametersE = [[-2;6],[-4;4],[0;30]];
        parametersCBO = parametersCBO;
        parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                                  {   [4;2],       4});
    end
elseif strcmp(name,'Wshaped2')
    if d>2
        error('Dimension Error. The Wshaped2 function is only available in 1d or 2d.')
    elseif d==1
        [E, parametersE, parametersCBO, parametersInitialization] = objective_function('Wshaped',d);
    elseif d==2
        E = @(v) 0.1*(v(1,:).^2.*((v(1,:)-4).^2+1)+v(2,:).^2.*((v(2,:)-4).^2+1));
        parametersE = [[-2;6],[-2;6],[0;30]];
        parametersCBO = parametersCBO;
        parametersInitialization = containers.Map({'V0mean', 'V0std'},...
                                                  {   [3;3],       4});
    end
else
    disp('This objective function is not implemented yet.')

end
