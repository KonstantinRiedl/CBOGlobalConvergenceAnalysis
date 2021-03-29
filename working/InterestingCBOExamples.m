% Interesting CBO examples
%
% This file is a collection of interesting parameter settings for CBO.
%

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initially increasing variance, but monotonically decreasing J
while 1
%% Energy Function E

% % dimension of the ambient space
d = 1; % at the moment only 1d due to the convex envelope computation

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R, i.e.,
% elementwise in the setting d=1)

% lopsided W-shaped function in 1d
E = @(v) v.^2.*((v-4).^2+1);

% range of v (and v and y for plotting)
vrange_plot = 5*[-1,2];
yrange_plot = [0,100];
vrange = 100*vrange_plot;


%% Parameters of CBO Algorithm
 
% discrete time size
dt = 0.01;
 
% number of particles
N = 20000;
 
% lambda (parameter of consensus drift term)
lambda = 4;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 0.5;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 40;
 
 
%% Initialization
V0 = 4+0.25*randn(d,N-N/200);
V00 = -1+0.25*randn(d,N/200);
V = [V0 V00];
% alternatively:
%gm = gmdistribution([4;-1],0.1,[1-1/200;1/200]);
%[V,~] = random(gm,N);V=V';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Slightly initially increasing variance, but monotonically decreasing J
while 1
%% Energy Function E

% % dimension of the ambient space
d = 1; % at the moment only 1d due to the convex envelope computation

% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R, i.e.,
% elementwise in the setting d=1)

% lopsided W-shaped function in 1d
E = @(v) v.^2.*((v-4).^2+1);

% range of v (and v and y for plotting)
vrange_plot = 5*[-1,2];
yrange_plot = [0,100];
vrange = 100*vrange_plot;


%% Parameters of CBO Algorithm
 
% discrete time size
dt = 0.01;
 
% number of particles
N = 20000;
 
% lambda (parameter of consensus drift term)
lambda = 10;
% gamma (parameter of gradient drift term)
gamma = 0;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 1.75;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 40;
 
 
%% Initialization
V0 = 4+1.2*randn(d,N);
V = V0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initially increasing variance and initially increasing J (due to large 
% noise term, effect disappears as soon as consensus drift term dominates)
while 1
%% Parameters of CBO Algorithm
 
% discrete time size
dt = 0.01;
 
% number of particles
N = 20000;
 
% lambda (parameter of consensus drift term)
lambda = 10;
% gamma (parameter of gradient drift term)
gamma = 1;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 5;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 100;
 
 
%% Initialization
V = 8+2*randn(d,N);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparison of performance with and without gradient drift term:
%   with gradient drift (gamma = 20, l_r = 0.01): 1.628849e-04
%   with gradient drift (gamma = 40, l_r = 0.01): 4.285573e-07
%   without gradient drift (gamma = 0): 7.050065e-02
while 1
%% Energy Function E

% % dimension of the ambient space
d = 1; % at the moment only 1d due to the convex envelope computation


% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R, i.e.,
% elementwise in the setting d=1)

% lopsided W-shaped function in 1d
E = @(v) v.^2.*((v-4).^2+1);
% Ackley function in 1d
%E = @(v) max(-20*exp(-0.2*abs(v))-exp(cos(2*pi*v))+20+exp(1),10^-12*v.^8);
% Rastringin function in 1d
%E = @(v) 10+v.^2-10.*cos(2*pi*v);


% range of v (and v and y for plotting)
vrange_plot = 5*[-1,2];
yrange_plot = [0,100];
vrange = 100*vrange_plot;


%% Parameters of CBO Algorithm
 
% discrete time size
dt = 0.01;
 
% number of particles
N = 20000;
 
% lambda (parameter of consensus drift term)
lambda = 20;
% gamma (parameter of gradient drift term)
gamma = 20/0;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 2;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 20;
 
 
%% Initialization
V = 4+2*randn(d,N);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Global minimization of the Rastringin function with very few particles
% and suboptimal initialization thanks to lots of noise 
while 1
%% Energy Function E

% % dimension of the ambient space
d = 1; % at the moment only 1d due to the convex envelope computation


% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R, i.e.,
% elementwise in the setting d=1)

% Rastringin function in 1d
E = @(v) 10+v.^2-10.*cos(2*pi*v);


% range of v (and v and y for plotting)
vrange_plot = 5*[-1,2];
yrange_plot = [0,100];
vrange = 100*vrange_plot;


%% Parameters of CBO Algorithm
 
% discrete time size
dt = 0.01;
 
% number of particles
N = 10;
 
% lambda (parameter of consensus drift term)
lambda = 5;
% gamma (parameter of gradient drift term)
gamma = 1;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 10;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 100;
 
 
%% Initialization
V = 8+2*randn(d,N);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% general initial setting
while 1
%% Energy Function E

% % dimension of the ambient space
d = 1; % at the moment only 1d due to the convex envelope computation


% % energy function E
% (E is a function mapping columnwise from R^{d\times N} to R, i.e.,
% elementwise in the setting d=1)

% lopsided W-shaped function in 1d
E = @(v) v.^2.*((v-4).^2+1);
% Ackley function in 1d
%E = @(v) max(-20*exp(-0.2*abs(v))-exp(cos(2*pi*v))+20+exp(1),10^-12*v.^8);
% Rastringin function in 1d
%E = @(v) 10+v.^2-10.*cos(2*pi*v);


%E = @(v) v.^2.*(v-2).^2.*(v-4).^2+10*(1-exp(-1/10*v.^2))+10*(1-exp(-1/10*v.^2));

%E = @(v) (v<1).*v.^2.*(v-2).^2+(1<=v) + (v>2.5).*(3*(v-2).^2.*(v-3).^2.*(v-4).^2-0.4);

%E = @(v) 1/10*((v<2/sqrt(3)&v>-2/sqrt(3)).*v.^2.*(v-2).^2.*(v+2).^2 + 256/27*(v>2/sqrt(3)&v<2.3806) + 256/27*(v<-2/sqrt(3)&v>-2.3806) +(v<-2.3806).*(256/27+20*(-0.207725+(v+2).^2.*(v+3).^2.*(v+4).^2 + 1/10*(v+3))) +(v>2.3806).*(256/27+20*(-0.207725+(v-2).^2.*(v-3).^2.*(v-4).^2 - 1/10*(v-3))));



%fplot(E, 5*[-1,2], "color", co(1,:), 'LineWidth', 2);
%ylim([0 10])

%return

% range of v (and v and y for plotting)
vrange_plot = 5*[-1,2];
yrange_plot = [0,100];
vrange = 100*vrange_plot;

%vrange_plot = 5*[-1,1];
%yrange_plot = [0,5];
%vrange = 100*vrange_plot;


%% Parameters of CBO Algorithm
 
% discrete time size
dt = 0.01;
 
% number of particles
N = 20000;
 
% lambda (parameter of consensus drift term)
lambda = 20;
% gamma (parameter of gradient drift term)
gamma = 20;
learning_rate = 0.01;
% sigma (parameter of exploration term)
sigma = 2;
 
% alpha (weight in Gibbs measure for consensus point computation)
alpha = 20;
 
 
%% Initialization
% V0 = 4+0.25*randn(d,N-N/200);
% V00 = -1+0.25*randn(d,N/200);
% V = [V0 V00];
% 
% V0 = 3+0.25*randn(d,N-N/100);
% V00 = -1+0.25*randn(d,N/100);
% V = [V0 V00];
% 
% V1 = 3+0.005*randn(d,N/2);
% V2 = -3+0.1*randn(d,N/2);
% V = [V1 V2];

V = 4+2*randn(d,N);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



