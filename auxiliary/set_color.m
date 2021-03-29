% set_color
% 
% This function returns the standard MATLAB colors used for plotting.
% 
% 
% co = set_color()
% 
% output:   co = color matrix
% 

function co = set_color()

co = [0         0.4470    0.7410
      0.8500    0.3250    0.0980
      0.9290    0.6940    0.1250
      0.4940    0.1840    0.5560
      0.4660    0.6740    0.1880
      0.3010    0.7450    0.9330
      0.6350    0.0780    0.1840
      0.6667    0.3333    0
      0.8000    0.8000    0.8000];
set(groot,'defaultAxesColorOrder',co)

end