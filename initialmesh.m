function [p,e,t] = initialmesh()
% returns Matlab mesh structure for provided NACA geometry
% Note, nacageom.mat must be in the same directory

% load naca geometry, a Matlab decomposed geometry matrix
load nacageom.mat

% make initial mesh using PDE toolbox
[p,e,t] = initmesh(nacageom);

end