function [ ] = post_process( V , E2N, U, iteration )
%POST_PROCESS Full post-processing subroutine for the mach and mesh plots
%   INPUT : V     = [nnode x 2] array with the number of nodes and x & y
%                   locations in the mesh
%           E2N   = [nelem x 3] array mapping triangles to nodes
%           U     = [nelem x 4] array with the conservative variables
%           iteration = current adaptive mesh iteration number
%   OUTPUT: NONE

% Setting the axis for the plots
xyrange = [-1 2 -1 1];
% Making the strings for the mesh and mach number eps plots
plot1 = sprintf('mesh%d',iteration);
plot2 = sprintf('mach%d.eps',iteration);
% Quickly creating the mesh and mach plots, saving, and then closing the
% plots for processing speed
figure(2)
plotmesh(V,E2N, xyrange);
saveas(gcf,plot1,'eps'); close;
figure(3)
sname = 'mach'; srange = [0 , 1.5];
plotsolution(V,E2N, U, sname, srange, xyrange);
saveas(gcf,plot2, 'psc2'); close;
end

