function plotmesh(V,E2N, xyrange)
% PURPOSE: Plots mesh in V,E2N
%
% INPUTS:
%    V,E2N   : Mesh points and triangles
%    xyrange : window extent: [xmin, xmax, ymin, ymax]
%

%--------------------------------------
% plot using pdeplot
t = E2N';
t = [t; ones(1,size(t,2))];
pdeplot(V', [],t,'mesh','on');
%set(gcf,'Renderer','zbuffer')

%-------------------------------------
% set zoom
axis equal;
axis(xyrange);
axis off;
AR = (xyrange(2)-xyrange(1))/(xyrange(4)-xyrange(3));
DY = 600; % plot size in y (not relevant for width/height of export)
set(gcf, 'position', [100, 100, DY*AR, DY]);
set(gca, 'position', [0,0,1,1]);
end