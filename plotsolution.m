function plotsolution(V,E2N, U, sname, srange, xyrange)
% PURPOSE: Plots cell data U, post-processed to compute scalar sname,
% over mesh V,E2N
%
% INPUTS:
%    V,E2N   : Mesh points and triangles
%    U       : cell-centered solution
%    sname   : name of scalar, see fcns below
%              'mach' -> mach number
%              'entropy' -> entropy
%              'pressure' -> pressure
%    srange  : range for scalar: [min, max]
%    xyrange : window extent: [xmin, xmax, ymin, ymax]
%

%--------------------------------------
% compute desired scalar -> store it in S
smethod = GetScalarMethod(sname);
S = zeros(size(U,1), 1);
for e=1:size(U,1),
  S(e) = ComputeScalar(smethod, U(e,:));
end

%--------------------------------------
% Set colorbar axis, colormap
colormap(jet(2^5));

%--------------------------------------
% plot using pdeplot
t = E2N';
t = [t; ones(1,size(t,2))];
pdeplot(V', [],t,'xydata',S,'xystyle','interp', 'colormap', 'jet');
caxis(srange);
h = colorbar('East');
set(h, 'fontsize', 10);
set(h, 'ytick', linspace(srange(1), srange(end), 5));
set(gcf,'Renderer','zbuffer')

%-------------------------------------
% set zoom
axis equal;
axis(xyrange);
axis off;
AR = (xyrange(2)-xyrange(1))/(xyrange(4)-xyrange(3));
DY = 600; % plot size in y (not relevant for width/height of export)
set(gcf, 'position', [100, 100, DY*AR, DY]);
set(gca, 'position', [0,0,1,1]);



%%%%%%%%%%%%%
% FUNCTIONS %
%%%%%%%%%%%%%

%-------------------------------------------------------
% Computes scalar method: string -> integer for fast access
function smethod = GetScalarMethod(sname)

switch lower(sname)
  case 'mach'
    smethod = 1;
  case 'entropy',
    smethod = 2;
  case 'pressure',
    smethod = 3;
  otherwise
    error 'unsupported scalar';
end


%-------------------------------------------------------
% Computes scalar based on method (stored as an integer)
function scalar = ComputeScalar(smethod, u)

sr = size(u,2);

% Navier-Stokes
gam = 1.4; Rgas = 1;   % Hardcoded
gmi = gam-1;
r  = u(:,1);
ru = u(:,2);
rv = u(:,3);
rE = u(:,sr);

% choose appropriate scalar method
switch smethod
  
  case 1 % mach
    q = sqrt(ru.^2 + rv.^2) ./ r;
    p = gmi * (rE - 0.5*q.*q.*r);
    c = sqrt(gam * p ./ r);
    scalar = q ./ c;
  case 2 % entropy
      q = sqrt(ru.^2 + rv.^2) ./ r;
      p = gmi * (rE - 0.5*q.*q.*r);
    scalar = Rgas/gmi*(log(p) - gam*log(r));
  case 3 % pressure
    q = sqrt(ru.^2 + rv.^2) ./ r;
    scalar = gmi * (rE - 0.5*q.*q.*r);
    
  otherwise
    error 'unsupported scalar computation method'
end


%-------------------------------------------------------
% checks if all points are inside xyrange
function bool = NotInZoom(xyN, xyrange, xyrangepad)

bool = true;
bool = bool && ( (max(xyN(:,1))<xyrange(1)) || (min(xyN(:,1))>xyrange(2)) );
bool = bool && ( (max(xyN(:,2))<xyrange(3)) || (min(xyN(:,2))>xyrange(4)) );

