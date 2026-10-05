function [pnew,enew,tnew,Unew] = adaptmesh(p,e,t,U,tflag)
% PURPOSE: 
%   Isotropically refines triangles in a mesh based on flag and
%   transfers cell-centered solution to new mesh.
% INPUTS:
%   p,e,t : original Matlab mesh structure
%   U     : original solution over triangles, (ntri x sr)
%           where sr = state rank, e.g. 4
%   tflag : list of triangles to refine (column vector)
%
% OUTPUTS:
%   pnew, enew, tnew : refined Matlab mesh structure
%   Unew             : solution transfered to refined mesh
%

% transfer cell data U to nodal data Up via averaging
Up = zeros(length(p), size(U,2));
np = zeros(length(p),1);
for i=1:size(U,1),
  e2n = t(1:3, i);
  for k=1:3,
    Up(e2n(k),:) = Up(e2n(k),:) + U(i,:);
  end
  np(e2n) = np(e2n) + 1;
end
for k = 1:size(U,2),
  Up(:,k) = Up(:,k) ./ np;
end

% load naca geometry
load nacageom;

% use PDE toolbox function "refinemesh"
[pnew,enew,tnew,Upnew] = refinemesh(nacageom, p, e, t, Up, tflag);

% transfer refined Upnew data to Up via averaging
nelem = size(tnew,2);
Unew = zeros(nelem, size(U,2));
for i=1:nelem,
  e2n = tnew(1:3, i);
  Unew(i,:) = sum(Upnew(e2n,:))/3;
end

end