function [V, E2N, BN] = convertmesh(p,e,t)
% Converts Matlab mesh format (p,e,t) to:
%   V   : nnode x 2 matrix of node coordinates
%   E2N : nelem x 3 matrix of triangle node numbers for each element
%   BN  : ?? x 2 matrix; each row contains a node number of a 
%         boundary node and the associated boundary index.  Note,
%         the boundary index can be:
%          -1 means airfoil
%          -2 means farfield
%
% This function is specific to the provided NACA case

% node coordinates are already in p (transposed)
V = p';

% triangle node numbers are in t (transposed)
E2N = t(1:3,:)';

% all boundary nodes appear in the first two rows of e
B = union(e(1,:), e(2,:));
r = sqrt(V(B,1).^2 + V(B,2).^2);
I = find(r<5); % assume airfoil is within circle of radius 5
BN = [B', -2*ones(size(B'))];
BN(I,2) = -1;

end