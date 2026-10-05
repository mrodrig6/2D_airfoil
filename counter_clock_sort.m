function [E2N] = counter_clock_sort(E2N,node)
%COUNTER_CLOCK_SORT Subroutine that winds the nodes counter-clockwise
% INPUT : E2N   = [nelem x 3] array mapping triangles to nodes
%         node  = [nnode x 2] array with the number of nodes and x & y
%                 locations in the mesh
% OUTPUT: E2N   = [nelem x 3] array mapping triangles to nodes

% Calculate the number of edges 
nelem = size(E2N,1);
for elem = 1:nelem
    % Looping over the elements and extracting the node numbers
    nodes = E2N(elem,:);
    % Extracting the x and y components of the nodes and labeling the array
    % abc = [x locations, y locations];
    abc = [node(nodes,1), node(nodes,2)];
    % Keeping the first location node constant and calculating the vectors
    % from a -> b = ab and a -> c = ac 
    ab = abc(2,:) - abc(1,:);
    ac = abc(3,:) - abc(1,:);
    % Calculating the normal vector of ab = n_ab
    n_ab = [ab(2), -ab(1)];
    % Calculating the dot product of n_ab and ac, if the dot product is
    % positive the node numbers in E2N must be switched, otherwise the node
    % arrangement is already in counter-clockwise winding
    if dot(n_ab,ac)>0
        node3 = E2N(elem,3);
        E2N(elem,3) = E2N(elem,2);
        E2N(elem,2) = node3;
    end
end
% Output E2N
end