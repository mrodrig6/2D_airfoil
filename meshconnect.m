function [A,P,d_hydraulic,NIE,NBE] = meshconnect(V,E2N,bnode)
%MESHCONNECT Subroutine preallocating element and edge information for
%computational speed
%   INPUT : V     = [nnode x 2] array with the number of nodes and x & y
%                   locations in the mesh
%           E2N   = [nelem x 3] array mapping triangles to nodes
%           bnode = [nbedges x 3] array giving (elem, edge, bindex)
%                   information for each boundary edge
%   OUTPUT: A     = [nelem x 4] array with the areas of each element in
%                   each respective row, duplicated for simplicity
%           P     = [nelem x 1] array with perimeter of each element
%                   indexed according the element global number
%           d_hydraulic = [nelem x 1] array with the hydraulic diameter for
%                   each element, indexed corresponding to global element #
%           NIE   = [niedge x 5] array containing the [left element, right
%                   element, edge length, n_x, n_y] for the interior edges
%           NBE   = [nbedge x 6] array containg the [boundary element, edge
%                   length, n_x, n_y, boundary index, and average x 
%                   location] for the boundary edges

% Calculate the number elements and initialize A and P array
numelem = size(E2N,1); A = zeros(numelem,1); P = A;
nelem = size(E2N,1);             % number of elements
nnode = max(max(E2N));           % number of nodes
H = sparse(nnode,nnode);         % Create a hash list to identify edges
NIE = zeros(ceil(nelem*3/2), 5); % Initialize the I2E array
niedge = 0;                      % number of interior edges (running total)

% Loop over elements and identify all edges
for elem = 1:nelem
    % Calculate two vectors from the nodes of the element and calculate the
    % area of the element via A = |a x b|
    a = [V(E2N(elem,2),:)' - V(E2N(elem,1),:)'; 0];
    b = [V(E2N(elem,3),:)' - V(E2N(elem,1),:)'; 0]; 
    A(elem,1) = 0.5*norm(cross(a,b));
    % Extract node numbers for the element
    nv = E2N(elem,:); 
    for edge = 1:3
        n1 = nv(mod(edge  ,3)+1);
        n2 = nv(mod(edge+1,3)+1);
        % n1 & n2 pair form the edge of the node nv
        if (H(n1,n2) == 0) % edge hit for the first time
            % could be a boundary or interior; assume boundary
            H(n1,n2) = elem;  H(n2,n1) = elem;
        else % this is an interior edge, hit for the second time
            oldelem = H(n1,n2);
            if (oldelem < 0)
                error 'Mesh input error'; 
            end
            % update the total number of interior edges
            niedge = niedge+1; 
            % obtain the indices that create edge from the local edge
            % number
            edgenum1 = mod(edge,3)+1; edgenum2 = mod(edge+1,3)+1;
            edge_x = V(E2N(elem,edgenum2),1) - V(E2N(elem,edgenum1),1);
            edge_y = V(E2N(elem,edgenum2),2) - V(E2N(elem,edgenum1),2);
            % calculate the edge length using the deltas x and y
            elength = sqrt(edge_x^2 + edge_y^2);
            % calculate the perimeter by summing the edges lengths
            P(elem) = P(elem) + elength; P(oldelem) = P(oldelem) + elength;
            % consider the current element as the left element to the edge
            % and the previous element as the right element and using the
            % counterclock-wise organization, the normal to the edge is
            % straightfoward to calculate
            NIE(niedge,:) = [oldelem, elem, elength, -edge_y/elength, edge_x/elength];
            H(n1,n2) = -1;  H(n2,n1) = -1;
        end
    end
end
% Store the A array for the state variable calculation ease
A = [A,A,A,A];  
NIE = NIE(1:niedge,:);        % clip NIE
% find boundary element numbers and store them in NBE
[n1,n2,~] = find(triu(H)>0); nbnodes = size(n1,1);
belem  = diag(H(n1,n2));
NBE = [belem, zeros(nbnodes,5)];

% Loop over the boundary nodes to create NBE
for node = 1:nbnodes
    % loop over all of the bnodes, guess that the local edge number is one
    % and modify by using a while loop over E2N and store in B2E
    bedge = 1;
    while (E2N(belem(node),bedge) == n1(node)) ||(E2N(belem(node),bedge) == n2(node))
        bedge = bedge + 1;
    end
    % if local edge number surpasses 3 print an error
    if (bedge > 3)
        error 'local edge error'; 
    end
    % loop over the boundary edges and calculate neighboring local edges
    % and thus find the boundary element
    edgenum1 = mod(bedge,3)+1; edgenum2 = mod(bedge+1,3)+1;
    % calculate delta x and delta y of the nodes making the edge as well as
    % the average x location of the edge
    edge_x = V(E2N(belem(node),edgenum2),1) - V(E2N(belem(node),edgenum1),1);
    edge_y = V(E2N(belem(node),edgenum2),2) - V(E2N(belem(node),edgenum1),2);
    ave_x = (V(E2N(belem(node),edgenum2),1) + V(E2N(belem(node),edgenum1),1))/2;
    % calculate the edge length as well as sum the boundary edge to the
    % total perimeter
    elength = sqrt(edge_x^2 + edge_y^2);
    P(belem(node)) = P(belem(node)) + elength;
    % find the boundary nodes index in bnode array for the edge's nodes
    [x1,~] = find (n1(node) == bnode(:,1));
    [x2,~] = find (n2(node) == bnode(:,1));
    % guess that the bindex is in the index of the first node if the first
    % node has two indexes then use the index of the second node to store
    % the correct bindex in B2E
    bindex = bnode(x1(1),2);
    if ( size(x1,1) ~= 1 )
        bindex = bnode(x2(1),2);
    end
    % given the counter-clockwise winding of the nodes the normal to the
    % edge vector for the boundary element is a straighforward calculation
    % and the NBE structure can be stored
    NBE(node,2:6) = [elength, edge_y/elength, -edge_x/elength, bindex, ave_x];
end
NBE = full(NBE); % make matrix of B2E
% having calculated the area and perimeter, the hydraulic diameter can now
% be calculated
d_hydraulic = 2*A(:,1)./P;
end