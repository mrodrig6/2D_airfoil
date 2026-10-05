function [R,dt] = residual_calc(U,U_b,d_hydraulic,P,NIE,NBE,gamma)
% RESIDUAL_CALC Summary of this function goes here
%   INPUT : U     = [nelem x 4] array with the conservative variables
%           U_b   = [1 x 4] array with freestream convervative variables
%           bnode = [nbedges x 3] array giving (elem, edge, bindex)
%                   information for each boundary edge
%           A     = [nelem x 4] array with the areas of each element in
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
%           gamma = 1.4
%  OUTPUT : R     = [nelem x 4] array containing the flux residual
%           dt    = [nelem x 4] array containing the local time marching
%                   step for every element

% Initializing the flux residual and maximumwave speed arrays
R = zeros(size(U)); s_i = zeros(size(U,1),1);
% Evaluating the total number of interior and boundary edges
niedges = size(NIE,1); nbedges = size(NBE,1);
% Setting CFL number for local time marching
CFL = 0.9;

% Looping over the interior edges to calculate the Roe flux
for iedge = 1:niedges
    % Using the NIE array to obtain the left element, right element, and
    % edge length alread precalculated
    lelem = NIE(iedge,1); relem = NIE(iedge,2); s_edge = NIE(iedge,3);
    % Using the roe_flux subroutine to calculate the F_L, F_R, RHS, and
    % maximum wave speed for each edge
    [Flux,max_wspeed] = roe_flux(U(lelem,:),U(relem,:),NIE(iedge,4),NIE(iedge,5),gamma);
    % Adding flux to the left element and subtracting from the right
    % element
    R(lelem,:) = R(lelem,:) + Flux*s_edge;
    R(relem,:) = R(relem,:) - Flux*s_edge;
    % Calculating the averaged maximum wave speed for the left and right
    % elements
    s_i(lelem) = s_i(lelem) + max_wspeed*s_edge/P(lelem);
    s_i(relem) = s_i(relem) + max_wspeed*s_edge/P(relem);
end

% Looping over the boundary edges to calculate the Roe flux
for bedge = 1:nbedges
    % need to check, -1 is the airfoil, -2 is the far-field
    belem = NBE(bedge,1); s_edge = NBE(bedge,2);
    if (NBE(bedge,5) == -1)
        % this is the airfoil using an inviscid wall bc subroutine to
        % calculate the flux
        [Flux,max_wspeed] = inviscid_wall_bc(U(belem,:),NBE(bedge,3),NBE(bedge,4),gamma);
        % Adding flux to the boundary element
        R(belem,:) = R(belem,:) + Flux*s_edge;
        % Calculating the averaged maximum wave speed for the boundary elem
        s_i(belem) = s_i(belem) + max_wspeed*s_edge/P(belem);
    else if (NBE(bedge,5) == -2)
        % this is the far-field calculation using full-state bc and thus
        % the Roe flux solver to evaluate the flux
        [Flux,max_wspeed] = roe_flux(U(belem,:),U_b,NBE(bedge,3),NBE(bedge,4),gamma);
        % Adding the flux to the boundary element
        R(belem,:) = R(belem,:) + Flux*s_edge;
        % Calculating the averaged maximum wave speed for the boundary elem
        s_i(belem) = s_i(belem) + max_wspeed*s_edge/P(belem);
        else
            sprintf('no bc implemented on boundary edge %f',bedge)
        end
     end
end
% Calculating the dt using the local time marching for this implementation
dt_1 = CFL*d_hydraulic./s_i;
dt = [dt_1,dt_1,dt_1,dt_1];
end


function [Flux,max_wspeed] = roe_flux(Ul,Ur,nx,ny,gamma)
% ROE_FLUX subroutine to calculate the F_L, F_R, RHS, and
% maximum wave speed for each edge
%   INPUT : Ul    = [4 x 1] array with left element variables
%           Ur    = [4 x 1] array with right element variables
%           nx    = [1 x 1] x-normal of the edge
%           ny    = [1 x 1] y-normal of the edge
%           gamma = 1.4
%  OUTPUT : Flux   = [4 x 1] array containing the Roe flux
%           max_wspeed = [1 x 1] maximum wave speed from the eigenvalues

% calculating the coordinate free left element flux
pl = (gamma-1)*(Ul(4)-0.5*(Ul(2)^2+Ul(3)^2)/Ul(1)); Hl = (Ul(4)+pl)/Ul(1);
Fli = [Ul(2), (Ul(2)^2)/Ul(1) + pl, Ul(2)*Ul(3)/Ul(1), Ul(2)*Hl];
Flj = [Ul(3), Ul(2)*Ul(3)/Ul(1), (Ul(3)^2)/Ul(1) + pl, Ul(3)*Hl];
F_L = Fli*nx+Flj*ny;

% calculating the coordinate free right element flux
pr = (gamma-1)*(Ur(4)-0.5*(Ur(2)^2+Ur(3)^2)/Ur(1)); Hr = (Ur(4)+pr)/Ur(1);
Fri = [Ur(2), (Ur(2)^2)/Ur(1) + pr, Ur(2)*Ur(3)/Ur(1), Ur(2)*Hr];
Frj = [Ur(3), Ur(2)*Ur(3)/Ur(1), (Ur(3)^2)/Ur(1) + pr, Ur(3)*Hr];
F_R = Fri*nx+Frj*ny;

% calculating the Roe-averaged state
rhol = sqrt(Ul(1)); rhor = sqrt(Ur(1)); rhosum = rhol+rhor;
u = (rhol*Ul(2)/Ul(1) + rhor*Ur(2)/Ur(1))/rhosum; 
v = (rhol*Ul(3)/Ul(1) + rhor*Ur(3)/Ur(1))/rhosum;
q2 = u^2 + v^2; H = (rhol*Hl + rhor*Hr)/rhosum;

% safety check to prevent code crashing
radicand = (gamma-1)*(H-0.5*q2);
% if (sign(radicand) == -1)
%     error 'negative roe-averaged speed of sound'
% end

% Roe averaged speed of sound
c = sqrt(radicand);

% calculating the wavespeeds and performing the entropy fix
epsilon = 0.01*c; un = u*nx + v*ny; eig = abs([un+c,un-c,un,un]);
max_wspeed = max(eig);
for i = 1:4
    if (eig(i)<epsilon)
        eig(i) = (eig(i)^2 + epsilon^2)/(2*epsilon);
    end
end

% calculating the values for the RHS as seen in page 4 of hand out 2
delr = Ur(1) - Ul(1); delru = Ur(2)-Ul(2) ; 
delrv = Ur(3)-Ul(3); delrE = Ur(4)-Ul(4);
G1 = (gamma-1)*(0.5*q2*delr - u*delru - v*delrv + delrE);
G2 = -un*delr + delru*nx + delrv*ny;
s1 = 0.5*(eig(1) + eig(2)); s2 = 0.5*(eig(1) - eig(2)); 
C1 = (G1/c^2)*(s1-eig(3)) + (G2/c)*s2; C2 = G1*s2/c + (s1-eig(3))*G2;
% compiling results into the RHS see page 3 of handout 2
RHS = [eig(3)*delr + C1;
       eig(3)*delru + C1*u + C2*nx;
       eig(3)*delrv + C1*v + C2*ny;
       eig(3)*delrE + C1*H + C2*un];
% calculates the total flux using the Roe method
Flux = 0.5*(F_L + F_R) - 0.5*RHS';
end

function [Flux,max_wspeed] = inviscid_wall_bc(U_plus,nx,ny,gamma)
% INVISCID_WALL_BC subroutine to calculate the wall boundary condition
%   INPUT : U_plus= [4 x 1] array with boundary element variables
%           nx    = [1 x 1] x-normal of the edge
%           ny    = [1 x 1] y-normal of the edge
%           gamma = 1.4
%  OUTPUT : Flux  = [4 x 1] array with the boundary flux
%           max_wspeed = [1 x 1] array with the maximum wave speed

% extracting the necessary conservative variables
rho_E = U_plus(4); rho_plus = U_plus(1); 

% calculating the velocity tangent to the boundary edge
v_plus = [U_plus(2);U_plus(3)]./rho_plus; n = [nx;ny];  
v_parallel = v_plus - dot(v_plus,n)*n;

% using invisid formulation in page 5 of handout 2 for the boundary
% pressure
p_b = (gamma-1)*(rho_E - 0.5*rho_plus*(v_parallel(1)^2+v_parallel(2)^2));
max_wspeed = sqrt(gamma*p_b/rho_plus);

% evaluating boundary flux
Flux = [0, p_b*nx, p_b*ny, 0];
end
