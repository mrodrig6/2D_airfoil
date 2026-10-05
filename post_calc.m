function [tflag,cl,cd] = post_calc(U,U_b,p_inf,alpha,NIE,NBE,iteration)
%POST_CALC Post-processing subroutine to calculate element flags and lift,
% drag and pressure coefficients
%   INPUT : U     = [nelem x 4] array with the conservative variables
%           U_b   = [1 x 4] array with freestream convervative variables
%           p_inf = [1 x 1] freestream pressure
%           NIE   = [niedge x 5] array containing the [left element, right
%                   element, edge length, n_x, n_y] for the interior edges
%           NBE   = [nbedge x 6] array containg the [boundary element, edge
%                   length, n_x, n_y, boundary index, and average x 
%                   location] for the boundary edges
%           iteration = current adaptive mesh iteration number
%   OUTPUT: tflag = [? x 1] array containing the global element numbers
%                   that require refinement
%           cl    = lift coefficient for this adaptive mesh
%           cd    = drag coefficient for this adaptive mesh

% extracting the conservative variables from the U array 
gamma = 1.4; gmi = gamma-1; r  = U(:,1); ru = U(:,2); rv = U(:,3); rE = U(:,4);
q = sqrt(ru.^2 + rv.^2) ./ r; p = gmi * (rE - 0.5*q.*q.*r);
% calculating element speed of sound and mach numbers
c = sqrt(gamma * p ./ r); mach = q ./ c; 
% initializing the error indicator as defined on page 2 of project handout
error_k = zeros(size(U,1),1);

% using the mach number jumps across elements as the indicator for
% mesh refinement
lelem = NIE(:,1); relem = NIE(:,2); s_iedge = NIE(:,3); niedge = size(NIE,1);
for i = 1:niedge
    error_k(lelem(i)) = error_k(lelem(i)) + abs(mach(lelem(i)) - mach(relem(i))).*s_iedge(i);
    error_k(relem(i)) = error_k(relem(i)) + abs(mach(relem(i)) - mach(lelem(i))).*s_iedge(i);
end


% finding the indices of NBE of the airfoil boundary elements
[bedge1,~] = find((NBE(:,5) == -1));
% extracting the boundary elements, edge lengths, and normals
belem = NBE(bedge1,1); s_edge = NBE(bedge1,2); n = NBE(bedge1,3:4);
nbelem = size(belem,1);
% calculating the velocity at the boundary element
v_plus = [ru(belem)./r(belem),rv(belem)./r(belem)];
% calculating the velocity normal to the edge
v_normal = (n(:,1).*ru(belem)+n(:,2).*rv(belem))./r(belem); 
% calculating the velocity tangent to the edge
v_parallel = v_plus - [v_normal.*n(:,1) v_normal.*n(:,2)];

% calculating the pressure, speed of sound, and mach number at the 
% boundary element and adding this to the error indicator
q_b = sqrt(v_parallel(:,1).^2 + v_parallel(:,2).^2); 
p_b = gmi*(rE(belem) - 0.5*r(belem).*q_b.*q_b);
c_b = sqrt(gamma * p_b ./ r(belem)); mach_normal = v_normal ./ c_b;

for i = 1:nbelem
    error_k(belem(i)) = error_k(belem(i)) + abs(mach_normal(i)).*s_edge(i);
end

% setting the fraction refinement and then setting the global element
% numbers that require refinement to the tflag array
f = 0.05; refine = (size(U,1) - round(f*size(U,1))) + 1;
[~,flag] = sortrows(error_k); tflag = flag(refine:end);

% calculating the free stream force and dynamic pressure
rho_inf = U_b(1); v_inf = sqrt((U_b(2)^2 + U_b(3)^2)/U_b(1)); 
mom_inf = 0.5*rho_inf*v_inf^2; 

% horizontal force, drag force/unit length
D = -sum(p_b.*sign(n(:,1)).*s_edge)*sin(alpha);
% vertical force, lift force/unit length
L = sum(p_b.*sign(n(:,2)).*s_edge)*cos(alpha);
% evaluating the lift and drag coefficients
cl = L/mom_inf; cd = D/mom_inf;

% plotting the pressure coeffient distributions along x of the airfoil
% every 2 iterations
if (mod(iteration,2) == 0)
    figure(1); hold on;
    ave_x = NBE(bedge1,6);
    cp = (p_inf-p_b)./mom_inf;
    plot(ave_x,cp,'.','color',[rand rand rand]);
end

end