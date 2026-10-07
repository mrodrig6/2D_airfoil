function [U,U_b,gamma,p_inf,alpha] = initialstate(numelem)
%INITIAL_CONDITION Initializing the state vector
% Fully commented code, for questions please email mrdz@umich.edu

% Allocating the state vector
U = zeros(numelem,4); 
% Using the provide mach number and angle of attack for this problem
M_inf = 0.8; alpha = 1.25*(pi/180); gamma = 1.4;
% Non-dimensionalizing the code and considering unitary values for the
% freestream state
rho_inf = 1; v_inf = 1; 
% Pressure and speed of sound of the free-stream calculation
p_inf = (rho_inf/gamma)*(v_inf/M_inf)^2;
Rc = 0.4; T_inf = p_inf/rho_inf/Rc;
c_inf = sqrt(gamma*Rc*T_inf);
% Initializing the state vector as the free-stream state
U(:,1) = rho_inf;
U(:,2) = rho_inf*v_inf*cos(alpha);
U(:,3) = rho_inf*v_inf*sin(alpha);
U(:,4) = p_inf/(gamma-1) + (rho_inf*v_inf^2)/2;
% Storing the free-stream state for c_p calculation
U_b = U(1,:);
end

