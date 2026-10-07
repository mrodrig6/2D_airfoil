function [Flux,max_wspeed] = roe_test(Ul,Ur,nx,ny,gamma)
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
if (sign(radicand) == -1)
    error 'negative roe-averaged speed of sound'
end

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