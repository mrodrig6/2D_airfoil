function [ ] = main( )
% MAIN Primary routine calling subroutines and doing the time marching
% Fully commented code, for questions please email mrdz@umich.edu
% Note: I have added a roe_flux test code to check that my code does run
% the Roe solver correctly to make grading easier, it is called roe_test.m
clear all; close all; clc;
% Initialize the mesh and convert the mesh to create the E2N array
[p,e,t] = initialmesh(); [~,E2N,~] = convertmesh(p,e,t);
% Initialize the state vector array and the free-stream information
[U,U_b,gamma,p_inf,alpha] = initialstate(size(E2N,1)); residual_vec = [];
% Initialize the cd and cl vectors for storate at each adaptive iteration
cd_vec = []; cl_vec = []; iter = 0;
% Start for loop for the ten adaptive mesh iterations
for iteration = 1:3
    % Convert the mesh after the mesh adaptation after the last iteration
    [V,E2N,BN] = convertmesh(p,e,t); 
    % Sort the node numbers in counterclock-wise
    [E2N] = counter_clock_sort(E2N,V);
    % For speed, preallocate the element areas, perimeters, hydraulic
    % diameters, and edge normals, lengths, and adjacent element global #s
    [A,P,d_hydraulic,NIE,NBE] = meshconnect(V,E2N,BN);
    % Implement while loop that initialize residual norm for residual
    % storage and threshold condition
    residual_norm = 1;
    while (residual_norm > 1e-2)
        % Calculate the residual flux vector using the Roe Flux
        [R,dt] = residual_calc(U,U_b,d_hydraulic,P,NIE,NBE,gamma);
        % Calculate the residual for the state vector and update the state
        % vector using first order Euler time-marching and areas of the
        % elements
        U = U - dt.*R./A; iter = iter+1;
        % Calculate the Linf norm of the residual calculate relative error of
        % the residual for the threshold
        residual_norm = norm(R,Inf); 
        if (mod(iter,50) == 0)
            fprintf(1,'residual error = %e , iter = %i\n',residual_norm,iter);
        end
        residual_vec = [residual_vec; norm(R,2)];
    end
    % Post-processing and calculating the flags for the mesh adaptation and
    % lift and drag coefficients for storage for this adaptive iteration
    [tflag,cl,cd] = post_calc(U,U_b,p_inf,alpha,NIE,NBE,iteration);
    cl_vec = [cl_vec;cl]; cd_vec = [cd_vec;cd];
    % Post-processing step for mesh and mach number distribution plots
    post_process( V , E2N, U, iteration)
    fprintf(1,'iteration = %i , nelem = %i , cl = %e, cd = %e\n',iteration+1,...
        size(E2N,1),cl,cd);
    % Mesh adaptation subroutine
    [p,e,t,U] = adaptmesh(p,e,t,U,tflag);
end
% Saving final variables and post-processing of the drag and lift 
% coefficients and the residual values
figure(1)
xlabel('x/c','Interpreter','Latex','FontSize',14);
ylabel('-$c_p$','Interpreter','Latex','FontSize',14);
set(gca,'FontName','Times New Roman','FontSize',12);
L = legend('iter 2','iter 4','iter 6','iter 8','iter 10');
set(L,'Interpreter','Latex','FontSize',10);
saveas(gcf,'cp','psc2');
figure(4)
plot(cl_vec,'-k','LineWidth',1.5)
xlabel('adaptive iterations','Interpreter','Latex','FontSize',14);
ylabel('$C_L$','Interpreter','Latex','FontSize',14);
set(gca,'FontName','Times New Roman','FontSize',12);
saveas(gcf,'cl','eps');
figure(5)
plot(cd_vec,'-k','LineWidth',1.5)
xlabel('adaptive iterations','Interpreter','Latex','FontSize',14);
ylabel('$C_D$','Interpreter','Latex','FontSize',14);
set(gca,'FontName','Times New Roman','FontSize',12);
saveas(gcf,'cd','eps');
figure(6)
semilogy(residual_vec,'-k','LineWidth',1.5)
xlabel('total iterations','Interpreter','Latex','FontSize',14);
ylabel('$L_2$ norm of the residual','Interpreter','Latex','FontSize',14);
set(gca,'FontName','Times New Roman','FontSize',12);
saveas(gcf,'residual','eps'); 
end