function sol = poisson1fem(L,N,ftype,fval)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Resolution of Poisson equation in 1D using finite element
% Poisson equation: \Delta u=f
% Poisson equation 1D : u^{''}(x) = f(x)
% Discretized Laplace equation 1D
% 
% INPUTS
% L : domain [x0, xN+1]. 2 x 1 vector
% N : discretization size: N+1 interval, N+2 total nodes, N interior nodes. integer
% ftype : mathematical form of source term. string 'AFF' for affine,
% fval : values of source term over domain. [f0 fN+1] if ftype = 'AFF'
% 
% 
% OUTPUTS
% sol : solution at mesh nodes. N+2 x 1 vector
%
%
% CURRENT IMPLEMENTATION
% Solves 1D FEM Poisson for P1 (linear) elements with affine source term
%
% TODO
% Implement Non-homogeneous Dirichlet BC
% Implement Neumann BC 
% Implement user-defined source tern as a [x f(x)] vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DISCRETIZATION %
h = (L(2)-L(1))/(N+1);%Discretization length
c = 1/h;

% INTEGRATION %
D = -2*c*ones(N,1);%Main diagonal (0 diagonal) of the stiffness matrix 
E = c*ones(N-1,1);%Extradiagonal (1 diagonal) of the stiffness matrix


% Source term
f = NaN(N,1);
if(strcmp(ftype,'AFF'))%Affine source term
    source = [fval(1) (fval(1) + ((1:N)*h - L(1))/(L(2) - L(1))*(fval(2)-fval(1))) fval(2)]';
    f = (source(1:end-2)+source(3:end))*h/2;
else %Error in the source term function condition type
    error('Source term function type not yet implemented');
end



%ASSEMBLE AND SOLVE THE SYSTEM %
K = diag(D) + diag(E,1) + diag(E,-1);%Assemble stiffness matrix
tic;
sol = linsolve(K,f);
toc;

%Adding Homogeneous Dirichlet BC at x0 and xN+1 to solution
sol = [0.0 ; sol ; 0.0];
end

