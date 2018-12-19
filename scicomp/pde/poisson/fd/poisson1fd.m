function sol = poisson1fd(L,N,f,BCtype0,BCval0,BCtype1,BCval1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Resolution of Poisson equation in 1D using finite-differences
% Poisson equation: \Delta u=f
% Poisson equation 1D : u^{''}(x) = f(x)
% Discretized Laplace equation 1D
% 
% INPUTS
% L : domain [x0, xN+1]. 2 x 1 vector
% N : discretization size: N+1 interval, N+2 total nodes, N interior nodes. integer
% f : source term ()contains boundary terms if needed. N+2 x 1 vector
% BCtype0 / 1 : type of Boundary conditions at x0 / xN+1. string 'DIR' for Dirichlet,
% 'NEU1' for Neumann (first-degree operator), 'NEU2' for Neumann (second-degree operator)
% BCtval0 / 1 : numeric value of Boundary conditions at x0 / xN+1. double 
% 
% 
% OUTPUTS
% solution : solution at mesh nodes. N+2 x 1 vector
%
% TODO
% Implement Neumann BC with directional derivatives: change signs of NEU1
% BC RHS and BC in solution
% Implement solver for tridiagonal matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DISCRETIZATION %
h = (L(2)-L(1))/(N+1);%Discretization length
c = 1/h^2;
D = -2*c*ones(N-2,1);%Main diagonal (0 diagonal) of the core of the stiffness matrix unaffected by Boundary Conditions: dim N-2 x N-2
Eu = c*ones(N-2,1);%Upper extradiagonal (1 diagonal) of the core of the stiffness matrix unaffected by Boundary Conditions: dim N-2 x N-2
El = Eu;%Lower extradiagonal (-1 diagonal) of the Core of the stiffness matrix unaffected by Boundary Conditions: dim N-2 x N-2

% BOUNDARY CONDITIONS AT x0 %
if(strcmp(BCtype0,'DIR'))%Dirichlet Boundary Conditions at x0
    D = [-2*c ; D];% Add matrix diagonal term for x1
    Eu = [c ; Eu];% Add matrix upper extradiagonal term for x1
    f = f(2:end);%Delete term at x0, unnecessary for solving
    f(1) = f(1) - c * BCval0;%Add right-hand side BC term for x1
elseif(strcmp(BCtype0,'NEU1'))%Neumann Boundary Conditions at x0, 1st degree operator
    D = [-c ; D];% Add matrix diagonal term for x1
    Eu = [c ; Eu];% Add matrix upper extradiagonal term for x1
    f = f(2:end);%Delete term at x0, unnecessary for solving
    f(1) = f(1) + c * h * BCval0;%Add right-hand side BC term for x1
elseif(strcmp(BCtype0,'NEU2'))%Neumann Boundary Conditions at x0, 2nd degree operator
    D = [-2*c ; -2*c ; D];% Add matrix diagonal term for x0 and x1
    Eu = [2*c ; c ; Eu];% Add matrix upper extradiagonal term for x0 and x1
    El = [c ; El];% Add matrix lower extradiagonal term for x1
    f(1) = f(1) + 2*c * h * BCval0;%Add right-hand side BC term for xN
else %Error in the Boundary condition type
    error('Wrong Boundary Condition specified at x0');
end
% BOUNDARY CONDITIONS AT xN+1 %
if(strcmp(BCtype1,'DIR'))%Dirichlet Boundary Conditions at xN+1
    D = [D ; -2*c];% Add matrix diagonal term for xN
    El = [El ; c];% Add matrix lower extradiagonal term for xN
    f = f(1:end-1);% Delete term at xN+1, unnecessary for solving
    f(end) = f(end) - c * BCval1;%Add right-hand side BC term for xN
elseif(strcmp(BCtype1,'NEU1'))%Neumann Boundary Conditions at xN+1, 1st degree operator
    D = [D ; -c];% Add matrix diagonal term for xN
    El = [El ; c];% Add matrix lower extradiagonal term for xN
    f = f(1:end-1);% Delete term at xN+1, unnecessary for solving
    f(end) = f(end) - c * h * BCval1;%Add right-hand side BC term for xN
elseif(strcmp(BCtype1,'NEU2'))%Neumann Boundary Conditions at xN+1, 2nd degree operator
    D = [D ; -2*c ; -2*c];% Add matrix diagonal term for xN and xN+1
    El = [El ; c ; 2*c];% Add matrix lower extradiagonal term for xN and xN+1
    Eu = [Eu ; c];% Add matrix upper extradiagonal term for xN
    f(end) = f(end) - 2*c * h * BCval1;%Add right-hand side BC term for xN
else %Error in the Boundary condition type
    error('Wrong Boundary Condition specified at xN+1');
end

%ASSEMBLE AND SOLVE THE SYSTEM %
K = diag(D) + diag(Eu,1) + diag(El,-1);%Assemble stiffness matrix
tic;
sol = linsolve(K,f);
toc;

if(strcmp(BCtype0,'DIR'))%Adding BC at x0 to solution if needed
    sol = [BCval0 ; sol];
elseif(strcmp(BCtype0,'NEU1'))
    sol = [(sol(1) - h*BCval0) ; sol];
end
if(strcmp(BCtype1,'DIR'))%Adding BC at xN+1 to solution if needed
    sol = [sol ; BCval1];
elseif(strcmp(BCtype1,'NEU1'))
    sol = [sol ; (sol(end) + h*BCval1)];
end

end

function solution = linesolve_tridiag(A,b)%Solve Ax = b using xx algorithm for performance for tridiagonal matrices
    A;
    b;
end

