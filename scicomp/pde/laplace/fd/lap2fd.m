function [] = lap2fd()
%Solving 2D Laplace Equation on rectangular domain using finite-differences
% TODO : IMPLEMENT functions on BC (flat, ramp, sine)
clear;
Lx = [0 1]; %Dimensions of the rectangular domain in x-direction
Ly = [0 1]; %Dimensions of the rectangular domain in y-direction
hx = 1/51; %Size of the discretization step in x-direction
hy = 1/51; %Size of the discretization step in y-direction
nx = floor((Lx(2)-Lx(1))/hx)-1; %Number of discretization points of the domain in x-direction
ny = floor((Ly(2)-Ly(1))/hy)-1; %Number of discretization points of the domain in y-direction
cx = 1.0/hx^2;
cy = 1.0/hy^2;

% The unknown would be gathered in a vector, varying along x first and y
% second e.g. for a 3x3 system Uxy = {u11 u21 u31 u12 u22 u32 u13
% u23 u33}

%Dirichlet boundary conditions
BCx0 = 1.0;% x = 0
BCx1 = 0.0;% x = Lx
BCy0 = 0.0;% y = 0
BCy1 = 0.0;% y = Lx

%Explicit Boundqry Condition: all boundary values are set and not
%considered as unknowns of the problem
%Assembly of the stiffness matrix
D = diag((-2*cx -2*cy)*ones(nx,1)) + diag(1*cx*ones(nx-1,1),1) + diag(1*cx*ones(nx-1,1),-1); % Diagonal block submatrix
E = diag(1*cy*ones(nx,1)); % Extradiagonal block submatrix
K = diag_block(D,ny,0) + diag_block(E,ny-1,1) + diag_block(E,ny-1,-1); % Global Stiffness matrix K

%Second member from counray conditions
b = zeros(nx*ny,1);%Second member, accounts for boundary conditions explicitly (no Lagrange Multipliers)
for j=1:ny
    for i=1:nx
        index = i + (j-1)*nx;
        if(i == 1) %Node (i-1,j) on boundary x=0
            b(index,1) = b(index,1) -1*cx*BCx0;
        elseif(i==nx) %Node (i+1,j) on boundary x=1
            b(index,1) = b(index,1) -1*cx*BCx1;
        end
        
        if(j == 1) %Node (i,j-1) on boundary y=0
            b(index,1) = b(index,1) -1*cy*BCy0;
        elseif(j==ny) %Node (i,j+1) on boundary y=1
            b(index,1) = b(index,1) -1*cy*BCy1;
        end        
    end
end

%Resolution of the linear problem
u = linsolve(K,b);

u_grid = vec2mat(u,nx); % x/y-direction = first (row)/second (column) dimension
figure;
contourf(hx:hx:nx*hx,hy:hy:ny*hy,u_grid,50,'LineColor','none');
colormap(jet);
colorbar;
title('Isovalues of the 2D Laplace equation solution');
xlabel('X');
ylabel('Y');
axis equal;
end

function B = diag_block(A,n,diag)
% Works similarly as the diag function but for block matricx entries instead of
% vectors
% A : m x m block matrix to be replicated in the diagonal
% n : number of blocks desired
% diag : integer, index of the diagonal to be used. 0 for main diagonal, +/- i for ith upper/lower diagonal
% B : resulting matrix assembled by block of A matrix, its size is N*N with
% N = m * [n + abs(diag)]
m = size(A,1);
B = zeros(m * (n + abs(diag)));
for t = 1:n
    i_block = t + abs(min(diag,0));% row index of block matrix
    j_block = t + abs(max(diag,0));% column index of block matrix
    B(1+(i_block-1)*m:i_block*m,1+(j_block-1)*m:j_block*m) = A;
end
end