function [] = heat1fd()
%Resolution of Heat equation in 1D using finite-differences
% Heat equation: \rho c_p \frac {\partial T}{\partial t} - \nabla \cdot \left(k\nabla T\right)=q
% Heat equation 1D : \rho c_p \frac{dT}{dt} - \frac{dk}{dx} \frac{dT}{dx} - k \frac{d^2k}{dx^2} = q
% Discretized heat equation 1D

%Create functions for heat source and BC/initial shapes (ramp, wiggle,
%flat), time and space dependent
% Make known quantities q, BC time dependent and solve for each time

clear;
L = [0 1];%Size of the domain
n = 102;%Number total discretization points, including boundary points
ni = n-2;%Number of interior discretization points, excluding boundary points
dt = 1e-3;%timestep of the numerical method
time_int = 'implicit';% Time integration : 'explicit' or 'implicit'
nstep = 300;
cp = 1.0; %Thermal capacity, assumed constant over space
rho = 1.0; %density, assumed constant over space
q = 0.5*zeros(ni,nstep);%volumetric heat source, assumed constant over time
k = 1.0; %thermal conductivity, assumed constant over space
h = (L(2)-L(1))/(ni+1);%Discretization length

cx = k/h^2;
ct = rho*cp/dt;
CFL_value = 2*cx/ct;
CFL = CFL_value <= 1.0;%CFL stability condition for explicit scheme

% BOUNDARY CONDITIONS
BC0 = wiggle_BC(0.0,2.0,nstep*dt*0.1,dt,nstep);%Dirichlet boundary conditions at x=0
BC1 = const_BC(0.0,nstep);%Dirichlet boundary conditions at x=1
u_0 = const_IC(0.0,ni);%Initial temperature field at t=0

D = -2*cx/ct;%Diagonal term of the stiffness matrix
E = 1*cx/ct;%Extradiagonal term of the stiffness matrix
K = diag(D*ones(ni,1)) + diag(E*ones(ni-1,1),1) + diag(E*ones(ni-1,1),-1);%Assembled stiffness matrix

%add boundary conditions contribution to source terms in the second member b
b = q;
b(1,:) = (b(1,:) + BC0*cx)/ct;
b(ni,:) = (b(ni,:) + BC1*cx)/ct;

%SOLVING THE SySTEM

solution = zeros(n,nstep);%Saving intermediate solutions
solution(1,:) = BC0;%Prescribed value on first node
solution(end,:) = BC1;%Prescribed value on last node
u_t = u_0;% Temperature field initialized with the initial conditions at time t
for t=1:nstep
    if(strcmp(time_int,'implicit'))% Implicit time integration: (1-K) u_tp1 = u_t + q/ct
        u_tp1 = linsolve(eye(ni)-K,u_t+b(:,t));% Temperature field at time t+1
        solution(2:end-1,t) = u_tp1;
        u_t = u_tp1;
    elseif(strcmp(time_int,'explicit'))% Explicit time integration: u_tp1 = (1+K) U_t + q/ct
        if(~CFL)
            disp('WARNING: CFL condition not met. Explicit scheme UNSTABLE');
        end
        u_tp1 = (eye(ni) + K)*u_t + b(:,t);% Temperature field at time t+1
        solution(2:end-1,t) = u_tp1;
        u_t = u_tp1;
    end
end

figure;
colormap(jet);
u_min = min(min(solution));
u_max = max(max(solution));
for t=1:1:nstep
    plot((L(1) + (0:1:n-1)*(L(2)-L(1))/(n-1)),solution(:,t),'k-');hold on;
    scatter((L(1) + (0:1:n-1)*(L(2)-L(1))/(n-1)),solution(:,t),[],solution(:,t),'filled');
    %colorbar; % slows down the plot
    caxis([u_min u_max]);
    title(['Temperature map at step ',num2str(t), ' - ',time_int,' scheme (CFL = ',num2str(CFL_value),')']);
    xlabel('X');
    ylabel('Temperature');
    ylim([L(1) L(2)]);
    ylim([u_min u_max]);
    hold off;
    drawnow;
end

figure;
contourf((L(1):h:L(2)),(dt:dt:nstep*dt),solution',100,'LineColor','none');
title('Space-time temperature map of the entire simulation');
xlabel('X');
ylabel('Time');
colormap(jet);
colorbar;

end

function BC = const_BC(BC_value,nstep)%Constant boundary conditions
    BC = ones(1,nstep)*BC_value;
end

function BC = ramp_BC(BC_start,BC_stop,nstep)%Ramp boundary conditions
    BC = BC_start*ones(1,nstep) + (BC_stop-BC_start)*(1:nstep)/nstep;
end

function BC = wiggle_BC(BC_mean,BC_amplitude,BC_period,dt,nstep)%wiggle boundary conditions, period in time units
    BC = BC_mean + BC_amplitude*sin(2*pi*(1:nstep)*dt/BC_period);
end

function IC = const_IC(IC_value,ni)%Constant initial conditions
    IC = ones(ni,1)*IC_value;
end