# Resolution of axisymmetric Heat equation in 1D using implicit (time) central (space) finite-differences
# Heat equation: \rho c_p \frac {\partial T}{\partial t} - \nabla \cdot \left(k\nabla T\right) = q
# Heat equation axisymmetric 1D : \rho c_p \frac{dT}{dt} - k \left(\frac{1}{r}\frac{dT}{dr} + \frac{d^2T}{dr^2}\right) = q
# Discretized heat equation axisymmetric 1D: \rho c_p (T_{r,t+1}-T_{r,t})/dt - k/r (T_{r+1,t+1}-T_{r-1,t+1})/(2h) - k (T_{r+1,t+1}-2T_{r,t+1}+T_{r-1,t+1})/(h^2) - q_{r,t+1} = 0
# SI units
# Dirichlet Boundary conditions

# TODO: input functions from script with sys.argv[]
# Solution with boundary temperatures


import sys
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Functions for preset time-dependent boundary conditions
def BC_satexp(t, param):
    # Saturaed exponential: T(t) = T_{ini} + (T_{fin}-T_{ini})*(1-\exp(-t/\tau))
    # param = tau, Tini, Tfin
    tau = float(param[0])
    Tini = float(param[1])
    Tfin = float(param[2])
    return Tini + (Tfin - Tini)*(1.0 - np.exp(-t/tau))

def BC_const(t, param):
    # Constant temperature: T(t) = T0
    # param = T0
    T0 = float(param[0])
    return T0

def main():
    # Parameters
    R = 0.075 # Radius, domain [0,R]
    rho = 1800.0 # Density, assumed constant over space
    cp = 795.0 # Specific heat capacity, assumed constant over space
    k = 0.33 # Thermal conductivity, assumed constant over space
    n = 102 # Number of total discretization points, including boundary points
    ni = n-2 # Number of interior discretization points, excluding boundary points
    r = np.linspace(0,R,n)
    nstep = 30000 # Number of timesteps
    dt = 1 # timestep of time integration
    nsave = 1000 # Number of times to save the data
    time = np.linspace(0,nstep*dt,nsave)
    scheme = 'implicit' # time-discretization scheme

    # Discretization and variables
    h = R/(ni+1) # Discretization length
    cr1 = k/(2*h)
    cr2 = k/h**2
    ct = rho*cp/dt
    q = np.zeros(ni) # volumetric heat source, assumed constant over time

    # Assembly of stiffness matrix (dimensions (ni,ni), interior points only) and flux vector
    DL = -cr1/(ct*r[1:-1]) + cr2/ct # bottom diagonal, factor of left temperature T_{r-1}
    DC = -2*cr2/ct*np.ones(ni) # central (main) diagonal, factor of center temperature T_{r}
    DR = cr1/(ct*r[1:-1]) + cr2/ct # top diagonal, factor of right temperatur T_{r+1}
    K = np.diag(DL[1:],k=-1) + np.diag(DC,k=0) + np.diag(DR[0:-1],k=1) # Assembled stiffness matrix
    Q = q/ct; # Flux vector

    # Initial and Boundary Conditions
    temp = np.zeros(ni) # Initial temperature field at t=0
    """
    BCtype = ['NEU','DIR'] # Type of Boundary conditions, Dirichlet 'DIR' or Neumann 'NEU' at [r=0, r=R]
    BCform = [0.0,BC_satexp] # Time-dependent Boundary Conditions functions wrappers at [r=0, r=R]. Only valid for Dirichlet
    BCparams = [[0.0],[600, 0, 40]] # Parameters for the boundary condition function wrappers at [r=0, r=R]. Only valid for Dirichlet
    """
    BCtype = ['NEU','DIR'] # Type of Boundary conditions, Dirichlet 'DIR' or Neumann 'NEU' at [r=0, r=R]
    BCform = [None,BC_satexp] # Time-dependent Boundary Conditions functions wrappers at [r=0, r=R]. Only valid for Dirichlet
    BCparams = [[None],[600.0, 0.0, 10.0]] # Parameters for the boundary condition function wrappers at [r=0, r=R]. Only valid for Dirichlet
    
    # Time-dependent BC as contributions to flux vector Q
    Q_BC = np.zeros(ni)
    # Time-independent zero-flux Neumann conditions
    # 1st order zero-flux for simplicity, i.e. T_0 = T_1
    # Changes in the permanent stiffness matrix
    if (BCtype[0] == 'NEU'):
        K[0,0] += DL[0]
    if (BCtype[1] == 'NEU'):
        K[-1,-1] += DR[-1]
        
    # Solve Heat equation
    nevery = nstep/nsave
    temp_save = np.zeros((nsave,ni))

    for t in range(nstep):
        if (t % nevery == 0):
            temp_save[t/nevery] = np.copy(temp)
        # Time-dependent Dirichlet Boundary conditions at r=0
        if (BCtype[0] == 'DIR'):
            Q_BC[0] = DL[0]*BCform[0]((t+1)*dt,BCparams[0]) # Implicit: imposed temperture at time (t+1)
        # Time-dependent Dirichlet Boundary conditions at r=0
        if (BCtype[1] == 'DIR'):
            Q_BC[-1] = DR[-1]*BCform[1]((t+1)*dt,BCparams[1]) # Implicit: imposed temperture at time (t+1)
        if (scheme == 'implicit'): #Implicit time integration: (1-K) T_tp1 = T_t + (Q+Q_BC)_tp1
            temp = LA.solve(np.identity(ni)-K,temp+Q+Q_BC)
        elif(scheme == 'explicit'): # Explicit time integration: T_tp1 = (1+K) T_t + (Q+Q_BC)_t
            raise ValueError('Explicit integration not implemented')
    
    
    # Compute average temperature, integral on disc
    temp_ave = np.zeros(nsave)
    for t in range(nsave):
        temp_ave[t] = 2.0/r[-2]**2*integrate.simps(temp_save[t]*r[1:-1],r[1:-1])
    
    # Output
    fname = "heat_1D_axisymmetric.csv"
    header = ['time','temp_out','temp_ave']
    data = np.concatenate((time[:,np.newaxis],BCparams[1][2]*(1.0 - np.exp(-time/BCparams[1][0]))[:,np.newaxis],temp_ave[:,np.newaxis]),axis=1)
    np.savetxt(fname, data, header=",".join(header), delimiter=',')
    
 
    # draft/debug plot and output 
    """
    fname = "heat_1D_axisymmetric.csv"
    header = ['time','temp']
    time2D = np.insert(time,0,-1)[:,np.newaxis] # fills corner zith -1. rows = radius, columns = time
    data = np.concatenate((time2D,np.transpose(np.concatenate((r[1:-1,np.newaxis],np.transpose(temp_save)),axis=1))),axis=1)
    np.savetxt(fname, data, header=",".join(header), delimiter=',')
    
    fig, ax = plt.subplots()
    ax.plot(r[1:-1],temp_save[0])
    ax.plot(r[1:-1],temp_save[nsave/2])
    ax.plot(r[1:-1],temp_save[nsave-1])
    
    fig, ax2 = plt.subplots()
    #ax2.plot(time,temp_save[:,0])
    #ax2.plot(time,temp_save[:,ni/2])
    #ax2.plot(time,temp_save[:,ni-1])
    ax2.plot(time/3600,BCparams[1][2]*(1.0 - np.exp(-time/BCparams[1][0])),'k', label='outer temperature')
    ax2.plot(time/3600,temp_ave,'k--',label='average temperature')
    ax2.plot(time/3600,BCparams[1][2]*(1.0 - np.exp(-time/BCparams[1][0])) - temp_ave,'r', label='temperature difference')
    ax2.legend()
    ax2.set_xlabel('time [h]')
    ax2.set_ylabel('temperature [C]')
    """
if __name__== "__main__":
  main()




