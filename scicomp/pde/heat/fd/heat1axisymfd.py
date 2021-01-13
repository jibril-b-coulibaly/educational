# Resolution of axisymmetric Heat equation in 1D using implicit (time) central (space) finite-differences
# Heat equation: \rho c_p \frac {\partial T}{\partial t} - \nabla \cdot \left(k\nabla T\right) = q
# Heat equation axisymmetric 1D : \rho c_p \frac{dT}{dt} - k \left(\frac{1}{r}\frac{dT}{dr} + \frac{d^2T}{dr^2}\right) = q
# Discretized heat equation axisymmetric 1D: \rho c_p (T_{r,t+1}-T_{r,t})/dt - k/r (T_{r+1,t+1}-T_{r-1,t+1})/(2h) - k (T_{r+1,t+1}-2T_{r,t+1}+T_{r-1,t+1})/(h^2) - q_{r,t+1} = 0
# SI units: All inputs and outputs !!!
# Dirichlet Boundary conditions

# TODO: Solution with boundary temperatures

# Executable script  with the following command line arguments:
# radius rho cp k n nstep dt BC1type BC1form BC1arg1 BC1arg2 ... BC2type BC2form BC2arg1 BC2arg2 ...

# radius: Radius of the domain [0;R] [m]
# rho: density of material [kg/m^3]
# cp: Specific heat capacity [J/(kg*K)]
# k: Thermal conductivity [W/(m*K)]
# n: number of total discretization points (includes boundary points) [-]
# nstep: number of timesteps [-]
# dt: timestep of time integration [s]
# nsave: number of saved states [-]
# BC1type: type of boundary condition at r=0, 'DIR' or 'NEU'
# BC1form: form of Dirichlet boundary condition at r=0, "const" or "satexp". Absent for BC1type=='NEU'
# BC1args: list of arguments for the Dirichlet boundary condition form at r=0, <TBC1>  for "const" or <tau1 Tini1 Tfin1 for "satexp". absent for BC1type=='NEU'
# BC2type: type of boundary condition at r=R, 'DIR' or 'NEU'
# BC2form: form of Dirichlet boundary condition at r=R, "const" or "satexp". Absent for BC2type=='NEU'
# BC2args: list of arguments for the Dirichlet boundary condition form at r=R, <TBC1>  for "const" or <tau2 Tini2 Tfin2 for "satexp". absent for BC2type=='NEU'



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

BC_dic = {"const" : BC_const,
          "satexp" : BC_satexp}

def main():
    # Material parameters
    R = float(sys.argv[1]) # 0.5*0.075 # Radius, domain [0,R]
    rho = float(sys.argv[2]) # 1800.0 # Density, assumed constant over space
    cp = float(sys.argv[3]) # 795.0 # Specific heat capacity, assumed constant over space
    k = float(sys.argv[4]) # 0.33 # Thermal conductivity, assumed constant over space
    n = int(sys.argv[5]) # 102 # Number of total discretization points, including boundary points
    # Simulation parameters
    nstep = int(sys.argv[6]) # 30000 # Number of timesteps
    dt = float(sys.argv[7]) # 1 # timestep of time integration
    nsave = int(sys.argv[8]) # 1000 # Number of times to save the data
    # Boundary conditions
    BCtype = [None,None]
    BCform = [None,None]
    BCargs = [None,None]
    BCstr = [None,None] # For output naming
    iarg=9
    for i in [0,1]:
        # i = 0 for BC at r=0, i = 1 for BC at r=R
        BCtype[i] = sys.argv[iarg] # Boundary condition: 'DIR' or 'NEU'
        if (BCtype[i] == 'DIR'):
            # Dirichlet Boundary condition
            BCform[i] = sys.argv[iarg+1] # Form of Dirichlet boundary condition at r=0: "const" or "satexp"
            if (BCform[i] == 'const'):
                BCargs[i] = [float(sys.argv[iarg+2])] # Imposed temperature
                BCstr[i] = "DIR_const_"+sys.argv[iarg+2]
                iarg = iarg + 3
            elif (BCform[i] == 'satexp'):
                BCargs[i] = [float(sys.argv[iarg+2]),float(sys.argv[iarg+3]),float(sys.argv[iarg+4])] # [tau, Tini, Tfin]
                BCstr[i] = "DIR_satexp_"+sys.argv[iarg+2]+"_"+sys.argv[iarg+3]+"_"+sys.argv[iarg+4]
                iarg = iarg + 5
            else:
                raise ValueError('Dirichlet BC must be <const> or <satexp>')
        elif (BCtype[i] == 'NEU'):
            # Neumann Boundary condition, zero flux
            BCform[i] = None
            BCargs[i] = [None]
            BCstr[i] = "NEU"
            iarg = iarg + 1
        else:
            raise ValueError('BC must be <DIR> or <NEU>')

    # Discretization and variables
    scheme = 'implicit' # time-discretization scheme
    ni = n-2 # Number of interior discretization points, excluding boundary points
    r = np.linspace(0,R,n) # radius variable
    time = np.linspace(0,nstep*dt,nsave) # time variable
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
    BCargs = [[0.0],[600, 0, 40]] # Parameters for the boundary condition function wrappers at [r=0, r=R]. Only valid for Dirichlet

    BCtype = ['NEU','DIR'] # Type of Boundary conditions, Dirichlet 'DIR' or Neumann 'NEU' at [r=0, r=R]
    BCform = [None,BC_satexp] # Time-dependent Boundary Conditions functions wrappers at [r=0, r=R]. Only valid for Dirichlet
    BCargs = [[None],[600.0, 0.0, 10.0]] # Parameters for the boundary condition function wrappers at [r=0, r=R]. Only valid for Dirichlet
    """
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
            Q_BC[0] = DL[0]*BC_dic[BCform[0]]((t+1)*dt,BCargs[0]) # Implicit: imposed temperture at time (t+1)
        # Time-dependent Dirichlet Boundary conditions at r=0
        if (BCtype[1] == 'DIR'):
            Q_BC[-1] = DR[-1]*BC_dic[BCform[1]]((t+1)*dt,BCargs[1]) # Implicit: imposed temperture at time (t+1)
        if (scheme == 'implicit'): #Implicit time integration: (1-K) T_tp1 = T_t + (Q+Q_BC)_tp1
            temp = LA.solve(np.identity(ni)-K,temp+Q+Q_BC)
        elif(scheme == 'explicit'): # Explicit time integration: T_tp1 = (1+K) T_t + (Q+Q_BC)_t
            raise ValueError('Explicit integration not implemented')
    
    
    # Compute average temperature, integral on disc
    temp_ave = np.zeros(nsave)
    for t in range(nsave):
        temp_ave[t] = 2.0/r[-2]**2*integrate.simps(temp_save[t]*r[1:-1],r[1:-1])

    # Output
    fname = "heat1axisymfd_rho={:.0f}_cp={:.0f}_k={:.2f}_R={:.2e}_BC0={}_BCR={}.csv".format(rho, cp, k, R, BCstr[0], BCstr[1])
    header = ['time','temp_out','temp_in','temp_ave']
    data = np.concatenate((time[:,np.newaxis],
                           BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0]))[:,np.newaxis],
                           temp_save[:,0][:,np.newaxis],
                           temp_ave[:,np.newaxis]),axis=1)
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
    ax2.plot(time/3600,BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0])),'k', label='outer temperature')
    ax2.plot(time/3600,temp_ave,'k--',label='average temperature')
    ax2.plot(time/3600,BCargs[1][2]*(1.0 - np.exp(-time/(6*BCargs[1][0]))),'b--', label='exponential tau_ave = 6 tau')
    ax2.plot(time/3600,BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0])) - temp_ave,'r', label='temperature difference')
    ax2.plot(time/3600,BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0])) - BCargs[1][2]*(1.0 - np.exp(-time/(6*BCargs[1][0]))),'b', label='simple model difference')
    ax2.legend()
    ax2.set_xlabel('time [h]')
    ax2.set_ylabel('temperature [C]')
    
    fig, ax3 = plt.subplots()
    ax3.plot(time/3600,1e2*33*1e-6*BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0])),'k', label='steel ring')
    ax3.plot(time/3600,1e2*2.6*1e-5*temp_ave,'k--',label='soil')
    ax3.plot(time/3600,1e2*33*1e-6*BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0])) - 1e2*2.6*1e-5*temp_ave,'r', label='average differential')
    ax3.plot(time/3600,1e2*(33*1e-6-2.6*1e-5)*BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0])),'b', label='ideal differential Tring=Tsoil')
    ax3.legend()
    ax3.set_xlabel('time [h]')
    ax3.set_ylabel('volumetric expansion [%]')
    
    fig, ax4 = plt.subplots()
    ax4.plot(time/3600,1e2*33*1e-6*np.ones(len(time)),'k', label='steel ring')
    ax4.plot(time/3600,1e2*2.6*1e-5*np.ones(len(time)),'k--',label='Quartz (soil material)')
    ax4.plot(time/3600,1e2*2.6*1e-5*temp_ave/(BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0]))),'r',label='average soil')
    ax4.plot(time/3600,1e2*2.6*1e-5*BCargs[1][2]*(1.0 - np.exp(-time/(6*BCargs[1][0])))/(BCargs[1][2]*(1.0 - np.exp(-time/BCargs[1][0]))),'b', label='average model expo')
    ax4.legend()
    ax4.set_xlabel('time [h]')
    ax4.set_ylabel('Thermal expansion coefficient [%/C]')
    """
if __name__== "__main__":
  main()




