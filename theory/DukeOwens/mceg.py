from tqdm import tqdm

#--scipy stack
import numpy as np

#--torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

#--matplotlib
import matplotlib
from matplotlib.lines import Line2D
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

import pylab as py
from matplotlib import colors
import matplotlib.gridspec as gridspec

#--custom
import params as par
from idis import IDIS

class MCEG:
    """MCEG stands for Monte Carlo Even Generator"""

    def __init__(self, rs=140.0, tar='p', W2min=10.0, nx=10, nQ2=10, mx=10, mQ2=10, ):
        """
        :param int nx: number of x-dimension bins
        :param int nQ2: number of Q2-dimension bins
        :param float W2min: resonances (relates to regions in the
            phase space where up/down quarks do not make sense); used to cutdown events
        :param float rs: center of mass energy
        :param mx: number of points in each bin in the x-direction to be used for sampling using inverse CDF
        :param mQ2: number of points in each bin in the Q^2-direction to be used for sampling using inverse CDF
        """
        self.idis = IDIS()
        self.nx = nx
        self.nQ2 = nQ2
        self.tar = tar
        self.rs = rs
        self.W2min = W2min
        self.gen_xQ2_grid(nx, nQ2)
        self.gen_xsection()
        self.setup_inv_transorm_sampler(mx=mx, mQ2=mQ2)

    def gen_xQ2_grid(self,nx,nQ2):
        """
        Sample phase space grid points in log-space
        """
        # Sample Q^2 direction in phase space (log)
        Q2min = par.mc2  # absolute minimum Q^2 that is of interst (theory limiation)
        Q2max=self.rs**2-par.M2  # absolute minimum Q^2 that is of interst (attainable)
        LQ2=torch.linspace(np.log(Q2min),np.log(Q2max),nQ2)
        dLQ2=LQ2[1:]-LQ2[:-1]
        LQ2mid=0.5*(LQ2[1:]+LQ2[:-1])
        LQ2max=LQ2[1:]
        LQ2min=LQ2[:-1]

        # Sample x direction in phase space (log)
        xmin = Q2min/(self.rs**2-par.M2)  # min attainable x
        xmax = 1  # max x
        Lx = torch.linspace(np.log(xmin),np.log(xmax),nx)
        dLx = Lx[1:]-Lx[:-1]
        Lxmid = 0.5*(Lx[1:]+Lx[:-1])
        Lxmax = Lx[1:]
        Lxmin = Lx[:-1]

        # Mesh discretization and grid
        Lxmid,LQ2mid=torch.meshgrid(Lxmid,LQ2mid)
        Lxmin,LQ2min=torch.meshgrid(Lxmin,LQ2min)
        Lxmax,LQ2max=torch.meshgrid(Lxmax,LQ2max)
        dLx, dLQ2 = torch.meshgrid(dLx,dLQ2)

        ## Filtering stages (filter out any gridpoints (events) )
        # First condition:
        x =torch.exp(Lxmin); Q2=torch.exp(LQ2max)
        Q2max = (self.rs**2-par.M2)*x
        cond1 = Q2max>=Q2

        # Second condition:
        x =torch.exp(Lxmax); Q2=torch.exp(LQ2min)
        W2=par.M2+Q2/x-Q2
        cond2=W2>=self.W2min
        acc = cond1*cond2  # Acceptance (boolean flags) bins that identify acceptable bins

        self.Lxmid=Lxmid; self.LQ2mid=LQ2mid;
        self.Lxmin=Lxmin; self.LQ2min=LQ2min;
        self.Lxmax=Lxmax; self.LQ2max=LQ2max;

        self.dLx=dLx; self.dLQ2=dLQ2;
        self.acc=acc

    def gen_xsection(self):
        """
        Calculate the cross-section values (density values)
        normalized and unnormalized over the discretized x, Q^2 grid
        """
        x = torch.exp(self.Lxmid)
        Q2 = torch.exp(self.LQ2mid)
        self.dxsec = torch.zeros(x.shape)
        entries = [(a,b) for a in range(x.shape[0]) for b in range(x.shape[1])]
        #for i,j in tqdm(entries):
        #    if self.acc[i,j]==False: continue
        #    self.dxsec[i,j]=self.idis.get_diff_xsec(x[i,j],Q2[i,j],self.rs,self.tar,'xQ2')

        # Evaluate differential cross-section accross the whole grid (point by point)
        self.dxsec = self.idis.get_diff_xsec(x,Q2,self.rs,self.tar,'xQ2')

        # Ignore regions those are physcially irrelevant
        self.dxsec[self.acc==False] = 0

        # Calculating integral of the cross-section wrt x, Q^2 (by change of variables to log-scale)
        integrand = self.dxsec * (x*self.dLx) * (Q2*self.dLQ2)*self.acc  # Apply effect of the Jacobian of the transformation
        self.total_xsec = torch.sum(integrand)

        # calculating weights (approximate of normalized cross-section over the discretization of x, Q^2)
        self.weights = integrand/self.total_xsec

    def setup_inv_transorm_sampler(self, mx=10, mQ2=10):
        """
        Set up the inverse transform sampler; build the CDF function accross discretization
        with finer grid with number of points withing bins set to mx, mQ2

        ..note::
            This function evaluates the CDF values (accross all segments). It does NOT construct
            actual inverse CDF, but it contais the information needed for sampling later
            (following inverse CDF approach)
        """
        entries01 = [(a,b) for a in range(self.Lxmid.shape[0]) for b in range(self.Lxmid.shape[1])]
        entries10 = [(a,b) for a in range(self.Lxmid.shape[1]) for b in range(self.Lxmid.shape[0])]
        grid = self.Lxmin.shape

        #print('grid',grid)

        # Construct the CDF accross the x-dimension (slicing accross the Q^2 dimesnion and keepin it constant) for all bins
        # Results stored in `self.cdfx`
        ## x dimesion
        Lxmin = self.Lxmin[:,0].reshape(-1,1)
        dLx=self.dLx[:,0].reshape(-1,1)
        u=torch.linspace(0,1,mx)
        x=torch.exp((Lxmin+u*dLx)).flatten()
        xbins=x.reshape(-1,mx)
        #print('xbins',xbins.shape)
        Q2=torch.exp(self.LQ2min.T[:,0]).flatten()

        Q2,x=torch.meshgrid(Q2,x)
        xsecx=self.idis.get_diff_xsec(x,Q2,self.rs,self.tar,'xQ2')
        #print('xsecx',xsecx.shape)
        xsecx = xsecx.reshape(grid[1],-1,mx)
        #print('xsecx',xsecx.shape)

        # Use trapezoid rule to calculate norm within bins (maybe segments better!)
        normx = torch.zeros((grid[1],grid[0]))
        #print('normx',normx.shape)
        for i,j in tqdm(entries10):
            normx[i,j]=torch.trapezoid(xsecx[i,j],xbins[j])

        rhox = torch.zeros(xsecx.shape)
        #print('rhox',rhox.shape)
        for i,j in tqdm(entries10):
            if self.acc[j,i]==True:
                rhox[i,j]=xsecx[i,j]/normx[i,j]

        cdfx=torch.zeros(xsecx.shape)
        #print('cdfx',cdfx.shape)
        for i,j in tqdm(entries10):
            if self.acc[j,i]==True:
                for k in range(mx):
                    cdfx[i,j,k]=torch.trapezoid(rhox[i,j,k:],xbins[j,k:])
        self.cdfx = cdfx
        self.xbins = xbins  # arrays/gridpoints correspoinding to entries in self.cdfx


        # Construct the CDF accross the x-dimension (slicing accross the Q^2 dimesnion and keepin it constant) for all bins
        # Results stored in `self.cdfQ2`
        ## Q2 dimesion
        LQ2min=self.LQ2min.T[:,0].reshape(-1,1)
        dLQ2=self.dLQ2.T[:,0].reshape(-1,1)
        u=torch.linspace(0,1,mQ2)
        Q2=torch.exp((LQ2min+u*dLQ2)).flatten()
        Q2bins=Q2.reshape(-1,mQ2)
        #print('Q2bins',Q2bins.shape)
        x=torch.exp(self.Lxmin[:,0]).flatten()

        x,Q2=torch.meshgrid(x,Q2)
        xsecQ2=self.idis.get_diff_xsec(x,Q2,self.rs,self.tar,'xQ2')
        #print('xsecQ2',xsecQ2.shape)
        xsecQ2=xsecQ2.reshape(grid[0],-1,mQ2)
        #print('xsecQ2',xsecQ2.shape)

        normQ2=torch.zeros((grid[0],grid[1]))
        #print('normQ2',normQ2.shape)
        for i,j in tqdm(entries01):
            normQ2[i,j]=torch.trapezoid(xsecQ2[i,j],Q2bins[j])

        rhoQ2=torch.zeros(xsecQ2.shape)
        #print('rhoQ2',rhoQ2.shape)
        for i,j in tqdm(entries01):
            if self.acc[i,j]==True:
                rhoQ2[i,j]=xsecQ2[i,j]/normQ2[i,j]

        cdfQ2=torch.zeros(xsecQ2.shape)
        #print('cdfQ2',cdfQ2.shape)
        for i,j in tqdm(entries01):
            if self.acc[i,j]==True:
                for k in range(mx):
                    cdfQ2[i,j,k]=torch.trapezoid(rhoQ2[i,j,k:],Q2bins[j,k:])
        self.cdfQ2=cdfQ2
        self.Q2bins=Q2bins  # arrays/gridpoints correspoinding to entries in self.cdfQ2

    def gen_events(self, N, verb=False):
        """
        Sampling using the inverse CDF approach (by slicing)

        :param int N: number of samples/events requested
        :param bool verb: screen verbosity

        :returns:
            two dimensional array (torch tensor instance) with dimension (K x 2) with
            `K` being the number of generated events/samples. Each row is (x, Q^2) instance

        ..note::
            Total number of samles returned (`K`) can be different from `N` because
            1. number is dectated by the weights
            2. zero events are dropped after sampling
        """
        # (Approximate) number of sample point per bin
        n = torch.tensor(self.weights*N, dtype=int)

        evts = torch.zeros(2, N)
        entries = [(a,b) for a in range(self.Lxmid.shape[0])
                         for b in range(self.Lxmid.shape[1])]

        cnt = 0
        for i,j in tqdm(entries):

            #
            if self.acc[i,j] == False: continue

            # Interpolate the inverse CDF (given the values of the CDF over the grid)
            # NOTE:
            #  - this is with the CDF inverted in view [1--> 0] instead of [0-->1]
            #  - Calling self.interp with xp, fp flipped has the effect of calculating inveres CDF at uniform grid points.

            # Interpolation in x direction (inverse CDF holding Q^2-constant (slicing))
            evts[0, cnt:n[i,j]+cnt] = self.interp(
                torch.rand(n[i,j]),  # uniform (0, 1) random sample withing the corresponding bin
                torch.flip(self.cdfx[j,i], dims=(0,)),  # CDF values
                torch.flip(self.xbins[i],dims=(0,)),   # X-values of the CDF
            )
            # Interpolation in Q^2 direction (inverse CDF holding x-constant (slicing))
            evts[1, cnt:n[i,j]+cnt] = self.interp(
                torch.rand(n[i,j]),
                torch.flip(self.cdfQ2[i,j],dims=(0,)),
                torch.flip(self.Q2bins[j],dims=(0,)),
            )

            cnt += n[i,j]

        evts = evts.T
        evts = evts[evts[:,0]>0]

        if verb:
            print(self.total_xsec)
            ax=py.subplot(111)
            ax.plot(np.log(evts[:,0].detach().numpy()),np.log(evts[:,1].detach().numpy()),'r.')
            ax.plot(self.Lxmid[self.acc].detach().numpy(),self.LQ2mid[self.acc].detach().numpy(),'k.')
            ax.set_xlim(-10,0)
            ax.set_ylim(0,None)
            ax.tick_params(axis='both', which='major', labelsize=20,direction='in')
            ax.set_ylabel(r'$Q^2$',size=30)
            ax.set_xlabel(r'$x$',size=30)
            ax.set_xticks(np.log([1e-4,1e-3,1e-2,1e-1]))
            ax.set_xticklabels([r'$0.0001$',r'$0.001$',r'$0.01$',r'$0.1$'])
            ax.set_yticks(np.log([10,100,1000,10000]))
            ax.set_yticklabels([r'$10$',r'$100$',r'$1000$',r'$10000$']);
            py.show()

        return evts


    def interp(self, x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
        """
        Simple linear interpolation.
        :param x: gridpoints to interpolate from
        :param xp: gridpoints to interpolate to
        :param fp: values of the function (at grid points x) to interpolate (from x to xp)
        """
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])
        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)
        return m[indicies] * x + b[indicies]

