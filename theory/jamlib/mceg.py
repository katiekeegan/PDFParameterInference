import numpy as np
from tqdm import tqdm
import params as par 

#--matplotlib
import matplotlib
from matplotlib.lines import Line2D
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# matplotlib.rc('text',usetex=True)
import pylab as py
from matplotlib import colors
import matplotlib.gridspec as gridspec

class MCEG:
    """
    MCEG = Monte Carlo Event Generator
    """

    def __init__(self,idis,rs=140,tar='p',W2min=10,nx=10,nQ2=10):
        """
        inputs:
            idis = instance of the idis class that computes cross sections of deep inelastic scattering
            rs = "root s", square root of s = incoming center-of-mass momentum squared; type = float
            tar = target of the scattering experiment, string
            W2min = minimum allowed W2 for an analysis of 1d QCFs; type = number
            nx = number of x bins
            nQ2 = number of Q2 bins
        """
        self.idis=idis
        self.tar=tar
        self.rs=rs
        self.W2min=W2min
        self.gen_xQ2_grid(nx,nQ2)
        self.gen_xsection()
    
    def gen_xQ2_grid(self,nx,nQ2):
        """
        function to generate log-space grids in x and Q2.
        inputs:
            nx = number of x bins
            nQ2 = number of Q2 bins
        """
        
        Q2min=par.mc2
        Q2max=self.rs**2-par.M2
        LQ2=np.linspace(np.log(Q2min),np.log(Q2max),nQ2)
        dLQ2=LQ2[1:]-LQ2[:-1]
        LQ2mid=0.5*(LQ2[1:]+LQ2[:-1])
        LQ2max=LQ2[1:]
        LQ2min=LQ2[:-1]
        
        xmin=Q2min/(self.rs**2-par.M2)
        xmax=1
        Lx=np.linspace(np.log(xmin),np.log(xmax),nx)
        dLx=Lx[1:]-Lx[:-1]
        Lxmid=0.5*(Lx[1:]+Lx[:-1])
        Lxmax=Lx[1:]
        Lxmin=Lx[:-1]
        
        Lxmid,LQ2mid=np.meshgrid(Lxmid,LQ2mid)
        Lxmin,LQ2min=np.meshgrid(Lxmin,LQ2min)
        Lxmax,LQ2max=np.meshgrid(Lxmax,LQ2max)
        dLx,dLQ2=np.meshgrid(dLx,dLQ2)
        
        #x =np.exp(Lxmid); Q2=np.exp(LQ2mid)
        
        """
        Conditions:
        1) the maximum Q2 allowed by conservation of momentum (cannot create more energy than is put in!)
        2) W2 must be greater than self.W2min to allow for a clean analysis of 1d QCFs in this process.
        """
        x =np.exp(Lxmin); Q2=np.exp(LQ2max)
        Q2max=(self.rs**2-par.M2)*x
        cond1=Q2max>=Q2
        
        x =np.exp(Lxmax); Q2=np.exp(LQ2min)
        W2=par.M2+Q2/x-Q2
        cond2=W2>=self.W2min
        acc=cond1*cond2
        
        self.Lxmid=Lxmid; self.LQ2mid=LQ2mid; 
        self.Lxmin=Lxmin; self.LQ2min=LQ2min; 
        self.Lxmax=Lxmax; self.LQ2max=LQ2max; 

        self.dLx=dLx; self.dLQ2=dLQ2;
        self.acc=acc


        
    def gen_xsection(self):
        """
        This function generates the cross sections for each of the entries in the (x,Q2) bins.
        The total cross section is the sum of each of the differential cross sections (weighted by logarithmic integral).
        The weights are the individual cross section per bin divided by the total integrated cross section.
        """
        
        x =np.exp(self.Lxmid); Q2=np.exp(self.LQ2mid)
        self.dxsec=np.zeros(x.shape)
        
        entries=[(a,b) for a in range(x.shape[0]) for b in range(x.shape[1])]
        for i,j in tqdm(entries):
            if self.acc[i,j]==False: continue
            self.dxsec[i,j]=self.idis.get_diff_xsec(x[i,j],Q2[i,j],self.rs,self.tar,'xQ2')
        integrand=self.dxsec*(x*self.dLx)*(Q2*self.dLQ2)*self.acc #--because of the logarithmic integral (because of the bounds)
        self.total_xsec=np.sum(integrand)
        self.weights=integrand/self.total_xsec
        #self.weights=np.ones(self.acc.shape)
        
    def gen_events(self,N,verb=False):
        """
        Generates the events of (x,Q2) for a given number of total events, N.
        Events governed by weights, which determines probability of finding an event in the phase space.
        Verb = Boolean, determines whether to plot (True) or not (False) the events in Q2 vs x.
        """
        n=np.array(self.weights*N,dtype=int)
        x=np.array([0])
        Q2=np.array([0])
        entries = [(a,b) for a in range(self.Lxmid.shape[0]) 
                         for b in range(self.Lxmid.shape[1])]
        for i,j in tqdm(entries):
            if self.acc[i,j]==False: continue        
            _x =np.exp(self.Lxmin[i,j]  + self.dLx[i,j] *np.random.uniform(0,1,n[i,j]))
            _Q2=np.exp(self.LQ2min[i,j] + self.dLQ2[i,j]*np.random.uniform(0,1,n[i,j]))
            x=np.append(x,_x)
            Q2=np.append(Q2,_Q2)
        x=x.reshape(-1,1)
        Q2=Q2.reshape(-1,1)
        evts=np.concatenate((x,Q2),axis=1)
        
        if verb:
            print(self.total_xsec)
            ax=py.subplot(111)
            ax.plot(np.log(evts[:,0]),np.log(evts[:,1]),'r.')
            ax.plot(self.Lxmid[self.acc],self.LQ2mid[self.acc],'k.')
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

        return evts[1: ]
