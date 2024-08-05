import numpy as np
import params as par 
import cfg

class ALPHAS:
    """
    Base class to calculate the running of the strong coupling, alpha_S
    The derivative of alpha_S with respect to a scale, Q2, are the known beta functions, which depend on the strong coupling.
    The alpha_S is solved for using Runge-Kutta integration.
    Alpha_S is used in the calculation of QCD observables.
    """

    def __init__(self):
        
        self.beta=np.zeros((7,3))
        """
        The number of active quark flavors (Nf), depends on the scale Q2.
        The ab, ac, and a0 are boundary conditions at different Nf values.
        """
        for Nf in range(3,7): 
            self.beta[Nf,0]=11.0-2.0/3.0*Nf 
            self.beta[Nf,1]=102.-38.0/3.0*Nf 
            self.beta[Nf,2]=2857.0/2.0-5033.0/18.0*Nf+325.0/54.0*Nf**2 
  
            self.aZ  = par.alphaSMZ/(4*np.pi)
            self.ab=self.evolve_a(par.mZ2,self.aZ,par.mb2,5)
            self.ac=self.evolve_a(par.mb2,self.ab,par.mc2,4)
            self.a0=self.evolve_a(par.mc2,self.ac,cfg.Q20,3)

        #--can be stored because there are no free parameters in this calculation
        self.storage={}
        
    def get_Nf(self,Q2):
        """
        Returns the number of active flavors used for the calculation; input is a float Q2
        """
        Nf=3
        if Q2>=(par.mc2): Nf+=1
        if Q2>=(par.mb2): Nf+=1
        return Nf
  
    def beta_func(self,a,Nf):
        """
        Gets proper beta function for an input, Nf, and scaled strong coupling a (a=alphaS/4pi).
        """
        betaf = -self.beta[Nf,0]
        if cfg.alphaS_order>=1: betaf+=-a*self.beta[Nf,1]
        if cfg.alphaS_order>=2: betaf+=-a*self.beta[Nf,2]
        return betaf*a**2
  
    def evolve_a(self,Q20,a,Q2,Nf):
        # Runge-Kutta implemented in pegasus  
        LR = np.log(Q2/Q20)/20.0
        for k in range(20):
            XK0 = LR * self.beta_func(a,Nf)
            XK1 = LR * self.beta_func(a + 0.5 * XK0,Nf)
            XK2 = LR * self.beta_func(a + 0.5 * XK1,Nf)
            XK3 = LR * self.beta_func(a + XK2,Nf)
            a+= (XK0 + 2.* XK1 + 2.* XK2 + XK3) * 0.166666666666666
        return a
  
    def get_a(self,Q2):
        """
        This calls the evolution for particular Q2, which also determines the number of flavors.
        """
        if Q2 not in self.storage:
            if par.mb2<=Q2:
                Q20,a0,Nf=par.mb2,self.ab,5
            elif par.mc2<=Q2 and Q2<par.mb2: 
                Q20,a0,Nf=par.mc2,self.ac,4
            elif Q2<par.mc2:
                Q20,a0,Nf=cfg.Q20,self.a0,3
            self.storage[Q2]=self.evolve_a(Q20,a0,Q2,Nf)
        return self.storage[Q2]
  
    def get_alphaS(self,Q2):
        """
        This is the main output of the class.
        
        input: float Q2
        return: strong coupling constant, alphaS
        """
        return self.get_a(Q2)*4*np.pi


if __name__=='__main__':

    aS=ALPHAS()

    mc2=par.mc2
    mb2=par.mb2
    mZ2=par.mZ2


    print('========================')
    print('test alphaS running')
    print('========================')
    print('Q2=1           alphaS=%0.5f'%aS.get_alphaS(1.0))
    print('Q2=(1+mc2)/2   alphaS=%0.5f'%aS.get_alphaS(0.5*(1.0+mc2)))
    print('Q2=mc2         alphaS=%0.5f'%aS.get_alphaS(mc2))
    print('Q2=(mc2+mb2)/2 alphaS=%0.5f'%aS.get_alphaS(0.5*(mc2+mb2)))
    print('Q2=mb2         alphaS=%0.5f'%aS.get_alphaS(mb2))
    print('Q2=(mb2+mZ2)/2 alphaS=%0.5f'%aS.get_alphaS(0.5*(mb2+mZ2)))
    print('Q2=mZ2         alphaS=%0.5f'%aS.get_alphaS(mZ2))
  










