import numpy as np

class XSEC:
    
    def __init__(self,pdf,idis,params,rs,tar,W2min=10,option='xQ2'):

        """
        initializing the following inputs as these are needed and do not change:
            pdf  = instance of the PDF class -- the 1d QCF
            idis = instance of the idis class -- computes DIS cross sections
            params = collection of constants in the file params.py
            rs = "root s", square root of s = incoming center-of-mass momentum squared; type = float
            tar = target of the scattering experiment; type = string
            W2min = the minimum invariant mass of the outgoing debris of the reaction to cleanly perform an analysis of QCFs; type = float
            option = 'xQ2' or 'xy' for calculation of either dsigma/dxdQ2 or dsigma/dxdy, respectively.
        """
        self.pdf  = pdf
        self.idis = idis
        self.par = params
        self.rs = rs
        self.tar = tar
        self.W2min = W2min
        self.Q2min = self.par.mc2
        self.option=option
        
    def transform(self,alpha,beta):
        #--alpha,beta are between 0 and 1 and get mapped to physical (x,Q2)
        
        Q2min,Q2max = self.Q2min,self.rs**2-self.par.M2
        Q2 = np.exp(alpha * (np.log(Q2max)-np.log(Q2min)) + np.log(Q2min))

        xmax1 = Q2 / (self.W2min - self.par.M2 + Q2)
        xmax2 = np.ones(len(Q2))
        xmax = np.min([xmax1,xmax2],axis=0)

        xmin = Q2 / (self.rs**2 - self.par.M2)

        x = np.exp(beta * (np.log(xmax)-np.log(xmin)) + np.log(xmin))

        return x,Q2
        
    def inv_transform(self,x,Q2):
        #--takes in a physical (x,Q2) and maps to (alpha,beta) which are both in the range of 0 to 1
        
        Q2min = self.Q2min
        Q2max = self.rs**2 - self.par.M2
        alpha = (np.log(Q2) - np.log(Q2min)) / (np.log(Q2max) - np.log(Q2min))

        xmax1 = Q2 / (self.W2min - self.par.M2 + Q2)
        xmax2 = np.ones(len(Q2))
        xmax = np.min([xmax1,xmax2],axis=0)

        xmin = Q2 / (self.rs**2 - self.par.M2)

        beta = (np.log(x) - np.log(xmin)) / (np.log(xmax) - np.log(xmin))

        return alpha,beta
        
    def get_xsec(self,x,Q2,par):
        """
        Assumes physical (x,Q2)
        Computes a DIS cross section for a given combination of (x,Q2,parameters).
        inputs:
            x   = Bjorken variable; float
            Q2  = scale choice; float
            par = array of free parameters describing the 1d QCF
        """
        
        cond = True
        """
        Including the (x,Q2) conditional statements that only will provide cross sections in the allowed region.
        """
        
        Q2min = self.Q2min #--lower edge of the box
        Q2max = self.rs**2 - self.par.M2 #--upper edge of the box
        
        xmin = Q2min/(self.rs**2 - self.par.M2) #--left edge of the box
        xmax = 1 #--right edge of the box
        
        if (Q2 < Q2min) or (Q2 > Q2max) or (x < xmin) or (x > xmax):
            cond=False
        
        """
        New Q2max for a given x based on conservation of momentum.
        This condition sets the boundary of the 'triangle' of the box, omitting unphysical small x and large Q2 combinations.
        It eliminates almost half of the phase space.
        """
        Q2max = (self.rs**2 - self.par.M2) * x
        
        if Q2 > Q2max:
            cond = False
        
        """
        W2 is the invariant mass of the debris of the reaction.
        It must be larger than W2min, omitting extreme large x and small Q2 combinations.
        """
        W2 = self.par.M2 + Q2/x - Q2 #--invariant mass of the debris. Must be larger than W2min.
        
        if W2 < W2min:
            cond = False
            
        if cond:
        
            self.pdf.setup(par)
            xsec = self.idis.get_diff_xsec(x,Q2,self.rs,self.tar,self.option)

            """
            Cross sections cannot be negative. Here, this would be because of the parameters.
            """
            if xsec < 0: 
                return

            else:
                return xsec
    
if __name__=='__main__':
    import numpy as np
    from mpmath import fp
    import params as par
    import cfg
    from alphaS import ALPHAS
    from eweak import EWEAK
    from pdf import PDF
    from mellin import MELLIN
    from idis import THEORY

    
    mellin=MELLIN(npts=8)
    alphaS=ALPHAS()
    pdf = PDF(mellin,alphaS)
    eweak = EWEAK()
    idis = THEORY(mellin,pdf,alphaS,eweak)
    
    
    rs = 140.0
    tar = 'p'
    W2min = 10
    Q2min = par.mc2
    
    xsec = XSEC(pdf,idis,par,rs,tar,W2min=W2min,option='xQ2')
    
    par0 = pdf.get_current_par_array()
    
    x,Q2 = 0.1,10
    
    print(xsec.get_xsec(x,Q2,par0))
    
    n=20
    pmin=np.repeat([pdf.parmin], n,axis=0)
    pmax=np.repeat([pdf.parmax], n,axis=0)
    np.random.seed(12345)
    random=np.random.uniform(0,1,(n,len(pdf.parmin)))
    replicas = pmin + random*(pmax-pmin)
    
    for par_array in replicas:
        print(xsec.get_xsec(x,Q2,par_array))