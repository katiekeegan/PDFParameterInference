import numpy as np
from mpmath import fp
import params as par 
import cfg
from alphaS import ALPHAS
from eweak import EWEAK
from pdf import PDF
from mellin import MELLIN

class BASE:
    """
    Class that compiles functions used in the calculation of deep inelastic scattering (DIS).
    """

    def setup(self):
  
        ############################################
        # abbreviations
        ############################################
        zeta2=fp.zeta(2)
        zeta3=fp.zeta(3)
        N   = self.mellin.N 
        NP1 = N + 1
        NP2 = N + 2
        NM1 = N - 1
        NS=N*N
        psi=lambda i,N: fp.psi(i,complex(N.real,N.imag))
        S1 = np.array([fp.euler  + psi(0,n+1) for n in N])
        S2 = np.array([zeta2- psi(1,n+1) for n in N])
        S1S=S1**2
  
        ############################################
        # hard coeffs
        ############################################
        orders=2
  
        self.C2Q = np.zeros((orders,N.size),dtype=complex)
        self.C2G = np.zeros((orders,N.size),dtype=complex)
        self.CLQ = np.zeros((orders,N.size),dtype=complex)
        self.CLG = np.zeros((orders,N.size),dtype=complex)
        self.C3Q = np.zeros((orders,N.size),dtype=complex)
        self.PC1Q = np.zeros((orders,N.size),dtype=complex)
        self.PC1G = np.zeros((orders,N.size),dtype=complex)
        self.AHG1LOG = np.zeros((orders,N.size),dtype=complex)
  
        self.C2Q[0]  = np.ones(N.size)  
        self.C3Q[0]  = np.ones(N.size)  
        self.PC1Q[0] = np.ones(N.size)  
  
        CF=4.0/3.0
        TR=0.5
        # Nucl. Phys. B192 (1981) 417
        # Ellis book
        self.C2Q[1]  = CF*(2.0*S1S - 2.0*S2 + 3.0*S1 - 2.0*S1/N/NP1 + 3.0/N + 4.0/NP1 + 2.0/NS - 9.0)
        self.C2G[1]  = -2*TR*(S1*(NS + N + 2.0)/N/NP1/NP2 + 1.0/N - 1.0/NS - 6.0/NP1 + 6.0/NP2)  
  
        self.CLQ[1]  = CF*4/NP1  
        self.CLG[1]  = 8.0*TR/NP1/NP2
  
        self.C3Q[1]  = CF*(2.0*S1S - 2.0*S2 + 3.0*S1 - 2.0*S1/N/NP1 + 1.0/N + 2.0/NP1 + 2.0/NS - 9.0)
        self.AHG1LOG[1] = 2.*(1./N - 2./NP1 + 2./NP2) #NL0 glue coeffcient for F2^{hq}
  
        # Nucl. Phys. B192 (1981) 417??? need check
        self.PC1Q[1] = CF*(2*S1S-2*S2+2*S1/NP1-2*S1/N+3*S1-2/N/NP1+3/N+2/NS-9)
        self.PC1G[1] = 2*NM1*(1-N-N*S1)/NS/NP1
      
    def get_aX(self,aX,i,Q2,channel='all'):
  
        ############################################
        # couplings  (see page 18 of Pedro's thesis)
        # {q} = {u,d,s,c,b} 
        ############################################
  
        alpha = self.eweak.get_alpha(Q2)
        sin2w = self.eweak.get_sin2w(Q2)
        KQ2 = par.GF*par.mZ2/(2*2**0.5 * np.pi * alpha)* Q2/(Q2+par.mZ2)
  
        Ve=-0.5+2*sin2w
        Ae=-0.5
        Ve2pAe2=Ve*Ve+Ae*Ae
  
        eu=2.0/3.0
        Vu=0.5-2*eu*sin2w
        Au=0.5
        Vu2pAu2=Vu*Vu+Au*Au
  
        ed=-1.0/3.0
        Vd=-0.5-2*ed*sin2w
        Ad=-0.5
        Vd2pAd2=Vd*Vd+Ad*Ad
  
        #--for individual channels, conventions are taken from PDG
        #--pdg.lbl.gov/2019/reviews/rpp2019-rev-structure-functions.pdf
        #--see Eq. (18.18)
        if aX=='ap':
            if i==1 or i==4:
                if channel=='all':  return eu*eu - 2*eu*Ve*Vu*KQ2 + Ve2pAe2*Vu2pAu2*KQ2**2
                elif channel=='gg': return eu*eu
                elif channel=='ZZ': return Vu2pAu2
                elif channel=='gZ': return 2*eu*Vu
            elif i==2 or i==3 or i==5:
                if channel=='all':  return ed*ed - 2*ed*Ve*Vd*KQ2 + Ve2pAe2*Vd2pAd2*KQ2**2
                elif channel=='gg': return ed*ed
                elif channel=='ZZ': return Vd2pAd2
                elif channel=='gZ': return 2*ed*Vd
        if aX=='am':
            if i==1 or i==4:
                if channel=='all':  return -2*eu*Ae*Au*KQ2 + 4*Ve*Ae*Vu*Au*KQ2**2
                elif channel=='gg': return 0.0 
                elif channel=='ZZ': return 2*Vu*Au
                elif channel=='gZ': return 2*eu*Au
            elif i==2 or i==3 or i==5:
                if channel=='all':  return -2*ed*Ae*Ad*KQ2 + 4*Ve*Ae*Vd*Ad*KQ2**2
                elif channel=='gg': return 0.0 
                elif channel=='ZZ': return 2*Vd*Ad
                elif channel=='gZ': return 2*ed*Ad

class THEORY(BASE):
    """
    Class for the computation of inclusive deep inelastic scattering (IDIS or just DIS).
    Computes the differential cross section for a given (x,Q2).
    Computes the convolution of a perturbatively calculable, known hard coefficient function with the 1d QCFs.
    Computation is done primarily in Mellin space, where a convolution becomes a simple product.
    The cross section in momentum space is the inversion of the coefficient functions with the 1d QCFs.
    
    Has class inheritance from BASE, which is defined above.
    """
    
    def __init__(self,mellin,pdf,alphaS,eweak):
        
        """
        inputs:
            mellin = instance of the mellin class -- defines the Mellin contour
            pdf    = instance of the pdf class -- this describes the 1d QCF
            alphaS = instance of the alphaS class -- the strong coupling constant
            eweak  = instance of the eweak class -- electromagnetic coupling
        """

        self.mellin  = mellin
        self.pdf     = pdf
        self.alphaS  = alphaS
        self.eweak   = eweak
        self.order   = cfg.idis_order
        self.setup()

    #--twist 2 unpolarized structure functions 
  
    def get_T2CFX(self,stf,nucleon,Q2):
        """
        Function to compute a twist 2 structure function convolution in Mellin space.
        Hard coefficients convoluted with 1d QCFs, here called self.pdf.
        CF(2,L,3) = F2/x,FL/x,F3
        inputs:
            stf = type of structure function, string
            nucleon = type of nucleon, string
            Q2 = scale choice, float
        """
 
        self.pdf.evolve(Q2)

        g = self.pdf.storage[Q2]['g']    
        Nf=self.alphaS.get_Nf(Q2) 
        a=self.alphaS.get_a(Q2) 

        if stf=='F2':
            CQ = self.C2Q[0] + a*self.order*self.C2Q[1] 
            CG = self.C2G[0] + a*self.order*self.C2G[1]
            q=np.copy(self.pdf.storage[Q2]['qp'])   
            aX='ap'
 
        elif stf=='FL':
            CQ = a*self.order*self.CLQ[1]
            CG = a*self.order*self.CLG[1]
            q=np.copy(self.pdf.storage[Q2]['qp'])
            aX='ap'
  
        elif stf=='F3':
            CQ = self.C3Q[0] + a*self.order*self.C3Q[1]
            CG = 0
            q=np.copy(self.pdf.storage[Q2]['qm']) 
            aX='am'

        if nucleon=='n':
            qup=np.copy(q[1])
            qdn=np.copy(q[2])
            q[1]=qdn
            q[2]=qup

        
        FX  = np.zeros(self.mellin.N.size,dtype=complex) 
        for i in range(1,Nf+1):
            aXval = self.get_aX(aX,i,Q2)
            FX+=aXval*(CQ*q[i] + 2*CG*g)
 
        return FX

    def get_T2CWX(self,stf,nucleon,Q2,sign):  
        """
        Function to compute a twist 2 structure function convolution in Mellin space for charged current.
        Hard coefficients convoluted with 1d QCFs, here called self.pdf.
        inputs:
            stf = type of structure function, string
            nucleon = type of nucleon, string
            Q2 = scale choice, float
            sign = +1 or -1, the charge of the intermediate boson
        CF(2,L,3) = W2/x,WL/x,W3  
        """ 
        self.pdf.evolve(Q2)
        g =self.pdf.storage[Q2]['g'] 
        Nf=self.alphaS.get_Nf(Q2) 
        a=self.alphaS.get_a(Q2)  
  
        if stf=='W2':
            CQ = self.C2Q[0] + a*self.order*self.C2Q[1] 
            CG = self.C2G[0] + a*self.order*self.C2G[1]
  
        elif stf=='WL':
            CQ = a*self.order*self.CLQ[1]
            CG = a*self.order*self.CLG[1]
  
        elif stf=='W3':
            CQ = self.C3Q[0] + a*self.order*self.C3Q[1]
            CG = 0
  
        u =self.pdf.storage[Q2]['u']
        d =self.pdf.storage[Q2]['d']
        s =self.pdf.storage[Q2]['s']
        c =self.pdf.storage[Q2]['c']
        b =self.pdf.storage[Q2]['b']
        ub=self.pdf.storage[Q2]['ub']
        db=self.pdf.storage[Q2]['db']
        sb=self.pdf.storage[Q2]['sb']
        cb=self.pdf.storage[Q2]['cb']
        bb=self.pdf.storage[Q2]['bb']
  
        U =(CQ*u  + CG*g)
        D =(CQ*d  + CG*g) + (CQ*s  + CG*g) 
        UB=(CQ*ub + CG*g)
        DB=(CQ*db + CG*g) + (CQ*sb + CG*g)
  
        if Nf>3:
            U += CQ*c  + CG*g
            UB+= CQ*cb + CG*g
  
        # cannot produce a top
        #if Nf>4:
        #  D+=  CQ*b  + CG*g
        #  DB+= CQ*bb + CG*g 
 
        #--factor of 2 follows PDG definition
        #--pdg.lbl.gov/2019/reviews/rpp2019-rev-structure-functions.pdf
        #--see equation (18.19)
        if sign==+1:
            if   stf=='W2': return 2*(D+UB)
            elif stf=='WL': return 2*(D+UB)
            elif stf=='W3': return 2*(D-UB)
        elif sign==-1:
            if   stf=='W2': return 2*(U+DB)
            elif stf=='WL': return 2*(U+DB)
            elif stf=='W3': return 2*(U-DB)

    #--nucleon  structure functions
  
    def get_FXN(self,x,Q2,stf,nucleon):
        """
        Calculates a nucleon structure function.
        inputs:
            x = Bjorken variable, float
            Q2 = scale choice, float
            stf = type of structure function (F2, FL, or F3), string
            nucleon = type of nucleon ('p' or 'n'), string
        """
        if   stf=='F2': FX= x*self.get_T2CFX('F2',nucleon,Q2) 
        elif stf=='FL': FX= x*self.get_T2CFX('FL',nucleon,Q2)
        elif stf=='F3': FX=   self.get_T2CFX('F3',nucleon,Q2)
        return self.mellin.invert(x,FX)
  
    #--charge current 
  
    def get_WXN(self,x,Q2,stf='W2+',nucleon='p'):
        """
        Calculates a charged current structure function.
        inputs:
            x = Bjorken variable, float
            Q2 = scale choice, float
            stf = type of structure function (W2, WL, or W3), string
            nucleon = type of nucleon ('p' or 'n'), string
        """
        if '+' in stf:  sign=1
        if '-' in stf:  sign=-1
        if   'W2' in stf: WX= x*self.get_T2CWX('W2',nucleon,Q2,sign)
        elif 'WL' in stf: WX= x*self.get_T2CWX('WL',nucleon,Q2,sign)
        elif 'W3' in stf: WX=   self.get_T2CWX('W3',nucleon,Q2,sign)
        WX=self.mellin.invert(x,WX)
        if ht:  WX+=0
        return WX

    #--nuclear structure functions
        
    def get_FXA(self,x,Q2,stf,nucleus):
        """
        Calculates a nuclear structure function.
        inputs:
            x = Bjorken variable, float
            Q2 = scale choice, float
            stf = type of structure function (F2, FL, or F3), string
            nucleus = type of nucleus, string
        """

        if   nucleus=='d': p, n = 1,1
        elif nucleus=='h': p, n = 2,1
        elif nucleus=='t': p, n = 1,2

        FXp   = self.get_FXN(x,Q2,stf,nucleon='p')
        FXn   = self.get_FXN(x,Q2,stf,nucleon='n')
        return (p*FXp+n*FXn)#/(p+n)

    #--differntial cross sections

    def get_diff_xsec(self,x,Q2,rs,tar,option='xy'):
        """
        This function computes the differential cross section for DIS.
        Computes kinematic factors and normalizations,
        and calls the various structure functions, F2, FL, and F3, all of which are dependent on the 1d QCF.
        inputs:
            x  = Bjorken variable, float
            Q2 = scale choice, float
            rs = "root s", square root of s = incoming center-of-mass momentum squared; type = float
            tar = target of the experiment; type = string
            option = 'xQ2' or 'xy' for calculation of either dsigma/dxdQ2 or dsigma/dxdy, respectively.
        """
        
        y=Q2/(rs**2-par.M2)/x
        Yp=1+(1-y)**2
        Ym=1-(1-y)**2
        K2=(Yp+2*x**2*y**2*par.M2/Q2)
        KL=-y**2
        K3=Ym*x
        alfa=self.eweak.get_alpha(Q2)

        if option=='xy': #! diff in x,y
            norm=2*np.pi*alfa**2/x/y/Q2

        if option=='xQ2':#! diff in x,Q2
            norm=2*np.pi*alfa**2/x/y/Q2*y/Q2 

        if tar=='p':   
            F2=self.get_FXN(x,Q2,'F2','p')
            FL=self.get_FXN(x,Q2,'FL','p')
            F3=self.get_FXN(x,Q2,'F3','p')
        else:
            F2=self.get_FXA(x,Q2,'F2',tar)
            FL=self.get_FXA(x,Q2,'FL',tar)
            F3=self.get_FXA(x,Q2,'F3',tar)
   
        xsec=norm*(K2*F2 + KL*FL + K3*F3)
        return xsec

if __name__=='__main__':

    mellin = MELLIN(npts=8)
    alphaS = ALPHAS()
    eweak  = EWEAK()
    pdf    = PDF(mellin,alphaS)
    idis   = THEORY(mellin,pdf,alphaS,eweak)

    x=0.1; Q2=10; rs=140; tar='p'; option='xQ2'
    print(idis.get_diff_xsec(x,Q2,rs,tar,option))

    x=0.1; Q2=10; rs=140; tar='d'; option='xQ2'
    print(idis.get_diff_xsec(x,Q2,rs,tar,option))






