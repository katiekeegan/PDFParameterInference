import numpy as np
import params as par 

class EWEAK:
    """
    This class calculates the electroweak coupling, alpha,
    which runs as a function of a scale, Q2.
    alpha is needed in part to describe scattering of electrically charged particles.
    There are no free parameters.
    """

    def __init__(self):
        self.ml2 = np.array([par.me2,par.mmu2,par.mtau2])
        self.mq2 = np.array([par.mu2,par.md2,par.ms2,par.mc2,par.mb2])
        self.eq  = np.array([2./3, -1./3, -1./3, 2./3, -1./3])
        self.gvl = (-1.0)*(-1.0 + 4*par.s2w)/2
        self.i3q = np.array([0.5, -0.5, -0.5, 0.5, -0.5])
        self.gvq = self.i3q - 2*self.eq*par.s2w
        self.alfa = par.alfa
        self.s2w = par.s2w
        self.c2w = par.c2w
        self.logc2w = np.log(par.c2w)
        self.alpha  = {}
        self.sin2w  = {}
 
    def get_PIlepton(self,Q2,flag):
        # lepton loop contribution
        if Q2 >= 2.e-8:
            u = 4*self.ml2/Q2
            su = np.sqrt(1.0+u)
            tmp = ( (1.0-u/2)*su*np.log((su+1.0)/(su-1.0)) + u - 5.0/3 )/3
        else:
            tmp = Q2/15./self.ml2
        if flag=='gg':   return np.sum(tmp)
        elif flag=='gZ': return np.sum((-np.log(par.mZ2/self.ml2)/3 + tmp)*self.gvl)

    def get_PIquark(self,Q2,flag):
        # quark loop contribution
        if Q2 >= 2.e-5:
            u = 4*self.mq2/Q2
            su = np.sqrt(1.0+u)
            tmp = ( (1.0-u/2)*su*np.log((su+1.0)/(su-1.0)) + u - 5.0/3 )/3
        else:
            tmp = Q2/15./self.mq2
        if flag=='gg':   return np.sum(3*tmp*self.eq**2)
        elif flag=='gZ': return np.sum(3*(-np.log(par.mZ2/self.mq2)/3 + tmp)*self.gvq*self.eq)

    def get_kapb(self,Q2):
        # boson loop contribution to kappa
        if Q2 >= 1.0:
            z = par.mW2/Q2
            p = np.sqrt(1.0 + 4*z)
            logp = np.log((p+1.0)/(p-1.0))
            PIb = 1./18 \
                + ( -1.0 - 42*self.c2w ) * self.logc2w/12 \
                - ( self.c2w*(7.0 - 4*z) + (1.0 + 4*z)/6 ) * ( p*logp/2 - 1.0 ) \
                - z * ( 0.75 - z + p*(z-1.5)*logp + (2.0-z)*z*logp**2 )
            return -(self.alfa/np.pi)/(2*self.s2w) * PIb
        else:      
            return - self.alfa*(-43.0 - 444*self.c2w)*Q2 / (1440*par.mW2*np.pi*self.s2w) \
                     + self.alfa*(-16.0 - 12*self.c2w + 3*self.logc2w + 126*self.c2w*self.logc2w) \
                       / (72*np.pi*self.s2w)

    def get_alpha(self,Q2):
        # running of alpha_em
        if Q2 not in self.alpha:
            dalpl = (self.alfa/np.pi) * self.get_PIlepton(Q2,'gg')
            dalpq = (self.alfa/np.pi) * self.get_PIquark(Q2,'gg')
            self.alpha[Q2] = self.alfa/(1.0-dalpl-dalpq)
        return self.alpha[Q2]

    def get_sin2w(self,Q2):

        # running of sin2w
        if Q2 not in self.sin2w:
            kapl = -(self.alfa/np.pi)/(2*self.s2w) * self.get_PIlepton(Q2,'gZ')
            kapq = -(self.alfa/np.pi)/(2*self.s2w) * self.get_PIquark(Q2,'gZ')
            kappa = 1.0 + kapl + kapq + self.get_kapb(Q2)
            self.sin2w[Q2] = kappa * par.s2w
        return self.sin2w[Q2]

if __name__=='__main__':

    eweak=EWEAK()
    print(1/eweak.get_alpha(10.0))


