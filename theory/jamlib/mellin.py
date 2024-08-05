import numpy as np

class MELLIN:
    """
    This class sets up the Mellin contour and inverts Mellin moments back into x-space.
    Mellin transform is defined as: F(N) = \int_0^1 dx x^{N-1} f(x).
    """

    def __init__(self,npts=8,extended=False,c=None):
        """
        Mellin contour is parametrized by N = c + Z * exp(i * phi).
        c is on the real axis to the right of the right-most pole.
        phi ensures that the contour encloses the poles.
        inputs:
            npts = number of Gaussian points in each interval of the znodes; type = integer
            extended = increase (True) or not (False) the range of the Mellin contour
            c = real axis intercept
        """

        #--gen z and w values along coutour
        x,w=np.polynomial.legendre.leggauss(npts)
        znodes=[0,0.1,0.3,0.6,1.0,1.6,2.4,3.5,5,7,10,14,19,25,32,40,50,63]
        #if extended: znodes.extend([70,80,90,100,110,120,130])
        if extended: znodes.extend([70,80,90,100])

        ## the following two keys are useful when truncating the Mellin contour (in triple Mellin)
        self.n_node  = len(znodes)
        self.density = npts

        Z,W,JAC=[],[],[]
        for i in range(len(znodes)-1):
            a,b=znodes[i],znodes[i+1]
            Z.extend(0.5*(b-a)*x+0.5*(a+b))
            W.extend(w)
            JAC.extend([0.5*(b-a) for j in range(x.size)])
        Z=np.array(Z)
        #--globalize
        self.W=np.array(W)
        self.Z=Z
        self.JAC=np.array(JAC)
        #--gen mellin contour
        if c==None: c=1.9
        phi=3.0/4.0*np.pi
        self.N=c+Z*np.exp(complex(0,phi))
        self.phase= np.exp(complex(0,phi))
        #print 'MELL:',len(self.W),len(self.JAC)

    def invert(self,x,F):
        """
        Function to invert from Mellin space to x-space
        inputs:
            x = conjugate to the Mellin N; type = float
            F = array of Mellin moments; type = array(complex float); len(F) = len(self.N)
        """
        #print 'MELL:',len(self.W),len(self.JAC)
        return np.sum(np.imag(self.phase * x**(-self.N) * F)/np.pi * self.W * self.JAC)
    
    def invert_ein(self,xs,F):
        """
        Function to invert from Mellin space to x-space.
        Same as above, but allows to input an array of xs instead of a single float.
        inputs:
            x = conjugate to the Mellin N; type = array
            F = array of Mellin moments; type = array(complex float); len(F) = len(self.N)
        """
        print(xs.shape,xs.shape[0])
        Fs = np.tile(F,(xs.shape[0],1,1))
        Ns = np.tile(self.N,(xs.shape[0],1))
        xss = np.tile(xs,(self.N.shape[0],1)).transpose()
        pows=np.power(xss,-Ns)
        pdfs=np.einsum('ik,ijk->ijk',pows,Fs)
        pdfs=np.einsum('ijk,k,k->ij',np.imag(self.phase*pdfs),self.W,self.JAC)/np.pi
        return pdfs


if __name__=='__main__':

    from scipy.special import gamma

    mell=MELLIN(8)
    a=-1.8
    b=6.0
    N=mell.N

    mom=gamma(N+a)*gamma(b+1)/gamma(N+a+b+1)
    X=10**np.linspace(-5,-1,10)
    f=lambda x: x**a*(1-x)**b
    for x in X:
        print ('x=%10.4e  f=%10.4e  inv=%10.4e'%(x,f(x),mell.invert(x,mom)))






