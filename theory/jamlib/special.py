from mpmath import fp
from scipy.special import gamma
import numpy as np

def _get_psi(i,N):
    return fp.psi(i,complex(N.real,N.imag))

def get_psi(i,N):
    return np.array([_get_psi(i,n) for n in N],dtype=complex)

def beta(a,b):
    return gamma(a)*gamma(b)/gamma(a+b)






