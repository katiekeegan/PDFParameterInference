#--scipy stack
import numpy as np

#--torch
import torch

#--custom
import params as par
from pdf import PDF

class IDIS:

    def __init__(self):
        self.pdf = PDF()

    def get_diff_xsec(self, x, Q2 , rs, tar, option='xy'):
        """
        Evaluate the value of the unnormalized density at the x and Q^2

        :param x, Q^2: are the phase space coordinate points;
            specifically, the phase space of the outgoing electron
        :param rs: center of mass energy (related to collision energy),
            as it becomes bigger, coverage in x, Q^2 becomes bigger
        :param tar: 'p' for 'proton', or 'n' for neutron
        """
        y = Q2 / (rs**2 - par.M2) / x
        Yp = 1 + (1-y)**2
        Ym = 1-(1-y)**2
        K2 = (Yp+2*x**2*y**2*par.M2/Q2)
        KL = -y**2
        K3 = Ym*x
        alfa = alfa=1/137

        # norm is a prefactor of theory; not really a normalization factor
        if option == 'xy': #! diff in x,y
            norm = 2*np.pi*alfa**2/x/y/Q2

        elif option == 'xQ2':#! diff in x,Q2
            norm = 2*np.pi*alfa**2/x/y/Q2*y/Q2
        else:
            raise ValueError(
                f"Only xy or xQ2 values are accepted; received {option}!")

        xu=x*self.pdf.get_u(x,Q2)
        xd=x*self.pdf.get_d(x,Q2)

        eU2=4/9; eD2=1/9

        if tar=='p':
            F2=eU2*xu+eD2*xd
            FL=0
            F3=0
        elif tar=='n':
            F2=eU2*xd+eD2*xu
            FL=0
            F3=0
        else:
            raise ValueError(
                f"Only 'n' or 'p' values are accepted; received {tar}!")

        xsec = norm * (K2*F2 + KL*FL + K3*F3)
        return xsec


