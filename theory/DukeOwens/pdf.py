#--scipy stack
import numpy as np

#--torch
import torch

#--custom
import params as par

class PDF:

    def __init__(self):

        self.lam2 = 0.4
        self.Q02 = 1
        self.L = torch.tensor(np.log(self.Q02/self.lam2))

        # Parameters of the up/down quarks (par_u and par_d, respectively)
        # These are created as ground truth and create synthetic scenarios
        # par_u is for up-quark
        self.par_u = torch.tensor(
            [
                1.12995643e-01,  2.84816039e-02,  3.49711182e-02, -1.07992996e+00,
                -8.67724315e-02,  2.67006979e-02, -3.42002556e-01,  2.16083133e+00,
                -7.06215971e-02,  5.99208979e+01, -3.80095917e+01,  6.96188257e+00,
                -1.93266172e+02,  1.76589359e+02, -3.96195694e+01,  1.62782187e+02,
                -1.79165065e+02,  4.60385729e+01
             ],
            requires_grad=True,
        )
        # par_d is for down-quark
        self.par_d = torch.tensor(
            [
                -6.26740092e-02,  3.54363691e-01, -1.06902973e-01, -1.20394986e+00,
                 1.55339980e-01, -7.87939114e-02,  2.75190189e-01,  3.99743001e+00,
                -8.40757936e-01,  4.31272688e+01, -3.26453695e+01,  7.87648066e+00,
                -1.02351414e+02,  1.08446491e+02, -2.81885597e+01,  2.58442965e+01,
                -4.35307825e+01,  1.22755848e+01
            ],
            requires_grad=True,
        )

    def get_s(self,Q2):
        return torch.log(torch.log(Q2/self.lam2)/self.L)

    def get_parQ2(self,par,Q2):
        s = self.get_s(Q2)
        return par[0] + par[1] * s + par[2] * s**2

    def get_pdf(self, x ,Q2 , par):
        """
        Evaluate the value of the PDF for a given even (x, Q2), and instances of the physics parameter `par`

        ..note::
            This parameterization guarantees differentiability of the PDF w.r.t both x and Q^2
        """
        A=self.get_parQ2(par[:3], Q2)
        a=self.get_parQ2(par[3:6], Q2)
        b=self.get_parQ2(par[6:9], Q2)
        c=self.get_parQ2(par[9:12], Q2)
        d=self.get_parQ2(par[12:15], Q2)
        e=self.get_parQ2(par[15:18], Q2)
        # This is the DukenOwens parameterization (x and Q^2 dependce)
        return A * x**a * (1-x)**b * (1 + c*x + d * x**2 + e * x**3)

    # The following two interfaces `get_u` and `get_d` are just lazy ways of calling `get_pdf` with the two parameters
    def get_u(self, x, Q2):
        return self.get_pdf(x, Q2, self.par_u)

    def get_d(self, x, Q2):
        return self.get_pdf(x, Q2, self.par_d)





