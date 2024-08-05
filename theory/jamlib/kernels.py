import numpy as np
from mpmath import fp
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.filterwarnings("ignore", message="invalid value encountered in add")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")


class KERNELS:
    """
    Class to compile the proper splitting functions needed for evolution of 1d QCFs.
    """

    def __init__(self,mell,Type='upol'):
        """
        Input is an instance of the mellin class.
        """

        self.D={}
        self.D['N']=mell.N
        self.D['nflav']=6+1
        self.D['norder']=2+1
        self.D['Nsize']=mell.N.size

        self.set_abbreviations()
        self.LO_unpolarized_splitting_functions()
        self.NLO_unpolarized_splitting_functions()
        self.LO_polarized_splitting_functions()
        self.NLO_polarized_splitting_functions()
        self.LO_unpol_timelike_splitting_functions()
        self.NLO_unpol_timelike_splitting_functions()

        if Type=='upol': self.load_unpolarized_spl()
        elif Type=='pol': self.load_polarized_spl()
        elif Type=='upol_ff': self.load_unp_FF_spl()

    def set_abbreviations(self):

        D=self.D

        D['CA']=3.0
        D['CF']=4.0/3.0
        D['TR']=0.5

        zeta2=fp.zeta(2)
        zeta3=fp.zeta(3)

        N   = D['N']
        N1  = N + 1
        N2  = N + 2
        N3  = N + 3
        N4  = N + 4
        N5  = N + 5
        N6  = N + 6
        N7  = N + 7
        N8  = N + 8

        NM = N - 1.
        NM2 = N - 2.
        NMS = NM**2
        NMT = NM**3

        NS  = N**2
        NT  = N**3
        NFO = N**4
        NFI = N**5
        NSI = N**6
        NSE = N**7
        NE  = N**8
        NN  = N**9

        N1S = N1**2
        N1T = N1**3

        N2S = N2**2
        N2T = N2**3

        psi=lambda i,_N: fp.psi(i,complex(_N.real,_N.imag))

        S1f = lambda _N: fp.euler + psi(0,_N+1)
        S2f = lambda _N: zeta2 - psi(1,_N+1)

        S1 = np.array([S1f(n) for n in N])
        D['S1']=S1
        S2 = np.array([S2f(n) for n in N])

        S11 = S1  + 1/N1 
        S12 = S11 + 1/N2
        S13 = S12 + 1/N3
        S14 = S13 + 1/N4
        S15 = S14 + 1/N5
        S16 = S15 + 1/N6

        S21 = S2 + 1/N1S

        SPMOM = 1.0000  * (zeta2 - S1/  N ) / N  \
              -0.9992 * (zeta2 - S11/ N1) / N1 \
              +0.9851 * (zeta2 - S12/ N2) / N2 \
              -0.9005 * (zeta2 - S13/ N3) / N3 \
              +0.6621 * (zeta2 - S14/ N4) / N4 \
              -0.3174 * (zeta2 - S15/ N5) / N5 \
              +0.0699 * (zeta2 - S16/ N6) / N6 


        D['PSI(N/2)']=np.array([psi(0,n) for n in N/2])
        D['PSI(N1/2)']=np.array([psi(0,n) for n in N1/2])
        D['PSI(N2/2)']=np.array([psi(0,n) for n in N2/2])
        D['DPSI(N1/2,1)']=np.array([psi(1,n) for n in N1/2])
        D['DPSI(N2/2,1)']=np.array([psi(1,n) for n in N2/2])
        D['DPSI(N1/2,2)']=np.array([psi(2,n) for n in N1/2])
        D['DPSI(N2/2,2)']=np.array([psi(2,n) for n in N2/2])

        G1 = np.array([0.5*(psi(1,(n+1)/2.)-psi(1,n/2.)) for n in N])
        G11 = np.array([0.5*(psi(1,(n+2)/2.)-psi(1,(n+1.)/2.)) for n in N])

        SLC = - 5./8. * zeta3
        SLV = - zeta2/2.*(D['PSI(N1/2)']-D['PSI(N/2)'])+S1/NS+SPMOM
        SSCHLM = SLC - SLV
        SSTR2M = zeta2 - D['DPSI(N1/2,1)']
        SSTR3M = 0.5 * D['DPSI(N1/2,2)'] + zeta3
        SSCHLP = SLC + SLV
        SSTR2P = zeta2 - D['DPSI(N2/2,1)']
        SSTR3P = 0.5 * D['DPSI(N2/2,2)'] + zeta3

        D['PNMA'] = ( 16.* S1 * (2.* N + 1.) / (NS * N1S) +\
                  16.* (2.* S1 - 1./(N * N1)) * ( S2 - SSTR2M ) +\
                  64.* SSCHLM + 24.* S2 - 3. - 8.* SSTR3M -\
                  8.* (3.* NT + NS -1.) / (NT * N1T) +\
                  16.* (2.* NS + 2.* N +1.) / (NT * N1T) ) * (-0.5)
        D['PNPA'] = ( 16.* S1 * (2.* N + 1.) / (NS * N1S) +\
                  16.* (2.* S1 - 1./(N * N1)) * ( S2 - SSTR2P ) +\
                  64.* SSCHLP + 24.* S2 - 3. - 8.* SSTR3P -\
                  8.* (3.* NT + NS -1.) / (NT * N1T) -\
                  16.* (2.* NS + 2.* N +1.) / (NT * N1T) ) * (-0.5)
        D['PNSB'] = ( S1 * (536./9. + 8.* (2.* N + 1.) / (NS * N1S)) -\
                  (16.* S1 + 52./3.- 8./(N * N1)) * S2 - 43./6. -\
                  (151.* NFO + 263.* NT + 97.* NS + 3.* N + 9.) *\
                  4./ (9.* NT * N1T) ) * (-0.5)
        D['PNSC'] = ( -160./9.* S1 + 32./3.* S2 + 4./3. +\
                  16.* (11.* NS + 5.* N - 3.) / (9.* NS * N1S) ) * (-0.5)


        D['PPSA'] = (5.* NFI + 32.* NFO + 49.* NT + 38.* NS + 28.* N + 8.) \
             / (NM * NT * N1T * N2S) * 2.

        D['PQGA'] = (-2.* S1 * S1 + 2.* S2 - 2.* SSTR2P) \
               * (NS + N + 2.) / (N * N1 * N2) \
             + (8.* S1 * (2.* N + 3.)) / (N1S * N2S)\
             + 2.* (NN + 6.* NE + 15. * NSE + 25.* NSI + 36.* NFI\
               + 85.* NFO + 128.* NT + 104.* NS + 64.* N + 16.)\
               / (NM * NT * N1T * N2T)
        D['PQGB'] = (2.* S1 * S1 - 2.* S2 + 5.) * (NS + N + 2.)\
               / (N * N1 * N2)   -   4.* S1 / NS\
             + (11.* NFO + 26.* NT + 15.* NS + 8.* N + 4.)\
               / (NT * N1T * N2)
        D['PGQA'] = (- S1 * S1 + 5.* S1 - S2) * (NS + N + 2.) \
               / (NM * N * N1)  -  2.* S1 / N1S\
             - (12.* NSI + 30.* NFI + 43.* NFO + 28.* NT - NS\
               - 12.* N - 4.) / (2.* NM * NT * N1T) 
        D['PGQB'] = (S1*S1 + S2 - SSTR2P) * (NS + N + 2.) / (NM * N * N1)\
             - S1 * (17.* NFO + 41.* NS - 22.* N - 12.) \
               / (3.* NMS * NS * N1)\
             + (109.* NN + 621.* NE + 1400.* NSE + 1678.* NSI\
               + 695.* NFI - 1031.* NFO - 1304.* NT - 152.* NS\
               + 432.* N + 144.) / (9.* NMS * NT * N1T * N2S)
        D['PGQC'] = (S1 - 8./3.) * (NS + N + 2.) / (NM * N * N1)  +  1./ N1S
        D['PGQC'] = 4./3.* D['PGQC']

        D['PGGA'] = - (2.* NFI + 5.* NFO + 8.* NT + 7.* NS - 2.* N - 2.)\
               * 8.* S1 / (NMS * NS * N1S * N2S) -  67./9.* S1 + 8./3.\
             - 4.* SSTR2P * (NS + N + 1.) / (NM * N * N1 * N2)\
             + 2.* S1 * SSTR2P - 4.* SSCHLP + 0.5 * SSTR3P\
             + (457.* NN + 2742.* NE + 6040.* NSE + 6098.* NSI\
               + 1567.* NFI - 2344.* NFO - 1632.* NT + 560.* NS\
               + 1488.* N + 576.) / (18.* NMS * NT * N1T * N2T)
        D['PGGB'] = (38.* NFO + 76.* NT + 94.* NS + 56.* N + 12.) * (-2.)\
               / (9.* NM * NS * N1S * N2)  +  20./9.* S1  -  4./3.
        D['PGGC'] = (2.* NSI + 4.* NFI + NFO - 10.* NT - 5.* NS - 4.* N\
               - 4.) * (-2.) / (NM * NT * N1T * N2)  -  1.


        D['PPQQA0'] = 3. - 4.* S1 + 2./(N * N1)
        D['PPQGA0'] = 4.* NM / (N * N1)
        D['PPGQA0'] = 2.* N2 / (N * N1)
        D['PPGGA0'] = 11./3. - 4.* S1 + 8./ (N * N1)
        D['PPGGB0'] = - 4./3.

        D['PPPSA'] = - 2.* N2 * (1.+ 2.* N + NT) / (NT* N1T) 
        D['PPQGA'] = ( (S1 * S1 - S2 + SSTR2M) * NM / (N * N1) \
             - 4.* S1 / (N * N1S)  -  (- 2. - 7.* N + 3.* NS - 4.* NT \
                + NFO + NFI) / (NT * N1T) ) * (-2.0)
        D['PPQGB'] = ( (- S1*S1 + S2 + 2.* S1 / N) * NM / (N * N1)  \
             -  NM * (1. + 3.5 * N + 4.* NS + 5.* NT + 2.5 * NFO) \
               / (NT * N1T)  +  4.* NM / (NS * N1S) ) * (-2.0)
        D['PPGQA'] = ( 2.* (S1*S1 + S2) * N2 / (N * N1)  -  2.* S1 * N2 \
               * (1.+ 3.* N) / (N * N1S)  -  N2 * (2.+ 15.* N \
               + 8.* NS - 12.* NT - 9.* NFO) / (NT * N1T) \
             + 8.* N2 / (NS * N1S) ) * (-0.5)
        D['PPGQB'] = ( (- S1*S1 - S2 + SSTR2M)* N2 / (N * N1) \
             +  S1 * (12.+ 22.* N + 11.* NS) / (3.* NS * N1)  \
             -  (36.+ 72.* N + 41.* NS + 254.* NT + 271.* NFO \
               + 76.* NFI) / (9.* NT* N1T) ) * (-1.0)
        D['PPGQC'] = (- S1 * N2 / (3.* N * N1) 
             +  N2 * (2.+ 5.* N) / (9.* N * N1S) ) * (-4.0)
        D['PPGGA'] = ( - 4.* S1 * SSTR2M - SSTR3M + 8.* SSCHLM\
             +  8.* SSTR2M / (N * N1)  +  2.* S1 * (72.+ 144.* N\
               + 67.* NS + 134.* NT + 67.* NFO) / (9.* NS* N1S)\
             - (144.+ 258.* N + 7.* NS + 698.* NT + 469.* NFO\
               + 144.* NFI + 48.* NSI) / (9.* NT * N1T) ) * (-0.5)
        D['PPGGB'] = ( - 5.* S1 / 9. + (- 3.+ 13.* N + 16.* NS + 6.* NT \
               + 3.* NFO) / (9.* NS* N1S) ) * (-4.0)
        D['PPGGC'] = ( 4.+ 2.* N - 8.* NS + NT + 5.* NFO + 3.* NFI + NSI) \
               / (NT * N1T) * (-1.0)


        # Gluck, Reya, and Vogt Timelike Splitting Functions

        D['PTNPA'] = (-4.*S1 + 3.+ 2./(N*N1))*(2.*S2 - np.pi**2/3. - (2.*N+1.)/(NS*N1S))  

        D['PTPSA']= D['PTNPA']

        D['PTPSB']= -80./(9.*NM) + 8./NT + 12./NS - 12./N + 8./N1T + 28./N1S - 4./N1\
                  + 32./(3.*N2S) + 224./(9.*N2)

        D['PTQGA']= (8./3.)*(S11*(NS+N+2.)/(N*N1*N2) + 1./NS - 5./(3.*N) - 1./(N*N1)\
                  -2./N1S + 4./(3.*N1) + 4./N2S - 4./(3.*N2))

        D['PTQGB']= (-2.*S11**2 + 2.*S11 + 10.*S21)*(NS+N+2.)/(N*N1*N2)\
                  + 4.*S11*( -1./NS + 1./N + 1./(N*N1) + 2./N1S - 4./N2S)\
                  - 2./NT + 5./NS - 12./N + 4./(NS*N1)\
                  - 12./(N*N1S) - 6./(N*N1) + 4./N1T - 4./N1S + 23./N1 - 20./N2

        D['PTQGC']= (2.*S11**2 - 10./3.*S11 - 6.*S21 + 2.*G11 - np.pi**2)*(NS+N+2.)/(N*N1*N2)\
                  - 4.*S11*( -2./NS + 1./N + 1./(N*N1) + 4./N1S - 6./N2S)\
                  - 40./(9.*NM) + 4./NT + 8./(3.*NS) + 26./(9.*N) - 8./(NS*N1S) + 22./(3.*N*N1)\
                  + 16./N1T + 68./(3.*N1S) - 190./(9.*N1) + 8./(N1S*N2) - 4./N2S + 356./(9.*N2)

        D['PTGQA']= (S1**2 - 3.*S2 - 2.*np.pi**2/3.)*(NS+N+2.)/(NM*N*N1)\
                  + 2.*S1*( 4./NMS - 2./(NM*N) - 4./NS + 3./N1S - 1./N1) - 8./(NMS*N)\
                  + 8./(NM*NS) + 2./NT + 8./NS - 1./(2.*N) + 1./N1T\
                  - 5./(2.*N1S) + 9./(2.*N1)

        D['PTGQB']= (-(S1**2) + 5.*S2 - G1 + np.pi**2/6.)*(NS+N+2.)/(NM*N*N1)\
                  + 2.*S1*( -2./NMS + 2./(NM*N) + 2./NS - 2./N1S + 1./N1) - 8./NMT\
                  + 6./NMS + 17./(9.*NM) + 4./(NMS*N) - 12./(NM*NS) - 8./NS + 5./N\
                  - 2./(NS*N1) - 2./N1T - 7./N1S - 1./N1 - 8./(3.*N2S) - 44./(9.*N2)

        D['PTGGA']= -16./(3.*NMS) + 80./(9.*NM) + 8./NT - 16./NS + 12./N + 8./N1T\
                  -24./N1S + 4./N1 - 16./(3.*N2S) - 224./(9.*N2)

        D['PTGGB']= (-8./3.)*(S2 - 1./NMS + 1./NS - 1./N1S + 1./N2S - np.pi**2/6.)

        D['PTGGC']= -8.*S1*S2 + 8.*S1*( 1./NMS - 1./NS + 1./N1S - 1./N2S + np.pi**2/6.)\
                  +(8.*S2 - 4.*np.pi**2/3.)*( 1./NM - 1./N + 1./N1 - 1./N2 + 11./12.)\
                  - 8./NMT + 22./(3.*NMS) - 8./(NMS*N) - 8./(NM*NS) - 8./NT\
                  - 14./(3.*NS) - 8./N1T + 14./(3.*N1S) - 8./(N1S*N2) - 8./(N1*N2S)\
                  - 8./N2T - 22./(3.*N2S)

    def LO_unpolarized_splitting_functions(self):

        D=self.D
        D['P0QQ']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['P0QG']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['P0GQ']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['P0GG']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)  
        D['P0'] = np.zeros((D['nflav'],2,2,D['Nsize']),dtype=complex)

        N=D['N']
        for Nf in range(3,D['nflav']):

            D['P0QQ'][Nf]=4.0/3.0*(3.0+2.0/N/(N+1)-4.0*D['S1'])
            D['P0QG'][Nf]=2.0*(N**2+N+2)/(N*(N+1)*(N+2))*Nf
            D['P0GQ'][Nf]=8.0/3.0*(N**2+N+2)/(N-1)/N/(N+1)
            D['P0GG'][Nf]=3.0*(11.0/3.0+4.0/N/(N-1)+4.0/(N+1)/(N+2)-4.0*D['S1'])-2.0/3.0*Nf

            D['P0'][Nf,0,0] = D['P0QQ'][Nf] 
            D['P0'][Nf,0,1] = D['P0QG'][Nf]
            D['P0'][Nf,1,0] = D['P0GQ'][Nf]
            D['P0'][Nf,1,1] = D['P0GG'][Nf]
 
    def NLO_unpolarized_splitting_functions(self):
        D=self.D

        CF=D['CF']
        CA=D['CA']
        TR=D['TR']

        D['P1NSP']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['P1NSM']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['P1']=np.zeros((D['nflav'],2,2,D['Nsize']),dtype=complex)

        for NF in range(3,D['nflav']):

            D['P1NSP'][NF]=  CF*((CF-CA/2.)*D['PNPA'] + CA* D['PNSB'] + TR*NF*D['PNSC'])
            D['P1NSM'][NF]=  CF*((CF-CA/2.)*D['PNMA'] + CA* D['PNSB'] + TR*NF*D['PNSC'])

            D['P1'][NF,0,0]=D['P1NSP'][NF]+TR*NF*CF*D['PPSA']*4.
            D['P1'][NF,0,1]=TR*NF*(CA*D['PQGA']+CF*D['PQGB'])*4.
            D['P1'][NF,1,0]=(CF*CF*D['PGQA']+CF*CA*D['PGQB']+TR*NF*CF*D['PGQC'])*4.
            D['P1'][NF,1,1]=(CA*CA*D['PGGA']+TR*NF*(CA*D['PGGB']+CF*D['PGGC']))*4.

    def LO_polarized_splitting_functions(self):
        D=self.D

        CF=D['CF']
        CA=D['CA']
        TR=D['TR']

        D['PP0'] = np.zeros((D['nflav'],2,2,D['Nsize']),dtype=complex)

        N=D['N']
        for NF in range(3,D['nflav']):
            D['PP0'][NF,0,0]=CF*D['PPQQA0']
            D['PP0'][NF,0,1]=TR*NF*D['PPQGA0']
            D['PP0'][NF,1,0]=CF*D['PPGQA0']
            D['PP0'][NF,1,1]=CA*D['PPGGA0']+TR*NF*D['PPGGB0']

    def NLO_polarized_splitting_functions(self):
        D=self.D

        CF=D['CF']
        CA=D['CA']
        TR=D['TR']

        D['PP1'] = np.zeros((D['nflav'],2,2,D['Nsize']),dtype=complex)
        N=D['N']
        for NF in range(3,D['nflav']):
            D['PP1'][NF,0,0]=D['P1NSM'][NF]+TR*NF*CF*D['PPPSA']*4.
            D['PP1'][NF,0,1]=TR*NF*(CA*D['PPQGA']+CF*D['PPQGB'])*4.
            D['PP1'][NF,1,0]=(CF*CF*D['PPGQA']+CF*CA*D['PPGQB']+TR*NF*CF*D['PPGQC'])*4.
            D['PP1'][NF,1,1]=(CA*CA*D['PPGGA']+TR*NF*(CA*D['PPGGB']+CF*D['PPGGC']))*4.
        
    def LO_unpol_timelike_splitting_functions(self):
    
        D=self.D
        D['PT0QQ']=np.zeros((D['nflav'],D['Nsize']),dtype=complex) 
        D['PT0GQ']=np.zeros((D['nflav'],D['Nsize']),dtype=complex) 
        D['PT0QG']=np.zeros((D['nflav'],D['Nsize']),dtype=complex) 
        D['PT0GG']=np.zeros((D['nflav'],D['Nsize']),dtype=complex) 
        D['PT0'] = np.zeros((D['nflav'],2,2,D['Nsize']),dtype=complex)

        for Nf in range(3,D['nflav']):

            D['PT0QQ'][Nf]=D['P0QQ'][Nf]
            D['PT0QG'][Nf]=D['P0QG'][Nf]/2./Nf
            D['PT0GQ'][Nf]=D['P0GQ'][Nf]*2.*Nf
            D['PT0GG'][Nf]=D['P0GG'][Nf]

            D['PT0'][Nf,0,0] = D['PT0QQ'][Nf]
            D['PT0'][Nf,0,1] = D['PT0GQ'][Nf]
            D['PT0'][Nf,1,0] = D['PT0QG'][Nf]
            D['PT0'][Nf,1,1] = D['PT0GG'][Nf]
        
    def NLO_unpol_timelike_splitting_functions(self):
    
        D=self.D

        CF=D['CF']
        CA=D['CA']
        TR=D['TR']

        D['PT1NSP2']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['PT1NSP']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['PT1NSM2']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['PT1NSM']=np.zeros((D['nflav'],D['Nsize']),dtype=complex)
        D['PT1']=np.zeros((D['nflav'],2,2,D['Nsize']),dtype=complex)
        D['PT2']=np.zeros((D['nflav'],2,2,D['Nsize']),dtype=complex)
        D['PT02']=np.zeros((D['nflav'],2,2,D['Nsize']),dtype=complex)

        for Nf in range(3,D['nflav']):

            D['PT1NSP'][Nf] = D['P1NSP'][Nf] + CF**2*D['PTNPA']*4.0  
            D['PT1NSM'][Nf] = D['P1NSM'][Nf] + CF**2*D['PTNPA']*4.0

            D['PT1'][Nf,0,0] = D['P1'][Nf,0,0]+4*(CF**2*D['PTPSA']+CF*TR*Nf*D['PTPSB'])
            D['PT1'][Nf,1,0] = 4*(((TR*Nf)**2*D['PTQGA']+CF*TR*Nf*D['PTQGB']+CA*TR*Nf*D['PTQGC'])/(2.0*Nf))
            D['PT1'][Nf,0,1] = 4*((CF**2*D['PTGQA']+CA*CF*D['PTGQB'])*(2.0*Nf))
            D['PT1'][Nf,1,1] = D['P1'][Nf,1,1]+4*(CF*TR*Nf*D['PTGGA']+CA*TR*Nf*D['PTGGB']+CA**2*D['PTGGC'])

    def load_unpolarized_spl(self):
        D=self.D
        Nsize=D['N'].size
        nflav=D['nflav']
        norder=D['norder']

        # initialize flav composed splitting functions arrays
        self.PNSP=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.PNSM=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.PNSV=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.P   =np.zeros((nflav,norder,2,2,Nsize),dtype=complex)

        for Nf in range(3,nflav):

            # LO unpolarized
            self.PNSP[Nf,0] = D['P0QQ'][Nf] 
            self.PNSM[Nf,0] = D['P0QQ'][Nf] 
            self.PNSV[Nf,0] = D['P0QQ'][Nf] 
            self.P[Nf,0]    = D['P0'][Nf]

            # NLO unpolarized
            self.PNSP[Nf,1] = D['P1NSP'][Nf] 
            self.PNSM[Nf,1] = D['P1NSM'][Nf]
            self.PNSV[Nf,1] = D['P1NSM'][Nf]
            self.P[Nf,1]    = D['P1'][Nf]

    def load_polarized_spl(self):
        D=self.D
        Nsize=D['N'].size
        nflav=D['nflav']
        norder=D['norder']

        # initialize flav composed splitting functions arrays
        self.PNSP=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.PNSM=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.PNSV=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.P    =np.zeros((nflav,norder,2,2,Nsize),dtype=complex)

        for Nf in range(3,nflav):

            # LO polarized
            self.PNSP[Nf,0] = D['P0QQ'][Nf]
            self.PNSM[Nf,0] = D['P0QQ'][Nf]
            self.PNSV[Nf,0] = D['P0QQ'][Nf]
            self.P[Nf,0]    = D['PP0'][Nf]

            # NLO polarized
            self.PNSP[Nf,1] = D['P1NSM'][Nf]
            self.PNSM[Nf,1] = D['P1NSP'][Nf]
            self.PNSV[Nf,1] = D['P1NSP'][Nf]
            self.P[Nf,1]    = D['PP1'][Nf]
        
    def load_unp_FF_spl(self):
    
        D=self.D
        Nsize=D['N'].size
        nflav=D['nflav']
        norder=D['norder']

        # initialize flav composed splitting functions arrays
        self.PNSP=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.PNSM=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.PNSV=np.zeros((nflav,norder,Nsize),dtype=complex)
        self.P   =np.zeros((nflav,norder,2,2,Nsize),dtype=complex)

        for Nf in range(3,nflav):

            # LO unpolarized
            self.PNSP[Nf,0] = D['PT0QQ'][Nf] 
            self.PNSM[Nf,0] = D['PT0QQ'][Nf] 
            self.PNSV[Nf,0] = D['PT0QQ'][Nf] 
            self.P[Nf,0]    = D['PT0'][Nf]

            # NLO unpolarized
            self.PNSP[Nf,1] = D['PT1NSP'][Nf] 
            self.PNSM[Nf,1] = D['PT1NSM'][Nf]
            self.PNSV[Nf,1] = D['PT1NSM'][Nf]
            self.P[Nf,1]    = D['PT1'][Nf]


