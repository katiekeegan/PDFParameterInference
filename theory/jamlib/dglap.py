import sys
import numpy as np


class DGLAP:
    """
    DGLAP is the evolution of parton distribution functions (1d QCFs).
    DGLAP stands for the authors of the calculation Dokshitzer–Gribov–Lipatov–Altarelli–Parisi.
    """
 
    def __init__(self,mell,asevo,spl,mode='truncated',order=1):
        """
        mell  = class mellin
        asevo = class alphaS
        spl   = class splitting function 
        mode  = flag truncated,iterated
        order = flag LO,NLO,NNLO
        itermax = maximun terms in the iterated singlet evolution
        
        Nomenclature:
        BC = bondary condition
        EO = evolution operator
        BM = beta matrix
        """
        self.mell=mell
        self.asevo=asevo
        self.spl=spl
        self.mode=mode
        self.itermax=16 
        self.order=order 
        self.get_beta_matrix()
        self.get_evolution_operators()

    def get_beta_matrix(self):
        """
        beta matrix stores the beta coeffs for 
        different Nf and perturbative orders
        """
        beta=self.asevo.beta
        nflav =6+1 # up to 6 flavors
        norder=2+1 # up to NNLO
        b0=np.zeros(nflav)  
        b=np.zeros((nflav,norder))
        for Nf in range(3,nflav):
            b0[Nf]=beta[Nf][0]
            for order in range(norder): 
                b[Nf,order]=beta[Nf,order]/beta[Nf,0]
        # globalize
        self.b=b
        self.nflav=nflav
        self.norder=norder

    def get_non_singlet_evolution_operator(self,P):
        """
        Needed operators for non_singlet evolution.
        """

        b=self.b
        nflav=self.nflav
        norder=self.norder
        Nsize=self.mell.N.size
        beta=self.asevo.beta

        # construct evolution operators
        R=np.zeros((nflav,norder,Nsize),dtype=complex)
        U=np.zeros((nflav,norder,Nsize),dtype=complex)
        RT=np.zeros((nflav,norder,Nsize),dtype=complex)

        # LO operators
        for Nf in range(3,nflav):
            R[Nf,0]=P[Nf,0]/beta[Nf,0]
            U[Nf,0]=np.ones(Nsize)

        # NLO operators
        for Nf in range(3,nflav):
            R[Nf,1]=P[Nf,1]/beta[Nf,0]-b[Nf,1]*R[Nf,0]
            RT[Nf,1]=R[Nf,1]*U[Nf,0]
            U[Nf,1]=-RT[Nf,1]/1.0

        # NNLO operators
        for Nf in range(3,nflav):
            R[Nf,2]=P[Nf,2]/beta[Nf,0]-b[Nf,1]*R[Nf,1]-b[Nf,2]*R[Nf,0]
            RT[Nf,2]=R[Nf,1]*U[Nf,1]+R[Nf,2]*U[Nf,0]
            U[Nf,2]=-RT[Nf,2]/2.0

        return {'U':U,'R':R}

    def get_singlet_evolution_operator(self,P):
        """
        Needed operators for singlet evolution.
        """
        
        b=self.b
        nflav=self.nflav
        norder=self.norder
        Nsize=self.mell.N.size
        beta=self.asevo.beta

        # construct evolution operators
        R=np.zeros((nflav,norder,2,2,Nsize),dtype=complex)
        U=np.zeros((nflav,norder,2,2,Nsize),dtype=complex)
        RT=np.zeros((nflav,norder,2,2,Nsize),dtype=complex)
        RP=np.zeros((nflav,2,2,Nsize),dtype=complex)
        RM=np.zeros((nflav,2,2,Nsize),dtype=complex)
        rp=np.zeros((nflav,Nsize),dtype=complex)
        rm=np.zeros((nflav,Nsize),dtype=complex)

        # LO operators
        for Nf in range(3,nflav):
            R[Nf,0]=P[Nf,0]/beta[Nf,0]
            r1=R[Nf,0,0,0]+R[Nf,0,1,1]
            r2=((R[Nf,0,0,0]-R[Nf,0,1,1])**2+4*R[Nf,0,0,1]*R[Nf,0,1,0])**0.5
            rp[Nf]=0.5*(r1+r2)
            rm[Nf]=0.5*(r1-r2)

            RP[Nf,0,0]=(R[Nf,0,0,0]-rm[Nf])/(rp[Nf]-rm[Nf])
            RP[Nf,1,1]=(R[Nf,0,1,1]-rm[Nf])/(rp[Nf]-rm[Nf])
            RP[Nf,0,1]=R[Nf,0,0,1]/(rp[Nf]-rm[Nf])
            RP[Nf,1,0]=R[Nf,0,1,0]/(rp[Nf]-rm[Nf])

            RM[Nf,0,0]=1-RP[Nf,0,0]
            RM[Nf,1,1]=1-RP[Nf,1,1]
            RM[Nf,0,1]= -RP[Nf,0,1]
            RM[Nf,1,0]= -RP[Nf,1,0]

            U[Nf,0,0,0]=np.zeros(Nsize)+1
            U[Nf,0,0,1]=np.zeros(Nsize)
            U[Nf,0,1,0]=np.zeros(Nsize)
            U[Nf,0,1,1]=np.zeros(Nsize)+1

        # NLO operators
        for Nf in range(3,nflav):
            R[Nf,1]=P[Nf,1]/beta[Nf,0]-b[Nf,1]*R[Nf,0]
            RT[Nf,1]=R[Nf,1]
            U11=-np.einsum('ij...,jk...,kl...->il...',RP[Nf],RT[Nf,1],RP[Nf])
            U12=-np.einsum('ij...,jk...,kl...->il...',RM[Nf],RT[Nf,1],RM[Nf])
            U13= np.einsum('ij...,jk...,kl...->il...',RP[Nf],RT[Nf,1],RM[Nf])/(rm[Nf]-rp[Nf]-1)
            U14= np.einsum('ij...,jk...,kl...->il...',RM[Nf],RT[Nf,1],RP[Nf])/(rp[Nf]-rm[Nf]-1)
            U[Nf,1]=U11+U12+U13+U14

        # NNLO operators
        for Nf in range(3,nflav):
            R[Nf,2]=P[Nf,2]/beta[Nf,0]-b[Nf,1]*R[Nf,1]-b[Nf,2]*R[Nf,0]
            RT[Nf,2,] =np.einsum('ij...,jk...->ik...',R[Nf,1],U[Nf,1])
            RT[Nf,2,]+=np.einsum('ij...,jk...->ik...',R[Nf,2],U[Nf,0])
            U21=-np.einsum('ij...,jk...,kl...->il...',RP[Nf],RT[Nf,2],RP[Nf])/2.0
            U22=-np.einsum('ij...,jk...,kl...->il...',RM[Nf],RT[Nf,2],RM[Nf])/2.0
            U23= np.einsum('ij...,jk...,kl...->il...',RP[Nf],RT[Nf,2],RM[Nf])/(rm[Nf]-rp[Nf]-2)
            U24= np.einsum('ij...,jk...,kl...->il...',RM[Nf],RT[Nf,2],RP[Nf])/(rp[Nf]-rm[Nf]-2)
            U[Nf,2]=U21+U22+U23+U24

        # for iterated solution
        RH  = np.zeros((nflav,self.itermax,2,2,Nsize),dtype=complex)
        RT  = np.zeros((nflav,self.itermax,2,2,Nsize),dtype=complex)
        U1H = np.zeros((nflav,self.itermax,2,2,Nsize),dtype=complex)

        for Nf in range(3,nflav):

            RH[Nf,1,:,:,:]  = np.copy(R[Nf,1])
            U1H[Nf,1,:,:,:] = np.copy(U[Nf,1,:,:,:])

            for HO in range(2,self.itermax):
                RH[Nf,HO,:,:,:] = - b[Nf,1] * RH[Nf,HO-1,:,:,:]
                RT[Nf,HO,:,:,:] =  np.copy(RH[Nf,HO,:,:,:])
                for j in range(1,HO):
                    for k in range(Nsize):
                        RT[Nf,HO,:,:,k]+= np.dot(RH[Nf,j,:,:,k],U1H[Nf,HO-j,:,:,k])

                U11=-np.einsum('ij...,jk...,kl...->il...',RP[Nf],RT[Nf,HO],RP[Nf])/HO
                U12=-np.einsum('ij...,jk...,kl...->il...',RM[Nf],RT[Nf,HO],RM[Nf])/HO
                U13= np.einsum('ij...,jk...,kl...->il...',RP[Nf],RT[Nf,HO],RM[Nf])/(rm[Nf]-rp[Nf]-HO)
                U14= np.einsum('ij...,jk...,kl...->il...',RM[Nf],RT[Nf,HO],RP[Nf])/(rp[Nf]-rm[Nf]-HO)
                U1H[Nf,HO]=U11+U12+U13+U14
    
        return {'U':U,'rp':rp,'rm':rm,'RP':RP,'RM':RM,'U1H':U1H}

    def get_evolution_operators(self):
        """
        Gather all evolution operators.
        """
        spl=self.spl
        self.EO_NSP = self.get_non_singlet_evolution_operator(spl.PNSP)
        self.EO_NSM = self.get_non_singlet_evolution_operator(spl.PNSM)
        self.EO_NSV = self.get_non_singlet_evolution_operator(spl.PNSV)
        self.EO_S   = self.get_singlet_evolution_operator(spl.P)

    def evolve_nonsinglet(self,EO,qini,Q2ini,Q2fin,Nf):

        # evolve
        a = self.asevo.get_a(Q2fin)
        a0= self.asevo.get_a(Q2ini)
        L=np.power(a/a0,-EO['R'][Nf,0])

        if self.mode=='truncated': # IMODE=4 of pegasus

            operator=np.copy(L)

            if  self.order>=1: 
                operator+= (a-a0)*EO['U'][Nf,1]*L

            if  self.order>=2: 
                operator+= a**2*EO['U'][Nf,2]*L-a*a0*EO['U'][Nf,1]*L*EO['U'][Nf,1]+a0**2*L*(EO['U'][Nf,1]*EO['U'][Nf,1]-EO['U'][Nf,2])  

        elif self.mode=='iterated': # IMODE=1 of pegasus

            b1=self.b[Nf,1]
            operator=np.copy(L)

            if  self.order==1:
                operator*=np.exp(np.log((1+b1*a)/(1+b1*a0))/b1*EO['U'][Nf,1])

            if  self.order==2:
                print('ERR(DGLAP): iterated non-singlet evolution not implemented for NNLO')
                sys.exit()

        q=operator*qini
        return q

    def evolve_singlet(self,EO,qini,Q2ini,Q2fin,Nf):

        Nsize=self.mell.N.size

        # evolve
        a = self.asevo.get_a(Q2fin)
        a0= self.asevo.get_a(Q2ini)
        L=np.power(a/a0,-EO['rp'][Nf])*EO['RP'][Nf]+np.power(a/a0,-EO['rm'][Nf])*EO['RM'][Nf]

        if self.mode=='truncated': # IMODE=4 of pegasus

            operator = np.copy(L)

            if  self.order>=1: 
                operator+= a*np.einsum('ij...,jk...->ik...',EO['U'][Nf,1],L,dtype=complex)\
                          -a0*np.einsum('ij...,jk...->ik...',L,EO['U'][Nf,1],dtype=complex)

            if  self.order>=2: 
                operator+=\
                    a**2  * np.einsum('ij...,jk...->ik...',EO['U'][Nf,2],L,dtype=complex)\
                   -a*a0  * np.einsum('ij...,jl...,lk...->ik...',EO['U'][Nf,1],L,EO['U'][Nf,1],dtype=complex)\
                   +a0**2 * np.einsum('ij...,jl...,lk...->ik...',L,EO['U'][Nf,1],EO['U'][Nf,1],dtype=complex)\
                   -a0**2 * np.einsum('ij...,jk...->ik...',L,EO['U'][Nf,2],dtype=complex)

        elif self.mode=='iterated': # IMODE=1 of pegasus

            U=EO['U']
            U1H=EO['U1H']

            # The O(a_s) U-matrices at the initial and final scale"

            UF=np.zeros((2,2,Nsize),dtype=complex)
            UF[0,0,:]=1
            UF[1,1,:]=1

            UI=np.zeros((2,2,Nsize),dtype=complex)
            UI[0,0,:]=1
            UI[1,1,:]=1

            if  self.order>=1:
                UF[:,:,:]+=a*U[Nf,1,:,:,:]
                UI[:,:,:]+=a0*U[Nf,1,:,:,:]

                #  Addition of the higher-order contributions to UF and UI for the
                #  two iterative solutions"

                aHO=a
                a0HO=a0
                for HO in range(2,self.itermax):
                    aHO*=a
                    a0HO*=a0
                    UF[:,:]+= aHO  * U1H[Nf,HO,:,:]
                    UI[:,:]+= a0HO * U1H[Nf,HO,:,:]

            # The full inverse UM of UI
            DETINV = 1./ (UI[0,0]*UI[1,1] - UI[0,1]*UI[1,0]) 
            UM=np.zeros((2,2,Nsize),dtype=complex)
            UM[0,0] =  UI[1,1] * DETINV
            UM[0,1] = -UI[0,1] * DETINV
            UM[1,0] = -UI[1,0] * DETINV
            UM[1,1] =  UI[0,0] * DETINV

            # The NLO evolution operators for IMODE = 1 or 2

            operator=np.einsum('ij...,jk...,kl...->il...',UF,L,UM,dtype=complex)

        q=np.einsum('ij...,j...->i...',operator,qini,dtype=complex)
        return q

    def evolve(self,BC,Q2ini,Q2fin,Nf):
        """
        This function performs the evolution of 1d QCFs.
        Importantly, this must be called for the various Nf thresholds.
        Different flavor combinations have different evolution procedures (singlet, nonsinglet).
        These are calculated in Mellin space (conjugate to x).
        inputs:
            BC = boundary condition
            Q2ini = initial Q2 value
            Q2fin = final Q2 value to evolve up to.
            Nf = number of active flavors.
        return:
            dictionary of resulting moments of evolved QCFs.
        """

        nflav=self.nflav
        Nsize=self.mell.N.size

        EO_NSP =  self.EO_NSP 
        EO_NSM =  self.EO_NSM 
        EO_NSV =  self.EO_NSV 
        EO_S   =  self.EO_S   

        # retrieve boundary condition
        vm,vp={},{}
        for k in [i**2-1 for i in range(1,nflav)]:
            vm[k]=np.copy(BC['vm'][k]) 
            vp[k]=np.copy(BC['vp'][k]) 
        qv=np.copy(BC['qv']) 
        q=np.copy(BC['q']) 

        # evolve
        qv=self.evolve_nonsinglet(EO_NSV,qv,Q2ini,Q2fin,Nf)
        q=self.evolve_singlet(EO_S,q,Q2ini,Q2fin,Nf)
        qs = q[0]

        if 'um_' in BC: um_=self.evolve_nonsinglet(EO_NSV,BC['um_'],Q2ini,Q2fin,Nf)
        if 'dm_' in BC: dm_=self.evolve_nonsinglet(EO_NSV,BC['dm_'],Q2ini,Q2fin,Nf)

        if   Nf==3:
            vm[3] =self.evolve_nonsinglet(EO_NSM,vm[3],Q2ini,Q2fin,Nf)
            vm[8] =self.evolve_nonsinglet(EO_NSM,vm[8],Q2ini,Q2fin,Nf)
            vm[15]=qv
            vm[24]=qv
            vm[35]=qv
            vp[3] =self.evolve_nonsinglet(EO_NSP,vp[3],Q2ini,Q2fin,Nf)
            vp[8] =self.evolve_nonsinglet(EO_NSP,vp[8],Q2ini,Q2fin,Nf)
            vp[15]=qs
            vp[24]=qs
            vp[35]=qs

        elif Nf==4:
            vm[3] =self.evolve_nonsinglet(EO_NSM,vm[3],Q2ini,Q2fin,Nf)
            vm[8] =self.evolve_nonsinglet(EO_NSM,vm[8],Q2ini,Q2fin,Nf)
            vm[15]=self.evolve_nonsinglet(EO_NSM,vm[15],Q2ini,Q2fin,Nf)
            vm[24]=qv
            vm[35]=qv
            vp[3] =self.evolve_nonsinglet(EO_NSP,vp[3],Q2ini,Q2fin,Nf)
            vp[8] =self.evolve_nonsinglet(EO_NSP,vp[8],Q2ini,Q2fin,Nf)
            vp[15]=self.evolve_nonsinglet(EO_NSP,vp[15],Q2ini,Q2fin,Nf)
            vp[24]=qs
            vp[35]=qs

        elif Nf==5:
            vm[3] =self.evolve_nonsinglet(EO_NSM,vm[3],Q2ini,Q2fin,Nf)
            vm[8] =self.evolve_nonsinglet(EO_NSM,vm[8],Q2ini,Q2fin,Nf)
            vm[15]=self.evolve_nonsinglet(EO_NSM,vm[15],Q2ini,Q2fin,Nf)
            vm[24]=self.evolve_nonsinglet(EO_NSM,vm[24],Q2ini,Q2fin,Nf)
            vm[35]=qv
            vp[3] =self.evolve_nonsinglet(EO_NSP,vp[3],Q2ini,Q2fin,Nf)
            vp[8] =self.evolve_nonsinglet(EO_NSP,vp[8],Q2ini,Q2fin,Nf)
            vp[15]=self.evolve_nonsinglet(EO_NSP,vp[15],Q2ini,Q2fin,Nf)
            vp[24]=self.evolve_nonsinglet(EO_NSP,vp[24],Q2ini,Q2fin,Nf)
            vp[35]=qs

        elif Nf==6:
            vm[3] =self.evolve_nonsinglet(EO_NSM,vm[3],Q2ini,Q2fin,Nf)
            vm[8] =self.evolve_nonsinglet(EO_NSM,vm[8],Q2ini,Q2fin,Nf)
            vm[15]=self.evolve_nonsinglet(EO_NSM,vm[15],Q2ini,Q2fin,Nf)
            vm[24]=self.evolve_nonsinglet(EO_NSM,vm[24],Q2ini,Q2fin,Nf)
            vm[35]=self.evolve_nonsinglet(EO_NSM,vm[35],Q2ini,Q2fin,Nf)
            vp[3] =self.evolve_nonsinglet(EO_NSP,vp[3],Q2ini,Q2fin,Nf)
            vp[8] =self.evolve_nonsinglet(EO_NSP,vp[8],Q2ini,Q2fin,Nf)
            vp[15]=self.evolve_nonsinglet(EO_NSP,vp[15],Q2ini,Q2fin,Nf)
            vp[24]=self.evolve_nonsinglet(EO_NSP,vp[24],Q2ini,Q2fin,Nf)
            vp[35]=self.evolve_nonsinglet(EO_NSP,vp[35],Q2ini,Q2fin,Nf)

        # flav decomposition
        qp =np.zeros((nflav,Nsize),dtype=complex)
        qm =np.zeros((nflav,Nsize),dtype=complex)

        qm[6]=(qv-vm[35])/6.
        qm[5]=qm[6]+(vm[35]-vm[24])/5.
        qm[4]=qm[5]+(vm[24]-vm[15])/4.
        qm[3]=qm[4]+(vm[15]-vm[8])/3.
        qm[2]=qm[3]+(vm[8]-vm[3])/2.
        qm[1]=qm[3]+(vm[8]+vm[3])/2.

        qp[6]=(qs-vp[35])/6.
        qp[5]=qp[6]+(vp[35]-vp[24])/5.
        qp[4]=qp[5]+(vp[24]-vp[15])/4.
        qp[3]=qp[4]+(vp[15]-vp[8])/3.
        qp[2]=qp[3]+(vp[8]-vp[3])/2.
        qp[1]=qp[3]+(vp[8]+vp[3])/2.

        # for internal usage 
        result={}
        result['vm']=vm 
        result['vp']=vp 
        result['qv']=qv 
        result['q'] =q 


        # for external usage
        result['qs'] =q[0]

        result['vm3'] =vm[3]
        result['vm8'] =vm[8]
        result['vm15']=vm[15] 
        result['vm24']=vm[24] 
        result['vm35']=vm[35] 

        result['vp3'] =vp[3]
        result['vp8'] =vp[8]
        result['vp15']=vp[15] 
        result['vp24']=vp[24] 
        result['vp35']=vp[35] 

        result['uv']=qm[1]
        result['dv']=qm[2]

        if 'um_' in BC: result['um_']=um_
        if 'dm_' in BC: result['dm_']=dm_

        result['up']=qp[1]
        result['um']=qm[1]
        result['dp']=qp[2]
        result['dm']=qm[2]
        result['sp']=qp[3]
        result['sm']=qm[3]
        result['cp']=qp[4]
        result['cm']=qm[4]
        result['bp']=qp[5]
        result['bm']=qm[5]
        result['tp']=qp[6]
        result['tm']=qm[6]


        result['qp']=qp
        result['qm']=qm


        result['g'] =q[1]
        result['u']=0.5*(qp[1]+qm[1])
        result['d']=0.5*(qp[2]+qm[2])
        result['s']=0.5*(qp[3]+qm[3])
        result['c']=0.5*(qp[4]+qm[4])
        result['b']=0.5*(qp[5]+qm[5])
        result['ub']=0.5*(qp[1]-qm[1])
        result['db']=0.5*(qp[2]-qm[2])
        result['sb']=0.5*(qp[3]-qm[3])
        result['cb']=0.5*(qp[4]-qm[4])
        result['bb']=0.5*(qp[5]-qm[5])


        result['Vpi'] =result['ub']-result['u']
        result['Spi'] =result['u']

        return result





