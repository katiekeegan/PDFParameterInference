import numpy as np
import params as par 
import cfg
from alphaS  import ALPHAS
from dglap   import DGLAP
from kernels import KERNELS
from mellin  import MELLIN
from special import beta

class PDF:
    
    """
    This class computes the 1d QCF, called a parton distribution function (PDF) for all flavors as a function of (x,Q2).
    Inputs:
        class instance mellin = contains routine to compute Mellin moments and inversions
        class instance alphaS = contains routine to evaluate the strong coupling constant at any scale
    """

    def __init__(self,mellin,alphaS):
  
        self.spl='upol'
        self.Q20=cfg.Q20
        self.mc2=par.mc2
        self.mb2=par.mb2
  
        self.kernel=KERNELS(mellin,self.spl) #--contains splitting functions needed for evolution in Q2
        self.dglap=DGLAP(mellin,alphaS,self.kernel,'truncated',cfg.dglap_order) #--describes the evolution in Q2 of the 1d QCF
        self.mellin=mellin
 
        self.set_params()
        self.get_current_par_array()
        self.setup(self.current_par)
  
    def set_params(self):
        """
        f(x) = norm * x**a * (1-x)**b * (1+c*x**0.5+d*x)
        """
        self.params={}
  
        #--first shapes -- individual flavors
        self.params['g1']    = [0.34,-0.068,10,-4.5,5.77]
        self.params['uv1']   = [0.32, -0.71,3.48,1.34,23.3]
        self.params['dv1']   = [0.13, -0.78,4.87,-5.80,47.1]
        self.params['sea1']  = [0.006,-0.41,22.5,0,0]
        self.params['sea2']  = [0.006,-0.41,22.5,0,0]
        self.params['db1']   = [0.031,-0.69,10.0,81.1,-82.2]
        self.params['ub1']   = [0.026,0.29,10.1,-3.73,4.27]
        self.params['s1']    = [ 0.02117505,-0.7834729,5.94912366,0.,0.]
        self.params['sb1']   = [ 0.03185304,-0.4761831,10.,0.,0.]
        #--mix parameters
        self.params['mix']   = [0.0,0.0,0.0,0.0,0.0]
 
        #--second shapes
        self.params['uv2']   = [0.00, -0.71,3.48,1.34,23.3]
        self.params['dv2']   = [0.00, -0.78,4.87,-5.80,47.1]
 
        """
        These fixed parameters are not to be changed during the fit.
        They specify the index of the array in self.params.
        The reason they are fixed in some cases is for physical boundaries,
            e.g. the number of up valence quarks is fixed to be 2, thus fixing the normalization for that flavor.
        Others (such as mix, uv2, dv2, etc.) are fixed due to insensitivity in the data.
        """
        #--fixed params
        self.fixed={}
        self.fixed['g1']=[0]
        self.fixed['uv1']=[0]
        self.fixed['dv1']=[0]
        self.fixed['s1']=[0,3,4]
        self.fixed['sb1']=[3,4]
        self.fixed['mix']=[0,1,2,3,4,5]
        self.fixed['sea1']=[3,4]
        self.fixed['sea2']=[3,4]
        self.fixed['uv2']=[0,1,2,3,4]
        self.fixed['dv2']=[0,1,2,3,4]
        
        
        self.features=['g1','uv1','dv1','sea1','sea2','db1','ub1','s1','sb1','uv2','dv2']
        
        self._parmin=[-10,-2, 0,-10,-10]
        self._parmax=[+10,10,10,+10,+10]
        
    def get_current_par_array(self):
        
        self.current_par=[]
        self.order=[]
        self.parmin=[]
        self.parmax=[]
        for _ in self.features:
            for i in range(len(self.params[_])):
                if _ in self.fixed:
                    if i in self.fixed[_]: continue
                self.current_par.append(self.params[_][i])
                self.order.append(_)
                if   _=='uv1' and i==1: 
                    self.parmin.append(-1)      
                    self.parmax.append(self._parmax[i])
                elif _=='dv1' and i==1: 
                    self.parmin.append(-1)
                    self.parmax.append(self._parmax[i])
                else:
                    self.parmin.append(self._parmin[i])
                    self.parmax.append(self._parmax[i])
        
        self.current_par=np.array(self.current_par)
        self.parmin=np.array(self.parmin)
        self.parmax=np.array(self.parmax)
        
        return self.current_par

    def update_params(self,par_array):
        
        """
        This function is essential to pass a new array of parameters into the class during a fit.
        input:
            par_array: an array of the free parameters; len(par_array)=total number of free parameters across the flavors.
        """
        
        cnt=0
        for _ in self.features:
            for i in range(len(self.params[_])):
                if _ in self.fixed:
                    if i in self.fixed[_]: continue
                self.params[_][i]=par_array[cnt]
                cnt+=1
                
    def get_mix(self,N):
        #--mix distribution parameters (dv --> dv + N_dv*M*x**a*uv)
        M = self.params['mix'][0] * self.params['uv1'][0] * self.params['dv1'][0]
        a = self.params['mix'][1] + self.params['uv1'][1]
        b = self.params['uv1'][2]
        c = self.params['uv1'][3]
        d = self.params['uv1'][4]
  
        #--moment of mix distribution
        mom=beta(N+a,b+1)+c*beta(N+a+0.5,b+1)+d*beta(N+a+1.0,b+1)
        norm=beta(2+a,b+1)+c*beta(2+a+0.5,b+1)+d*beta(2+a+1.0,b+1)
        mix  = mom*M/norm
  
        return mix
  
    def set_sumrules(self):
        """
        These sum rules fix certain parameters based on physical boundaries.
        """
  
        #--valence
        #--there are 2 up valence quarks in a proton
        self.params['uv1'][0]=1    
        #self.params['uv1'][0]=2/self.get_moments('uv1',1)
        self.params['uv1'][0]=(2 - self.get_moments('uv2',1))/self.get_moments('uv1',1)
  
        #--there is 1 down valence quark in a proton
        self.params['dv1'][0]=1
        #self.params['dv1'][0]=1/self.get_moments('dv1',1)
        self.params['dv1'][0]=(1 - self.get_moments('dv2',1))/self.get_moments('dv1',1)
 
        #--strange
        #--overall strange quark number = 0
        self.params['s1'][0]=1
        self.params['s1'][0]=self.get_moments('sb1',1)/self.get_moments('s1',1)
  
        #--msr=momentum sum rule. \int_0^1 dx \sum_i{x*f_i(x)} = 1.
        sea1=self.get_moments('sea1',2)
        sea2=self.get_moments('sea2',2)
        up=self.get_moments('uv1',2) + self.get_moments('uv2',2) + 2*(sea1+self.get_moments('ub1',2))
        dv=self.get_moments('dv1',2) + self.get_moments('dv2',2) 
        dp=dv+2*(sea1+self.get_moments('db1',2))
        sp=(sea2+self.get_moments('s1',2))+(sea2+self.get_moments('sb1',2))
        self.params['g1'][0]=1    
        self.params['g1'][0]=(1-up-dp-sp)/self.get_moments('g1',2)
        g=self.get_moments('g1',2)
        msr=g+up+dp+sp
 
        #--share
        self.sr={}
        self.sr['msr']      = msr
        self.sr['uv(1)']    = self.get_moments('uv1',1)
        self.sr['dv(1)']    = self.get_moments('dv1',1)+self.get_moments('mix',1)
        self.sr['s-sb(1)']  = self.get_moments('s1',1)-self.get_moments('sb1',1)
        self.sr['s-sb(2)']  = self.get_moments('s1',2)-self.get_moments('sb1',2)
        self.sr['db-ub(1)'] = self.get_moments('db1',1)-self.get_moments('ub1',1)
        self.sr['db-ub(2)'] = self.get_moments('db1',2)-self.get_moments('ub1',2)
  
    def set_moms(self):
        sea1=self.get_moments('sea1')
        sea2=self.get_moments('sea2')
        dv=self.get_moments('dv1')+self.get_moments('mix')
  
        moms={}
        moms['g']  = self.get_moments('g1')
        moms['up'] = self.get_moments('uv1') + self.get_moments('uv2') + 2*(sea1+self.get_moments('ub1'))
        moms['dp'] = dv                      + self.get_moments('dv2') +2*(sea1+self.get_moments('db1'))
        moms['sp'] = 2*sea2+self.get_moments('s1')+self.get_moments('sb1')
        moms['um'] = self.get_moments('uv1')
        moms['dm'] = self.get_moments('dv1')
        moms['sm'] = self.get_moments('s1')-self.get_moments('sb1')

        self.moms0=moms
        self.get_BC(moms)
  
    def setup(self,par_array):
        """
        This function should be called with each update of parameters.
        The propagation of parameters sets new sum rules, which needs to update the fixed parameters.
        """
        self.update_params(par_array)
        self.set_sumrules()
        self.set_moms()
        self.storage={} 
 
    def get_moments(self,flav,N=None,b_shift=0):
        """
        if N==None: then parametrization is to be use to compute moments along mellin contour
        else the Nth moment is returned
        """
        if N==None: N=self.mellin.N
        M,a,b,c,d=self.params[flav]
        b += b_shift
        mom=beta(N+a,b+1)+c*beta(N+a+0.5,b+1)+d*beta(N+a+1.0,b+1)
        norm=beta(2+a,b+1)+c*beta(2+a+0.5,b+1)+d*beta(2+a+1.0,b+1) #--each flavor is normalized to its second moment
  
        result = M*mom/norm
  
        #--add admixture
        if flav=='dv1':  result += self.get_mix(N)
  
        return result 
  
    def _get_BC(self,g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_):
        """
        This function computes the boundary conditions given the combination of quark and gluon moments.
        """
        N=self.mellin.N
        
        # flav composition
        vm,vp={},{}
        vm[35]= bm + cm + dm + sm - 5*tm + um
        vm[24]= -4*bm + cm + dm + sm + um 
        vm[15]= -3*cm + dm + sm + um
        vm[8] = dm - 2*sp + 2*(-sm + sp) + um
        vm[3] = -dm + um
        vm[0] = np.zeros(N.size,dtype=complex)
        vp[0] = np.zeros(N.size,dtype=complex)
        vp[3] = -dp + up
        vp[8] = dp - 2*sp + up 
        vp[15]= -3*cp + dp + sp + up 
        vp[24]= -4*bp + cp + dp + sp + up 
        vp[35]= bp + cp + dp + sp - 5*tp + up 
        qs    = bp + cp + dp + sp + tp + up 
        qv    = bm + cm + dm + sm + tm + um 
        q     = np.zeros((2,N.size),dtype=complex)
        q[0]=np.copy(qs)
        q[1]=np.copy(g)
  
        BC={}
        BC['vm']=vm 
        BC['vp']=vp 
        BC['qv']=qv
        BC['q'] =q
        BC['um_']=um_
        BC['dm_']=dm_
        return BC
    
    def get_BC(self,moms):
        """
        Computes the boundary conditions for the number of flavor threshold.
        inputs:
            moms = dictionary with keys of flavor and flavor combinations whose elements are arrays of Mellin moments over the Mellin contour N.
        """
  
        N=self.mellin.N
        zero=np.zeros(N.size,dtype=complex)
  
        ###############################################
        # BC for Nf=3
        g   = moms['g']
        up  = moms['up']
        um  = moms['um']
        dp  = moms['dp']
        dm  = moms['dm']
        sp  = moms['sp']
        sm  = moms['sm']
        cp  = zero
        cm  = zero
        bp  = zero
        bm  = zero
        self.BC3=self._get_BC(g,up,um,dp,dm,sp,sm,zero,zero,zero,zero,zero,zero,um,dm)
  
        ###############################################
        # BC for Nf=4
        BC4=self.dglap.evolve(self.BC3,self.Q20,self.mc2,3)
        g =BC4['g']
        up=BC4['up']
        dp=BC4['dp']
        sp=BC4['sp']
        cp=BC4['cp']
        bp=BC4['bp']
        tp=BC4['tp']
        um=BC4['um']
        dm=BC4['dm']
        sm=BC4['sm']
        cm=BC4['cm']
        bm=BC4['bm']
        tm=BC4['tm']
        um_=BC4['um_']
        dm_=BC4['dm_']
        self.BC4=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_)
  
        ###############################################
        # BC for Nf=5
        BC5=self.dglap.evolve(self.BC4,self.mc2,self.mb2,4)
        g =BC5['g']
        up=BC5['up']
        dp=BC5['dp']
        sp=BC5['sp']
        cp=BC5['cp']
        bp=BC5['bp']
        tp=BC5['tp']
        um=BC5['um']
        dm=BC5['dm']
        sm=BC5['sm']
        cm=BC5['cm']
        bm=BC5['bm']
        tm=BC5['tm']
        um_=BC5['um_']
        dm_=BC5['dm_']
        self.BC5=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_)
  
    def evolve(self,Q2):
        """
        Evolve the moments of the QCFs up to the input Q2, for the appropriate number of flavors, Nf.
        """
        Q2 = float(Q2) #--avoid errors in Python 3
        if Q2 not in self.storage:
            if self.mb2<Q2: 
                self.storage[Q2]=self.dglap.evolve(self.BC5,self.mb2,Q2,5)
            elif self.mc2<=Q2 and Q2<=self.mb2: 
                self.storage[Q2]=self.dglap.evolve(self.BC4,self.mc2,Q2,4)
            elif Q2<self.mc2: 
                self.storage[Q2]=self.dglap.evolve(self.BC3,self.Q20,Q2,3)
    
    def get_xF(self,x,Q2,flav,evolve=True):
        """
        Calculates the x-space 1d QCF through a Mellin inversion of the moments.
        This function is used for plotting QCFs and sometimes for codes of QCD observables.
        Inputs:
            x  = momentum fraction; type = float
            Q2 = scale choice; type = float
            flav = flavor; type = str
        """
        if evolve: self.evolve(Q2)
        return x*self.mellin.invert(x,self.storage[Q2][flav])
  
    def get_xF0(self,x,flav,evolve=True):
        """
        Calculates the initial-scale x-space 1d QCF through a Mellin inversion of the moments.
        Inputs:
            x  = momentum fraction; type = float
            flav = flavor; type = str
        """
        #if   flav=='um': mom=self.moms0['um']
        #elif flav=='dm': mom=self.moms0['dm']
        #elif flav=='sm': mom=self.moms0['sm']
        if   flav=='g' : mom=self.moms0['g']
        elif flav=='u' : mom=(self.moms0['up']+self.moms0['um'])/2
        elif flav=='d' : mom=(self.moms0['dp']+self.moms0['dm'])/2
        elif flav=='s' : mom=(self.moms0['sp']+self.moms0['sm'])/2
        elif flav=='c' : mom=self.mellin.N*0
        elif flav=='b' : mom=self.mellin.N*0
        elif flav=='ub': mom=(self.moms0['up']-self.moms0['um'])/2
        elif flav=='db': mom=(self.moms0['dp']-self.moms0['dm'])/2
        elif flav=='sb': mom=(self.moms0['sp']-self.moms0['sm'])/2
        elif flav=='cb': mom=self.mellin.N*0
        elif flav=='bb': mom=self.mellin.N*0
        #mom=self.beta(self.mellin.N-0.5,3+1)
        return x*conf['mellin'].invert(x,mom)


if __name__=='__main__':

    from scipy.integrate import quad 
    mellin=MELLIN(npts=8)
    alphaS=ALPHAS()
    pdf=PDF(mellin,alphaS)

    xmin=0; xmax=1
    for Q2 in [par.mc2+1e-2,10,100]:
        integrand=lambda x: (pdf.get_xF(x,Q2,'u')-pdf.get_xF(x,Q2,'ub'))/x
        mom=quad(integrand,xmin,xmax)[0]
        print('Q2=%4.0f uv(1)=%0.2f'%(Q2,mom))

    for Q2 in [par.mc2+1e-2,10,100]:
        integrand=lambda x: (pdf.get_xF(x,Q2,'d')-pdf.get_xF(x,Q2,'db'))/x
        mom=quad(integrand,xmin,xmax)[0]
        print('Q2=%4.0f dv(1)=%0.2f'%(Q2,mom))

    for Q2 in [par.mc2+1e-2,10,100]:
        integrand=lambda x: (pdf.get_xF(x,Q2,'s')-pdf.get_xF(x,Q2,'sb'))/x
        mom=quad(integrand,xmin,xmax)[0]
        print('Q2=%4.0f sv(1)=%0.2f'%(Q2,mom))






