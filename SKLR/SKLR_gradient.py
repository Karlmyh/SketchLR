import numpy as npimport timefrom sklearn.metrics.pairwise import pairwise_kernelsfrom .Sketches import SRHTfrom .Sketching import SketchGaussianfrom scipy.linalg import norm,inv__all__=["SKLR_gradient"]class SKLR_gradient(object):    def __init__(self,X,Y,sketch_dimension=0,lamda=1,kernel="rbf",method="gaussian",learning_rate=0.01):                # X n*d data matrix        # Y n labels        assert X.shape[0]==Y.shape[0]                        self._kernel=kernel        self._method=method        self._X=X        self._Y=Y        self._lamda=lamda        self._dim=X.shape[1]        self._n=X.shape[0]        self._m=sketch_dimension        if self._m==0:            self._m=int(self._n**0.5)        self._alpha=np.random.normal(size=self._m)                self._precision=1e-4        self._learning_rate=learning_rate                        #self.prediction=np.zeros(self._n)        self.loss=0        self.iteration=0        self._losssave=1e8        self.train_time=0                    @staticmethod    def kernel_matrix(X,kernel,X_test=None):        if X_test is not None:            K=pairwise_kernels(X,X_test, metric=kernel)        else:            K=pairwise_kernels(X, metric=kernel)                    return K        @staticmethod    def compute_SK(K,method,n,m,random_state=1):        if method=="srht":            SketchSRHT=SRHT(n,m,random_state=1)            SK=SketchSRHT.Apply(K)            #SKS=SketchSRHT.Apply(np.transpose(SK))        elif method=="gaussian":            SketchGAU=SketchGaussian(n,m,seed=random_state)            SK=SketchGAU.Apply(K)                                        return SK        @staticmethod    def compute_SKS(SK,method,n,m,random_state=1):        if method=="srht":            SketchSRHT=SRHT(n,m,random_state=1)            #SK=SketchSRHT.Apply(K)            SKS=SketchSRHT.Apply(np.transpose(SK))        elif method=="gaussian":            SketchGAU=SketchGaussian(n,m,seed=random_state)            #SK=SketchSRHT.Apply(K)            SKS=SketchGAU.Apply(np.transpose(SK))                    return SKS        @staticmethod    def compute_SKWKS(KS,n,m,W):        return ((KS[:,None].reshape(n,m,1)*KS[:,None])*W.reshape(n,1,1)).sum(axis=0)        @staticmethod    def sigmoid(X):                return .5 * (1 + np.tanh(.5 * X))        @classmethod    def W(cls,SK,alpha):        #print( np.matmul(alpha,SK))        p=cls.sigmoid( np.matmul(alpha,SK))        #sigma=cls.sigmoid( Y*np.matmul(alpha,SK))        #W=p-p**2                return p                @classmethod    def cost(cls,KS, Y, alpha):        return -np.log(cls.sigmoid(Y*( np.matmul(KS,alpha)))+1e-8 ).sum()                        def fit(self):                # fit kernel        self._K=self.kernel_matrix(self._X,self._kernel)        self._SK=self.compute_SK(self._K,self._method,self._n,self._m)        self._SKS=self.compute_SKS(self._SK,self._method,self._n,self._m)                time_start=time.time()        while(abs(self._losssave-self.loss)>self._precision):                        self.iteration+=1            if self.iteration>100:                self._learning_rate=0.001            self._losssave=self.loss            p=self.W(self._SK,self._alpha)            #print(p)                        left=np.matmul(self._SK*(p-1),self._Y)            right=self._lamda*np.matmul(self._SKS,self._alpha)            #inv_part=inv(self._K*W+self._lamda*np.diag(np.ones(self._n)))            #add_part=np.matmul(W.reshape(-1,1)*self._K,self._alpha)+self._Y*(1-p)                                    self._alpha=self._alpha-self._learning_rate*(left+right)            #print(self._alpha)            self.loss=self.cost(np.transpose(self._SK),self._Y,self._alpha)                    time_end=time.time()        #print(self._alpha)        self.train_time=time_end-time_start        self.train_probability=p                                    def predict(self,X_test):        time_start=time.time()        self._X_test=X_test        self._K_test=self.kernel_matrix(self._X,self._kernel,self._X_test)        self._SK_test=self.compute_SK(self._K_test,self._method,self._n,self._m)                                        self.test_probability=self.sigmoid(np.matmul(self._alpha,self._SK_test))        self.prediction=(self.test_probability>0.5)*2-1                time_end=time.time()        self.test_time=time_end-time_start                                                