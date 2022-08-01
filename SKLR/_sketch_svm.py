import numpy as np
from ._utils import kernel_matrix, optimize_jit, optimize_nonjit, sigmoid, optimize_check
from ._sketches import GaussianSketch, CountSketch, SubsamplingSketch, SRHT, Identity
from sklearn.utils import check_random_state
from time import time 



__all__=["SKSVM"]


VALID_KERNELS = [
    "additive_chi2",
    "rbf",
    "chi2",
    "poly",
    "polynomial",
    "linear",
    "sigmoid",
    "laplacian"
]

# todolist remove self object dumping in time
# contiguous array
# check dimension 200
SKETCH_DICT = {"GaussianSketch": GaussianSketch, 
               "CountSketch": CountSketch, 
               "SubsamplingSketch":SubsamplingSketch,
               "SRHT":SRHT,
               "Identity":Identity}

class SKSVM(object):
    def __init__(self,
                 sketch_dimension="auto",
                 lamda="auto",
                 kernel="auto",
                 sketch_method="auto",
                 precision=1e-2,
                 max_iter=50,
                 if_jit=False,
                 random_state=1,
                 ):
        
        
        
        
        self.kernel=kernel
        self.sketch_method=sketch_method
        
        self.sketch_dimension=sketch_dimension
        self.lamda=lamda
        
        self.precision=precision
        self.max_iter=max_iter
        
        self.if_jit=if_jit
        
        self.random_state=check_random_state(random_state).get_state()
        
        #self.prediction=np.zeros(self._n)
        #self.loss=0
        #self.iteration=0
        #self._losssave=1e8
        #self.train_time=0
        #self._alpha=np.random.normal(size=self._m)
    
    
    def _choose_kernel(self, kernel):
        
        if kernel == "auto":
            return "rbf"
        elif kernel in VALID_KERNELS:
            return kernel
        else:
            raise ValueError("invalid kernel: '{}'".format(kernel))
            
    def _choose_sketch(self, sketch_method):
        
        if sketch_method == "auto":
            return "CountSketch"
        elif sketch_method in SKETCH_DICT:
            return sketch_method
        else: 
            raise ValueError("invalid sketch: '{}'".format(sketch_method))
    
    
    def _choose_lamda(self, lamda, n, const=1):
        
        if lamda == "auto":
            return const
        elif isinstance(lamda,float) or isinstance(lamda,int):
            return lamda
        else: 
            raise ValueError("invalid lamda: '{}'".format(lamda))
            
    def _choose_sketch_dimension(self, sketch_dimension, n, const=4):
        
        if sketch_dimension == "auto":
            return int(const*np.log(n))
        elif isinstance(sketch_dimension,int):
            return sketch_dimension
        else: 
            raise ValueError("invalid sketch dimension: '{}'".format(sketch_dimension))
            
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['lamda',"sketch_dimension"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
            
            
    def fit(self, X_train, Y):
        
        if len(Y.shape)==2:
            self.Y_=Y
        elif len(Y.shape)==1:
            self.Y_=Y.reshape(-1,1)
        else:
            raise ValueError("invalid Y dimension: '{}'".format(Y.shape))
            
        if self.Y_.shape[0]==X_train.shape[0]:
            self.X_=X_train
        else:
            raise ValueError("Mismatch X dimension {} and Y dimension {}".format(X_train.shape,self.Y_.shape))
            
            
        #time_start=time()
        # set methods
        self.kernel_ = self._choose_kernel(self.kernel)
        self.sketcher_ = SKETCH_DICT[self._choose_sketch( self.sketch_method)]()
        

        # read X
        self.n_train_=X_train.shape[0]
        self.dim_=X_train.shape[1]
        
        # build kernel matrix

        
        if self._choose_sketch( self.sketch_method)=="SubsamplingSketch":
            
            # set lamda and m
            self.sketch_dimension_=self._choose_sketch_dimension(self.sketch_dimension,self.n_train_)
            self.lamda_=self._choose_lamda(self.lamda,self.n_train_)
            
            temp_state = np.random.get_state()
            np.random.set_state(self.random_state)
            
            
            hashed_index = np.random.choice(X_train.shape[0], self.sketch_dimension_, replace=False)
            
            np.random.set_state(temp_state)
            subsampled_training_data_=self.X_[hashed_index,:]
            subsampled_label_=self.Y_[hashed_index].reshape(-1,1)
            K_=kernel_matrix( subsampled_training_data_, self.kernel_)
        
       
            
            
           
            self.alpha_=np.random.normal(0,0.1,(self.sketch_dimension_,1)) 
            
            # optimize 
            if self.if_jit:
         
                self.iteration_, self.loss_, self.alpha_, self.probability_= optimize_jit(self.sketch_dimension_, self.sketch_dimension_,
                                                                                  self.lamda_, K_,
                                                                                  K_,K_,K_,
                                                                                  subsampled_label_, self.alpha_, 
                                                                                  self.precision, self.max_iter)
            else:

                self.iteration_, self.loss_, self.alpha_, self.probability_= optimize_nonjit(self.sketch_dimension_, self.sketch_dimension_,
                                                                                  self.lamda_, K_,
                                                                                  K_,K_,K_,
                                                                                  subsampled_label_, self.alpha_, 
                                                                                  self.precision, self.max_iter)
            
        else:
            K_=kernel_matrix( X_train, self.kernel_)
        
            #print("kernel")
            #print(time()-time_start)
            #time_start=time()
            
            # set lamda and m
            self.sketch_dimension_=self._choose_sketch_dimension(self.sketch_dimension,self.n_train_)
            self.lamda_=self._choose_lamda(self.lamda,self.n_train_)
            
            
            # pre-computing
            #print("dimesnsion setting")
            #print(time()-time_start)
            #time_start=time()
            SK_  = self.sketcher_.Apply(K_, sketch_dimension=self.sketch_dimension_)
           
            #print("SK")
            #print(time()-time_start)
            #time_start=time()
            KS_=np.transpose(SK_)
            #print("KS")
            #print(time()-time_start)
            #time_start=time()
            SKS_ = self.sketcher_.Apply(KS_ , sketch_dimension=self.sketch_dimension_)
            
            #print("SKS")
            #print(time()-time_start)
            #time_start=time()
            self.alpha_=np.random.normal(0,0.1,(self.sketch_dimension_,1))              
        
        
            # optimize 
            if self.if_jit:
                self.iteration_, self.loss_, self.alpha_, self.probability_= optimize_jit(self.n_train_, self.sketch_dimension_,
                                                                                  self.lamda_, K_,
                                                                                  SK_, KS_, SKS_,
                                                                                  self.Y_, self.alpha_, 
                                                                                  self.precision, self.max_iter)
            else:
                self.iteration_, self.loss_, self.alpha_, self.probability_= optimize_nonjit(self.n_train_,  self.sketch_dimension_,
                                                                                  self.lamda_, K_,
                                                                                  SK_, KS_, SKS_,
                                                                                  self.Y_, self.alpha_, 
                                                                                  self.precision, self.max_iter)
          

        
   
    
    def score(self, X, y):
        K=kernel_matrix(self.X_,self.kernel_,X)
        SK_test=self.sketcher_.Apply(K, sketch_dimension=self.sketch_dimension_)
        test_probability=sigmoid(np.matmul(self.alpha_.T, SK_test)).ravel()
        
        return ((y>0)==(test_probability>0.5)).mean()
    
    #def select_params(self, X, y, params):
        
        
    def predict(self,X_test):

      
        K_test=kernel_matrix(self.X_,self.kernel_,X_test)
        SK_test=self.sketcher_.Apply(K_test, sketch_dimension=self.sketch_dimension_)
        
        
        
        
        self.test_probability=sigmoid(np.matmul(self.alpha_.T, SK_test)).ravel()
        
        
        return self.test_probability
    
    
  