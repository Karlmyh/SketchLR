import numpy as np
from ._utils import kernel_matrix, optimize_jit, optimize_nonjit, sigmoid, optimize_check
from ._sketches import GaussianSketch, CountSketch, SubsamplingSketch, SRHT
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
               "SRHT":SRHT}

class SKSVM(object):
    def __init__(self,
                 sketch_dimension="auto",
                 lamda="auto",
                 kernel="auto",
                 sketch_method="auto",
                 precision=1e-2,
                 max_iter=50,
                 if_jit=True,
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
    
    
    def _choose_lamda(self, lamda, n, const=0.5):
        
        if lamda == "auto":
            return const/n
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
            self.subsampled_training_data_=self.X_[hashed_index,:]
            self.subsampled_label_=self.Y_[hashed_index].reshape(-1,1)
            self.K_=kernel_matrix( self.subsampled_training_data_, self.kernel_)
        
       
            
            self.SK_  = self.K_
            self.KS_  = self.K_
            self.SKS_ = self.K_
            
           
            self.alpha_=np.random.normal(0,0.1,(self.sketch_dimension_,1)) 
            
            # optimize 
            if self.if_jit:
         
                self.iteration_, self.loss_, self.alpha_, self.probability_= optimize_jit(self.sketch_dimension_, self.sketch_dimension_,
                                                                                  self.lamda_, self.K_,
                                                                                  self.SK_, self.KS_, self.SKS_,
                                                                                  self.subsampled_label_, self.alpha_, 
                                                                                  self.precision, self.max_iter)
            else:

                self.iteration_, self.loss_, self.alpha_, self.probability_= optimize_nonjit(self.sketch_dimension_, self.sketch_dimension_,
                                                                                  self.lamda_, self.K_,
                                                                                  self.SK_, self.KS_, self.SKS_,
                                                                                  self.subsampled_label_, self.alpha_, 
                                                                                  self.precision, self.max_iter)
            
        else:
            self.K_=kernel_matrix( X_train, self.kernel_)
        
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
            self.SK_  = self.sketcher_.Apply(self.K_, sketch_dimension=self.sketch_dimension_)
           
            #print("SK")
            #print(time()-time_start)
            #time_start=time()
            self.KS_=np.transpose(self.SK_)
            #print("KS")
            #print(time()-time_start)
            #time_start=time()
            self.SKS_ = self.sketcher_.Apply(self.KS_ , sketch_dimension=self.sketch_dimension_)
            
            #print("SKS")
            #print(time()-time_start)
            #time_start=time()
            self.alpha_=np.random.normal(0,0.1,(self.sketch_dimension_,1))              
        
        
            # optimize 
            if self.if_jit:
                self.iteration_, self.loss_, self.alpha_, self.probability_= optimize_jit(self.n_train_, self.sketch_dimension_,
                                                                                  self.lamda_, self.K_,
                                                                                  self.SK_, self.KS_, self.SKS_,
                                                                                  self.Y_, self.alpha_, 
                                                                                  self.precision, self.max_iter)
            else:
                self.iteration_, self.loss_, self.alpha_, self.probability_= optimize_nonjit(self.n_train_, self.sketch_dimension_,
                                                                                  self.lamda_, self.K_,
                                                                                  self.SK_, self.KS_, self.SKS_,
                                                                                  self.Y_, self.alpha_, 
                                                                                  self.precision, self.max_iter)
          

        
        
        
    def predict(self,X_test):

        self.X_test=X_test
        self.K_test=kernel_matrix(self.X_,self.kernel_,self.X_test)
        self.SK_test=self.sketcher_.Apply(self.K_test, sketch_dimension=self.sketch_dimension_)
        
        
        
        
        self.test_probability=sigmoid(np.matmul(self.alpha_.T, self.SK_test)).ravel()
        
        
        return self.test_probability
    
    
    def fit_check(self, X_train, Y):
        
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
            
            
        # set methods
        self.kernel_ = self._choose_kernel(self.kernel)
        self.sketcher_ = SKETCH_DICT[self._choose_sketch( self.sketch_method)]()
        
        # read X
        self.n_train_=X_train.shape[0]
        self.dim_=X_train.shape[1]
        
        # build kernel matrix
        self.training_data_=X_train
        self.K_=kernel_matrix( X_train, self.kernel_)
        
        # set lamda and m
        self.sketch_dimension_=self._choose_sketch_dimension(self.sketch_dimension,self.n_train_)
        self.lamda_=self._choose_lamda(self.lamda,self.n_train_)
        
        
        # pre-computing
       
        self.SK_  = self.sketcher_.Apply(self.K_, sketch_dimension=self.sketch_dimension_)
        
        self.KS_=np.transpose(self.SK_)
        self.SKS_ = self.sketcher_.Apply(self.KS_ , sketch_dimension=self.sketch_dimension_)
        
        self.alpha_=np.random.normal(0,0.1,(self.sketch_dimension_,1))
        
        # optimize 
        
        self.alpha_, self.alpha_save_= optimize_check(self.n_train_, self.sketch_dimension_,
                                                                              self.lamda_, self.K_,
                                                                              self.SK_, self.KS_, self.SKS_,
                                                                              self.Y_, self.alpha_, 
                                                                              self.precision, self.max_iter)

    
    
    def predict_check(self,X_test):

        self.X_test=X_test
        self.K_test=kernel_matrix(self.X_,self.kernel_,self.X_test)
        self.SK_test=self.sketcher_.Apply(self.K_test, sketch_dimension=self.sketch_dimension_)
        
        
        
        
        sigmoid(np.matmul(self.alpha_.T, self.SK_test))
        
        
        return sigmoid(np.matmul(self.alpha_.T, self.SK_test)), sigmoid(np.matmul(self.alpha_save_.T, self.SK_test))
            
            
        
        