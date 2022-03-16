import numpy as np
cimport numpy as np
import time
from sklearn.metrics.pairwise import pairwise_kernels
from .Sketches import SRHT
from .Sketching import SketchGaussian
from scipy.linalg import norm, inv

__all__ = ["SKLR"]


cdef class SKLR(object):

    cdef np.double_t train_time, test_time
    cdef np.double_t _precision, loss, _losssave, train_probability, test_probability
    cdef np.double_t _lamda, _learning_rate
    cdef int _dim, _n, _m, iteration
    cdef str _kernel, _method
    cdef object _X, _Y, _alpha, _K, _SK, _SKS, _X_test, _K_test, _SK_test, prediction,_KS,_SKKS

    def __init__(self,np.ndarray[np.double_t,ndim=2] X,np.ndarray[np.double_t,ndim=1] Y, sketch_dimension=0, lamda=1, kernel="rbf", method="gaussian"):

        # X n*d data matrix
        # Y n labels
        assert X.shape[0] == Y.shape[0]

        self._kernel = kernel
        self._method = method
        self._X = X
        self._Y = Y
        self._lamda = lamda
        self._dim = X.shape[1]
        self._n = X.shape[0]
        self._m = sketch_dimension
        if self._m == 0:
            #self._m=int(self._n**0.5)
            self._m = int(np.log(self._n))
        self._alpha = np.random.rand(self._m)

        self._precision = 1e-4

        #self.prediction=np.zeros(self._n)
        self.loss = 0
        self.iteration = 0
        self._losssave = 1e8
        self.train_time = 0

    #@staticmethod
    cpdef np.ndarray[np.double_t,ndim=2] kernel_matrix(self,np.ndarray[np.double_t,ndim=2] X, kernel, X_test=None):
        cdef np.ndarray[np.double_t,ndim=2] K
        if X_test is not None:
            K = pairwise_kernels(X, X_test, metric=kernel)
        else:
            K = pairwise_kernels(X, metric=kernel)

        return K

    #@staticmethod
    cpdef np.ndarray[np.double_t,ndim=2] compute_SK(self,np.ndarray[np.double_t,ndim=2] K, method,int n,int m, random_state=1):

        cdef np.ndarray[np.double_t,ndim=2] SketchSRHT,SK

        if method == "srht":
            SketchSRHT = SRHT(n, m, random_state=1)
            SK = SketchSRHT.Apply(K)
            #SKS=SketchSRHT.Apply(np.transpose(SK))
        elif method == "gaussian":
            SketchGAU = SketchGaussian(n, m, seed=random_state)
            SK = SketchGAU.Apply(K)

        return SK

    #@staticmethod
    cpdef np.ndarray[np.double_t,ndim=2] compute_SKS(self,np.ndarray[np.double_t,ndim=2] SK, method, int n,int m,int random_state=1):
        cdef np.ndarray[np.double_t,ndim=2] SketchSRHT,SKS
        if method == "srht":
            SketchSRHT = SRHT(n, m, random_state=1)
            #SK=SketchSRHT.Apply(K)
            SKS = SketchSRHT.Apply(np.transpose(SK))
        elif method == "gaussian":
            SketchGAU = SketchGaussian(n, m, seed=random_state)
            #SK=SketchSRHT.Apply(K)
            SKS = SketchGAU.Apply(np.transpose(SK))

        return SKS

    #@staticmethod
    cpdef np.ndarray[np.double_t,ndim=2] compute_SKWKS(self,np.ndarray[np.double_t,ndim=3] SKKS,int n,np.ndarray[np.double_t,ndim=1] W):
        return ((SKKS * W.reshape(n, 1, 1)).sum(axis=0))


    cpdef np.ndarray[np.double_t,ndim=1] sigmoid(self,np.ndarray[np.double_t,ndim=1] X):

        return .5 * (1 + np.tanh(.5 * X))

    #@classmethod
    cpdef np.ndarray[np.double_t,ndim=1] W(self,np.ndarray[np.double_t,ndim=2] SK,np.ndarray[np.double_t,ndim=1] Y,np.ndarray[np.double_t,ndim=1] alpha):
        #print( np.matmul(alpha,SK))
        cpdef np.ndarray[np.double_t,ndim=1] p,W


        p = self.sigmoid(np.matmul(alpha, SK))
        #sigma=cls.sigmoid( Y*np.matmul(alpha,SK))
        #W = p - p ** 2

        return p

    #@classmethod
    cpdef np.double_t cost(self,np.ndarray[np.double_t,ndim=2] KS,np.ndarray[np.double_t,ndim=1] Y,np.ndarray[np.double_t,ndim=1] alpha):
        return -np.log(self.sigmoid(Y * (np.matmul(KS, alpha))) + 1e-8).sum()

    cpdef void fit(self):

        # fit kernel

        self._K = self.kernel_matrix(self._X, self._kernel)
        self._SK = self.compute_SK(self._K, self._method, self._n, self._m)
        self._SKS = self.compute_SKS(self._SK, self._method, self._n, self._m)
        self._KS = np.transpose(self._SK)
        self._SKKS = (self._KS[:, None].reshape(self._n, self._m, 1) * self._KS[:, None])
        time_start = time.time()

        cpdef np.ndarray[np.double_t,ndim=1] p,W,add_part
        cpdef np.ndarray[np.double_t,ndim=2] SKWKS,inv_part

        while (abs(self._losssave - self.loss) > self._precision):
            self.iteration += 1
            self._losssave = self.loss
            p= self.W(self._SK, self._Y, self._alpha)
            print(self.iteration,'wrong')
            W=p-p**2

            #print(p)
            SKWKS = self.compute_SKWKS(self._SKKS, self._n, W)

            #if np.linalg.matrix_rank(SKWKS+self._lamda*self._SKS)<self._m:
            #print(self._SKS.shape)
            #print(self._m)
            #print(np.linalg.matrix_rank(self._SKS))
            #print(np.linalg.matrix_rank(SKWKS))
            #print(np.linalg.matrix_rank(SKWKS+self._lamda*self._SKS))
            #print('?')
            print('ji')
            inv_part = inv(SKWKS + self._lamda * self._SKS)
            add_part = np.matmul(SKWKS, self._alpha) + np.matmul(self._SK, self._Y * (1 - p))
            #print('??')
            self._alpha = np.matmul(inv_part, add_part)
            self.loss = self.cost(np.transpose(self._SK), self._Y, self._alpha)

        time_end = time.time()
        self.train_time = time_end - time_start
        self.train_probability = p

        print('safe')

    cpdef void predict(self,np.ndarray[np.double_t,ndim=2] X_test):
        time_start = time.time()
        self._X_test = X_test
        self._K_test = self.kernel_matrix(self._X, self._kernel, self._X_test)
        self._SK_test = self.compute_SK(self._K_test, self._method, self._n, self._m)

        self.test_probability = self.sigmoid(np.matmul(self._alpha, self._SK_test))
        self.prediction = (self.test_probability > 0.5) * 2 - 1

        time_end = time.time()
        self.test_time = time_end - time_start
        print('end of predict')



