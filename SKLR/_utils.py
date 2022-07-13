from sklearn.metrics.pairwise import pairwise_kernels

from numba import njit,prange
import numpy as np



def kernel_matrix(X,kernel,X_test=None):
    if X_test is not None:
        K=pairwise_kernels(X,X_test, metric=kernel,n_jobs=-1)
    else:
        K=pairwise_kernels(X, metric=kernel,n_jobs=-1)
    return K

def sigmoid(X):
    
    return .5 * (1 + np.tanh(.5 * X))

@njit
def optimize_jit(n, m, lamda, K, SK, KS, SKS, Y, alpha, epsilon, max_iter):
    

    SKKS=np.zeros((n,m,m))
    SKWKS=np.zeros((m,m))

    for i in prange(n):
        temp=KS[i]
        temp=np.ascontiguousarray(temp)
        SKKS[i]= temp.reshape(m,1) @ temp.reshape(1,m)

    loss_save=1e10
    loss=1e5
    iteration=0
    
    p=.5 * (1 + np.tanh(.5 * KS@ alpha))
    W=p-p**2
    # flag for oscilation
    flag=1

    while(abs(loss_save-loss)>epsilon and iteration <max_iter and loss>0.1 and flag):
        
        
        iteration+=1
        
        if iteration<10:
            loss_save=loss
        else:
            if loss_save<loss:
                flag=0
            else:
                loss_save=loss
        
        #print(SKKS)
        
        for i in prange(n):
            SKWKS=SKWKS+SKKS[i]*W[i]
            
        #print(SKWKS)
        
        #inv_part=np.linalg.pinv(SKWKS+lamda * SKS, rcond=1e-6 ,hermitian=True )
        inv_part=np.linalg.inv(SKWKS+lamda * SKS )
        add_part_1=SKWKS @ alpha
        add_part_2=SK@ (Y*(1-p))
        
        
        alpha=inv_part@(add_part_1+add_part_2)
        
        loss=-np.log(.5 * (1 + np.tanh(.5 * (Y*( KS @ alpha))))+1e-30 ).mean()
        p=.5 * (1 + np.tanh(.5 * KS@ alpha))
        W=p-p**2
        
    return iteration, loss, alpha, p



def optimize_nonjit(n, m, lamda, K, SK, KS, SKS, Y, alpha, epsilon, max_iter):
    
   
    SKKS=np.zeros((n,m,m))
    SKWKS=np.zeros((m,m))

    for i in range(n):
       
        SKKS[i]= KS[i].reshape(m,1) @ KS[i].reshape(1,m)

    loss_save=1e10
    loss=1e5
    iteration=0
    
    p=.5 * (1 + np.tanh(.5 * KS@ alpha))
    W=p-p**2
    # flag for oscilation
    flag=1

    while(abs(loss_save-loss)>epsilon and iteration <max_iter and loss>0.1 and flag):
        
        
        iteration+=1
        
        if iteration<10:
            loss_save=loss
        else:
            if loss_save<loss:
                flag=0
            else:
                loss_save=loss
        
        #print(SKKS)
        
        for i in range(n):
            SKWKS=SKWKS+SKKS[i]*W[i]
            
        #print(SKWKS)
        
        inv_part=np.linalg.pinv(SKWKS+lamda * SKS,hermitian=True )
        #inv_part=np.linalg.inv(SKWKS+lamda * SKS )
        add_part_1=SKWKS @ alpha
        add_part_2=SK@ (Y*(1-p))
        
        
        alpha=inv_part@(add_part_1+add_part_2)
        
        loss=-np.log(.5 * (1 + np.tanh(.5 * (Y*( KS @ alpha))))+1e-30 ).mean()
        p=.5 * (1 + np.tanh(.5 * KS@ alpha))
        W=p-p**2
        
    return iteration, loss, alpha, p

def optimize_check(n, m, lamda, K, SK, KS, SKS, Y, alpha, epsilon, max_iter):
    SKKS=(KS[:,None].reshape(n, m, 1)*KS[:,None])
    loss_save=1e10
    loss=1e5
    iteration=0
    p=sigmoid(np.matmul(KS, alpha))
    W=p-p**2
    alpha_save=0
    flag=1
    while(abs(loss_save-loss)>epsilon and iteration <max_iter and loss>0.1 and flag):
        
        alpha_save=alpha
        iteration+=1
        
        
        loss_save=loss
        '''
        if iteration<30:
            loss_save=loss
        else:
            if loss_save<loss:
                flag=0
            else:
                loss_save=loss
        '''
        
        
        SKWKS=(SKKS*W.reshape(n,1,1)).sum(axis=0)
        
  
        
      
            
            
        inv_part=np.linalg.inv(SKWKS+lamda * SKS)
        add_part=np.matmul(SKWKS, alpha)+np.matmul(SK, Y*(1-p))
        
        
        alpha=np.matmul(inv_part,add_part)
        
        loss=-np.log(sigmoid(Y*( np.matmul(KS,alpha)))+1e-30 ).mean()
        p=sigmoid(np.matmul(KS, alpha))
        W=p-p**2

        

    return alpha, alpha_save
   

    
    
   

def find_power_2(x):
    # scale up to smallest power of 2
    
    return int(2**np.ceil(np.log2(x)))