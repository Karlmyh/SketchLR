from SKLR.SKLR import SKLR
from DSKLR.DSKLR import DSKLR
from RobustDSKLR.RSKLR import RSKLR
from SKLR.SKLR_gradient import SKLR_gradient
from KLR.KLR import KLR
from KLR.KLR_gradient import KLR_gradient
import numpy as np
import time
import scipy


from distributions.synthetic_distributions import TestDistribution
#import importlib
#importlib.reload(some_module)


#from logistic_regression import LogisticRegression

from sklearn import linear_model,svm

n_train=512
n_test=3000
num_noise=50
n_noise=n_train
d=3



distribution=TestDistribution(dim=d,index=2).returnDistribution()
X_train,Y_train=distribution.sampling(n_train)

X_train_original=X_train.copy()
Y_train_original=Y_train.copy()

X_test,Y_test=distribution.sampling(n_test)

X_noise,Y_noise=distribution.sampling(n_noise)


class_probability_X=distribution.density(X_train)




influence_SVC=0
influence_DSKLR=0
influence_RSKLR=0
influence_KLR=0




for i in range(n_noise):
    print(i)
    
    
    index=np.random.choice(n_noise,size=num_noise,replace=False)
    #print(index)
    X_train=np.vstack([np.delete(X_train_original,index,axis=0),[X_noise[i] for _ in range(num_noise) ]])
    Y_train=np.append(np.delete(Y_train_original,index),np.random.choice([-1,1],size=num_noise,p=[1/2,1/2]))
    #print(X_train.shape)
    #print(Y_train.shape)
    data=np.column_stack((X_train,Y_train))
    np.random.shuffle(data)

    X_train=data[:,:-1]
    Y_train=data[:,-1]
    

    #time_start=time.time()
    model_DSKLR=DSKLR(X_train,Y_train)
    model_DSKLR.fit()
    model_DSKLR.predict(X_test)
    acc_modified=(model_DSKLR.prediction==Y_test).sum()
    
    
   
    
    model_DSKLR=DSKLR(X_train_original,Y_train_original)
    model_DSKLR.fit()
    model_DSKLR.predict(X_test)
    acc_original=(model_DSKLR.prediction==Y_test).sum()
    
    influence_DSKLR+=abs(acc_modified-acc_original)
    #influence_DSKLR+=acc_modified-acc_original
    
    
    #time_start=time.time()
    model_RSKLR=RSKLR(X_train,Y_train)
    model_RSKLR.fit()
    model_RSKLR.predict(X_test)
    acc_modified=(model_RSKLR.prediction==Y_test).sum()
    
    
   
    
    model_RSKLR=RSKLR(X_train_original,Y_train_original)
    model_RSKLR.fit()
    model_RSKLR.predict(X_test)
    acc_original=(model_RSKLR.prediction==Y_test).sum()
    
    influence_RSKLR+=abs(acc_modified-acc_original)
    #influence_RSKLR+=acc_modified-acc_original
    
    #time_end=time.time()
    #print('%.2e' % (time_end-time_start))
    
    #time_start=time.time()
    '''
    model_KLR=KLR(X_train,Y_train)
    model_KLR.fit()
    model_KLR.predict(X_test)
    acc_modified=(model_KLR.prediction==Y_test).sum()
    
    model_KLR=KLR(X_train_original,Y_train_original)
    model_KLR.fit()    
    model_KLR.predict(X_test)
    acc_original=(model_KLR.prediction==Y_test).sum()
    
    influence_KLR+=abs(acc_modified-acc_original)
    
    '''
    
    
    model_SVC=svm.SVC(tol=1e-3)
    model_SVC.fit(X_train,Y_train)
    
    prediction_SVC=model_SVC.predict(X_test)
    acc_modified=(prediction_SVC==(Y_test)).sum()
    
    model_SVC=svm.SVC(tol=1e-3)
    model_SVC.fit(X_train_original,Y_train_original)
    
    prediction_SVC=model_SVC.predict(X_test)
    acc_original=(prediction_SVC==(Y_test)).sum()
    

    influence_SVC+=abs(acc_modified-acc_original)
    #influence_SVC+=acc_modified-acc_original

print([influence_DSKLR,influence_RSKLR,influence_SVC])