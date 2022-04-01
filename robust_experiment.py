
from SKLR.SKLR import SKLR
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

n_train=128
n_test=500
n_noise=n_train
d=2



distribution=TestDistribution(dim=d,index=2).returnDistribution()
X_train,Y_train=distribution.sampling(n_train)

X_train_original=X_train.copy()
Y_train_original=Y_train.copy()

X_test,Y_test=distribution.sampling(n_test)

X_noise,Y_noise=distribution.sampling(n_noise)


class_probability_X=distribution.density(X_train)


num_noise=50

influence_SVC=0
influence_SKLR=0
influence_KLR=0

for i in range(n_noise):
    
    X_train[i]=X_noise[i]
    Y_train[i]=Y_noise[i]



    #time_start=time.time()
    model_SKLR=SKLR(X_train,Y_train)
    model_SKLR.fit()
    model_SKLR.predict(X_test)
    acc_modified=(model_SKLR.prediction==Y_test).sum()
    
    model_SKLR=SKLR(X_train_original,Y_train_original)
    model_SKLR.fit()
    model_SKLR.predict(X_test)
    acc_original=(model_SKLR.prediction==Y_test).sum()
    
    influence_SKLR+=abs(acc_modified-acc_original)
    
    #time_end=time.time()
    #print('%.2e' % (time_end-time_start))
    
    #time_start=time.time()
    model_KLR=KLR(X_train,Y_train)
    model_KLR.fit()
    model_KLR.predict(X_test)
    acc_modified=(model_KLR.prediction==Y_test).sum()
    
    model_KLR=KLR(X_train_original,Y_train_original)
    model_KLR.fit()
    model_KLR.predict(X_test)
    acc_original=(model_KLR.prediction==Y_test).sum()
    
    influence_KLR+=abs(acc_modified-acc_original)
    
    
    
    
    model_SVC=svm.SVC(tol=1e-3)
    model_SVC.fit(X_train,Y_train)
    
    prediction_SVC=model_SVC.predict(X_test)
    acc_modified=(prediction_SVC==(Y_test)).sum()
    
    model_SVC=svm.SVC(tol=1e-3)
    model_SVC.fit(X_train_original,Y_train_original)
    
    prediction_SVC=model_SVC.predict(X_test)
    acc_original=(prediction_SVC==(Y_test)).sum()
    

    influence_SVC+=abs(acc_modified-acc_original)

print([influence_SKLR,influence_SVC])