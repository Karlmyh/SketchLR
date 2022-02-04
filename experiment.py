
from SKLR.SKLR import SKLR
from KLR.KLR import KLR
import numpy as np
import time
import scipy



from distributions.synthetic_distributions import TestDistribution
#import importlib
#importlib.reload(some_module)


#from logistic_regression import LogisticRegression

from sklearn import linear_model,svm

n_train=128
n_test=3000
d=5



distribution=TestDistribution(dim=d,index=2).returnDistribution()
X_train,Y_train=distribution.sampling(n_train)


X_test,Y_test=distribution.sampling(n_test)


class_probability_X=distribution.density(X_train)





time_start=time.time()
model_SKLR=SKLR(X_train,Y_train)
model_SKLR.fit()
model_SKLR.predict(X_test)
print((model_SKLR.prediction==Y_test).sum()/n_test)
time_end=time.time()
print('%.2e' % (time_end-time_start))


time_start=time.time()
model_KLR=KLR(X_train,Y_train)
model_KLR.fit()
model_KLR.predict(X_test)
print((model_KLR.prediction==Y_test).sum()/n_test)
time_end=time.time()
print('%.2e' % (time_end-time_start))

time_start=time.time()
model_LR=linear_model.LogisticRegression()
model_LR.fit(X_train,Y_train)
prediction_LR=model_LR.predict(X_test)
print((prediction_LR==(Y_test)).sum()/n_test)
time_end=time.time()
print('%.2e' % (time_end-time_start))


time_start=time.time()
model_SVC=svm.SVC(tol=1e-4)
model_SVC.fit(X_train,Y_train)

prediction_SVC=model_SVC.predict(X_test)
print((prediction_SVC==(Y_test)).sum()/n_test)


time_end=time.time()
print('%.2e' % (time_end-time_start))

time_start=time.time()
model_linearSVC=svm.LinearSVC(tol=1e-4)
model_linearSVC.fit(X_train,Y_train)

prediction_linearSVC=model_linearSVC.predict(X_test)
print((prediction_linearSVC==(Y_test)).sum()/n_test)

time_end=time.time()
print('%.2e' % (time_end-time_start))
