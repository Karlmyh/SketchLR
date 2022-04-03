
from SKLR.SKLR import SKLR
from DSKLR.DSKLR import DSKLR
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

n_train=8192*4
n_test=3000
d=2



distribution=TestDistribution(dim=d,index=2).returnDistribution()
X_train,Y_train=distribution.sampling(n_train)

#print(Y_train.shape)

data=np.column_stack((X_train,Y_train))
np.random.shuffle(data)

X_train=data[:,:-1]
Y_train=data[:,-1]

X_test,Y_test=distribution.sampling(n_test)


class_probability_test=distribution.density(X_test)

print(class_probability_test.max(axis=1).mean())



time_start=time.time()
model_DSKLR=DSKLR(X_train,Y_train)
model_DSKLR.fit()
model_DSKLR.predict(X_test)
print((model_DSKLR.prediction==Y_test).sum()/n_test)
time_end=time.time()
print('%.2e' % (time_end-time_start))

time_start=time.time()
#model_SKLR=SKLR_gradient(X_train,Y_train)
#model_SKLR.fit()
#model_SKLR.predict(X_test)

#print((model_SKLR.prediction==Y_test).sum()/n_test)
time_end=time.time()
#print('%.2e' % (time_end-time_start))


time_start=time.time()
model_KLR=KLR(X_train,Y_train)
#model_KLR.fit()
#model_KLR.predict(X_test)
#print((model_KLR.prediction==Y_test).sum()/n_test)
time_end=time.time()
#print('%.2e' % (time_end-time_start))

time_start=time.time()
#model_KLR=KLR_gradient(X_train,Y_train)
#model_KLR.fit()
#model_KLR.predict(X_test)
#print((model_KLR.prediction==Y_test).sum()/n_test)
time_end=time.time()
#print('%.2e' % (time_end-time_start))

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
