
from SKLR.SKLR import SKLR
import numpy as np
import time
import scipy




#import importlib
#importlib.reload(some_module)


from logistic_regression import LogisticRegression

import sklearn



n=128
d=10
X0=np.random.rand(n//2,d)
X1=np.random.rand(n//2,d)
X=np.vstack([X0,-X1-1])
Y0=np.ones(n//2)
Y1=np.ones(n//2)
Y=np.append(-Y0,Y1)


time_start=time.time()
model_SKLR=SKLR(X,Y)
model_SKLR.fit()
time_end=time.time()
print(time_end-time_start)


time_start=time.time()
model_KLR=LogisticRegression(kernel="gaussian")
model_KLR.fit(X, Y)
time_end=time.time()
print(time_end-time_start)

time_start=time.time()
model_LR=sklearn.linear_model.LogisticRegression()
model_LR.fit(X, Y)
time_end=time.time()
print(time_end-time_start)

time_start=time.time()
model_LR=sklearn.linear_model.LogisticRegression()
model_LR.fit(X, Y)
time_end=time.time()
print(time_end-time_start)

time_start=time.time()
model_SVC=sklearn.svm.SVC()
model_SVC.fit(X, Y)
time_end=time.time()
print(time_end-time_start)

time_start=time.time()
model_linearSVC=sklearn.svm.linearSVC()
model_linearSVC.fit(X, Y)
time_end=time.time()
print(time_end-time_start)
