import math
import numpy as np
from time import time
from sklearn import svm
from SKLR import SKSVM

from distributions.synthetic_distributions import TestDistribution



n_train=1000
n_test=2000
d_vec=[1,5,10,50]
distribution_vec=[1,2,3,4]
sketch_dim=100
distribution=TestDistribution(dim=d,index=2).returnDistribution()
X_train,Y_train=distribution.sampling(n_train)

Y_prob=distribution.class_probability(X_train)

X_test,Y_test=distribution.sampling(n_test)
