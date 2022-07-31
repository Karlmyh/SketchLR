import math
import numpy as np
from time import time
import os

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from SKLR import SKSVM

from distributions.synthetic_distributions import TestDistribution



n_train=10000
n_test=2000
dim_vec=[5,10,50]
distribution_vec=[1,2,3]
sketch_dim=100



repeat_time=5


log_file_dir = "./result/simulation_result/"

for dim in dim_vec:
    

    for distribution_index in distribution_vec:
        



        for iterate in range(repeat_time):

            parameters={"lamda":[0.1,1,10,50]}

            np.random.seed(iterate)

            distribution=TestDistribution(dim=dim,index=distribution_index).returnDistribution()
            X_train,Y_train=distribution.sampling(n_train)



            X_test,Y_test=distribution.sampling(n_test)
            Y_prob=distribution.class_probability(X_test)[:,1]


            # Gaussian
            cv_model_SKSVM= GridSearchCV(estimator=SKSVM(sketch_method="GaussianSketch", sketch_dimension=sketch_dim), param_grid=parameters, n_jobs=1, cv=5)
            cv_model_SKSVM.fit(X_train,Y_train)
            lamda=cv_model_SKSVM.best_params_["lamda"]
            time_start=time()
            model=SKSVM(sketch_method="GaussianSketch", sketch_dimension=sketch_dim,lamda=lamda)
            model.fit(X_train,Y_train)
            pre_probability=model.predict(X_test)

            time_end=time()
            log_file_name = "{}.csv".format("GaussianSKSVM")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            accuracy=((2*(pre_probability.ravel()>0.5)-1)==Y_test).mean()
            tv_distance=np.abs(pre_probability-Y_prob).mean()
            with open(log_file_path, "a") as f:
                logs= "{},{},{},{},{},{},{},{},{},{}\n".format(distribution_index,dim,
                                             sketch_dim,iterate,n_train,n_test,accuracy,
                                             tv_distance,time_end-time_start,lamda
                                             )
                f.writelines(logs)

            # SRHT
            cv_model_SKSVM= GridSearchCV(estimator=SKSVM(sketch_method="SRHT", sketch_dimension=sketch_dim), param_grid=parameters, n_jobs=1, cv=5)
            cv_model_SKSVM.fit(X_train,Y_train)
            lamda=cv_model_SKSVM.best_params_["lamda"]
            time_start=time()
            model=SKSVM(sketch_method="SRHT", sketch_dimension=sketch_dim,lamda=lamda)
            model.fit(X_train,Y_train)
            pre_probability=model.predict(X_test)
            time_end=time()
            log_file_name = "{}.csv".format("SRHTSKSVM")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            accuracy=((2*(pre_probability.ravel()>0.5)-1)==Y_test).mean()
            tv_distance=np.abs(pre_probability-Y_prob).mean()
            with open(log_file_path, "a") as f:
                logs= "{},{},{},{},{},{},{},{},{},{}\n".format(distribution_index,dim,
                                             sketch_dim,iterate,n_train,n_test,accuracy,
                                             tv_distance,time_end-time_start,lamda
                                             )
                f.writelines(logs)

            # Count
            cv_model_SKSVM= GridSearchCV(estimator=SKSVM(sketch_method="CountSketch", sketch_dimension=sketch_dim), param_grid=parameters, n_jobs=1, cv=5)
            cv_model_SKSVM.fit(X_train,Y_train)
            lamda=cv_model_SKSVM.best_params_["lamda"]
            time_start=time()
            model=SKSVM(sketch_method="CountSketch", sketch_dimension=sketch_dim,lamda=lamda)
            model.fit(X_train,Y_train)
            pre_probability=model.predict(X_test)
            time_end=time()
            log_file_name = "{}.csv".format("CountSKSVM")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            accuracy=((2*(pre_probability.ravel()>0.5)-1)==Y_test).mean()
            tv_distance=np.abs(pre_probability-Y_prob).mean()
            with open(log_file_path, "a") as f:
                logs= "{},{},{},{},{},{},{},{},{},{}\n".format(distribution_index,dim,
                                             sketch_dim,iterate,n_train,n_test,accuracy,
                                             tv_distance,time_end-time_start,lamda
                                             )
                f.writelines(logs)


            # Subsampling
            cv_model_SKSVM= GridSearchCV(estimator=SKSVM(sketch_method="SubsamplingSketch", sketch_dimension=sketch_dim), param_grid=parameters, n_jobs=1, cv=5)
            cv_model_SKSVM.fit(X_train,Y_train)
            lamda=cv_model_SKSVM.best_params_["lamda"]
            time_start=time()
            model=SKSVM(sketch_method="SubsamplingSketch", sketch_dimension=sketch_dim,lamda=lamda)
            model.fit(X_train,Y_train)
            pre_probability=model.predict(X_test)
            time_end=time()
            log_file_name = "{}.csv".format("SubsamplingSKSVM")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            accuracy=((2*(pre_probability.ravel()>0.5)-1)==Y_test).mean()
            tv_distance=np.abs(pre_probability-Y_prob).mean()
            with open(log_file_path, "a") as f:
                logs= "{},{},{},{},{},{},{},{},{},{}\n".format(distribution_index,dim,
                                             sketch_dim,iterate,n_train,n_test,accuracy,
                                             tv_distance,time_end-time_start,lamda
                                             )
                f.writelines(logs)


            # SVC
            time_start=time()
            model=svm.SVC(tol=1e-3,probability=True)
            model.fit(X_train,Y_train)
            pre_probability=np.exp(model.predict_log_proba(X_test)[:,1])
            time_end=time()
            log_file_name = "{}.csv".format("SVC")
            log_file_path = os.path.join(log_file_dir, log_file_name)
            accuracy=((2*(pre_probability.ravel()>0.5)-1)==Y_test).mean()
            tv_distance=np.abs(pre_probability-Y_prob).mean()
            with open(log_file_path, "a") as f:
                logs= "{},{},{},{},{},{},{},{},{},{}\n".format(distribution_index,dim,
                                             sketch_dim,iterate,n_train,n_test,accuracy,
                                             tv_distance,time_end-time_start,lamda
                                             )
                f.writelines(logs)


                
                