import os
import numpy as np 
import pandas as pd
import glob

# simulation summarize
log_file_dir = "./result/simulation_result"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    

    log.columns = "distribution,dim,sketch_dim,iterate,n_train,n_test,accuracy,tv_distance,time".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
print(summarize_log.columns)
summary = pd.pivot_table(summarize_log, index=["dim", "method","sketch_dim"],columns=["distribution","n_train"], values=["accuracy", "time"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./result/sorted_result/simulation.xlsx")




