"""
Convenience script to print out the results of the estimation in a table
while estimation is running 
"""

import dill as pickle
from tabulate import tabulate   
import numpy as np
import csv
from scipy import stats

mod_name = 'tau_30'
#save_mod_name  = 'Final_MSR2/tau_10'

estimation_name = 'Final_MSR2_v4'
#estimation_name = 'Preliminary_all_v40'

# Folder for settings in home and declare scratch path
settings_folder = 'settings/'
#scr_path = '/g/data/pv33/edu_model_temp/' + estimation_name
scr_path = 'settings/estimates/' + estimation_name
 
dict_means1 = pickle.load(open(scr_path + "/estimates_{}.pickle"\
						.format(mod_name),"rb"))
settings_folder = 'settings'
param_random_bounds = {}


with open('{}/random_param_bounds.csv'\
	.format(settings_folder), newline='') as pscfile:
	reader_ran = csv.DictReader(pscfile)
	for row in reader_ran:
		param_random_bounds[row['parameter']] = np.float64([row['LB'],\
			row['UB']])

results = {}

for i,key in zip(np.arange(len(dict_means1[0])),param_random_bounds.keys()):
	results[key]  = dict_means1[0][i], np.sqrt(dict_means1[1][i,i])

headers = ["Parameter", "Estimate", "Std. Err."]
print(tabulate([(k,) + v for k,v in results.items()], headers = headers))

