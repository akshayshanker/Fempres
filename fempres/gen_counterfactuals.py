"""
Module generates the counterfactual profiles 

Script must be run using Mpi with 18 cores

Example (on Gadi normal compute node):

module load python3/3.7.4
module load openmpi/4.0.2

mpiexec -n 18 python3 -m mpi4py stderrs.py

"""

# Import external packages
import yaml
import gc
import numpy as np
import time
import warnings

import csv
import time
import dill as pickle 
import copy
from scipy import stats
import sys
import pandas as pd

# Import education model modules
import edumodel.edu_model as edu_model
from edumodel.studysolver import generate_study_pols

warnings.filterwarnings('ignore')
if __name__ == "__main__":

	## Set up file paths and names for results and settings ##
	## Do not edit for standard replication material ## 
	## CF simulated profiles will be saved in the counterfactuals folder
	## within the estimation folder

	# Estimation settings and labels   
	estimation_name = 'Final_MSR2_v4'
	#estimation_name = 'Preliminary_all_v31'
	#scr_path = '/g/data/pv33/edu_model_temp/' + estimation_name
	#scr_path = '/scratch/pv33/edu_model_temp/' + estimation_name
	scr_path = 'settings/estimates/' + estimation_name
	settings_folder = 'settings/'
	

	# Begin main code block 
	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)
	from pathlib import Path

	# ensure paths exist 
	Path(scr_path + '/counterfactuals')\
		.mkdir(parents=True, exist_ok=True)
	
	# world communicator class
	world = MPI4py.COMM_WORLD
	world_size = world.Get_size()
	world_rank = world.Get_rank()

	# number of cores/ number of groups = number of paramters 
	# color tells me the tau group I belong to 
	# key tells me my rank within the group 
	number_tau_groups = 2
	block_size_layer_1 = world_size/number_tau_groups
	color_layer_1 = int(world_rank/block_size_layer_1)
	key_layer1 = int(world_rank%block_size_layer_1)

	# Split the mpi communicators 
	tau_world = world.Split(color_layer_1,key_layer1)
	tau_world_rank = tau_world.Get_rank()

	# Communicators with all masters
	tau_masters = world.Split(tau_world_rank,color_layer_1)

	# Eeach core opens the baseline parameters and model
	# settings based on its color (tau group)
	# Load the data and sort and map the moments 
	# moments_data are moments for this group 
	moments_data_raw = pd.read_csv('{}moments_clean.csv'\
					.format(settings_folder))
	moments_data_mapped = edu_model.map_moments(moments_data_raw,settings_folder)
	
	# Assign model tau group according to each 
	# core according to processor color
	if color_layer_1 == 0:
		model_name = 'tau_10'
	else:
		model_name = 'tau_20'
		print(model_name)
		print(tau_world_rank)

	moments_data = moments_data_mapped[model_name]['data_moments']

	# Load model settings 
	with open("{}settings.yml".format(settings_folder), "r") as stream:
		edu_config = yaml.safe_load(stream)

	param_random_bounds = {}
	with open('{}random_param_bounds.csv'\
		.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],
																row['UB']])

	# Load the exogenous shocks 
	U = np.load(scr_path+'/'+ 'U.npy')
	U_z = np.load(scr_path+'/'+ 'U_z.npy')

	# Load the parameters for the CF group
	sampmom = pickle.load(open(scr_path + "/estimates_{}.pickle".format(model_name),"rb"))

	# Each core also loads model params for the treatment group (to get params for CF)
	sampmom_gr30  = pickle.load(open(scr_path + "/estimates_tau_30.pickle","rb"))

	# Define a function that takes in a paramter vector (possibly perturbed) and returns moments
	# Note we use the config for the tau group 
	# Covariance matrix is degenerate 

	def gen_moments(params, edu_config):
		# Create an edu model class using the input params
		param_cov = np.zeros(np.shape(sampmom[1]))
		edu_model_instance = edu_model\
				.EduModelParams(model_name,\
							edu_config[model_name],
							U, U_z, 
							random_draw = True,
							random_bounds = param_random_bounds, 
							param_random_means = params, 
							param_random_cov = param_cov, 
							uniform = False)
		# run the model 
		moments_sim = generate_study_pols(edu_model_instance.og)

		return moments_sim


	# Write list of parameters to be pertubed in CFs
	# index for parameter as that in the param_bounds.csv 
	# Note that order here is re-shuffled for 
	# final plot in paper in plot_profiles.py
	# CF 1: beta (patience)
	# CF 2: delta (self-control)
	# CF 3: alpha (ambition)
	# CF 4: psi_E (exam difficulty)
	# CF 5: zeta_star (ability)
	# CF 6: zeta_hstar (perception of ability)
	# CF 7: SAQ effort 
	# CF 8: All effort 

	cf_list = [0,3,8,9,4,6,14, 14]
	
	param_cf_adj = np.copy(np.array(sampmom[0]))
	
	# Adjust the CF param based on rank 
	# Rank 0 is the baseline
	if tau_world.rank > 0:
		param_cf_adj[int(cf_list[int(tau_world.rank-1)])]\
			 = sampmom_gr30[0][int(cf_list[int(tau_world.rank-1)])]

		if tau_world.rank == 8: 
			param_cf_adj[14] = sampmom_gr30[0][14]
			param_cf_adj[15] = sampmom_gr30[0][15]
			param_cf_adj[16] = sampmom_gr30[0][16]
			param_cf_adj[17] = sampmom_gr30[0][17]
	
	# generate the CF moments 
	tau_world.barrier()
	moments_sim_array = gen_moments(param_cf_adj,edu_config)
	world.barrier()
	moments_sim_all = tau_world.gather(moments_sim_array, root = 0)

	# Collect the 8 std errors and tau_0s on the main  and save
	if tau_world_rank == 0:
		np.save('settings/estimates/{}/counterfactuals/cf_moments_{}.npy'\
	  		.format(estimation_name,model_name),moments_sim_all)
	else:
		pass