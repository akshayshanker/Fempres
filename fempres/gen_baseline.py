"""
Module containing the code to simulate a baseline plot
given parameter estimates and generate the estimates 
table and fit table of grades. 

Script must be run using Mpi with number of cores = groups

Example (on Gadi normal compute node):

module load python3/3.7.4
module load openmpi/4.0.2

mpiexec -n 8 python3 -m mpi4py gen_baseline.py

Todo
----
- Make estimates table headings and order of headings
 as function inputs 

"""

# Import packages
import yaml
import gc
import time
import warnings
from collections import defaultdict
from numpy import genfromtxt
import csv
import time
import dill as pickle 
import copy
import sys
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections

# Education model modules
import edumodel.edu_model as edu_model
from util.tabulate_functions import make_estimates_tables, make_grades_table, plot_results_all, extract_moment_names
from edumodel.studysolver import generate_study_pols

warnings.filterwarnings('ignore')
if __name__ == "__main__":

	from mpi4py import MPI as MPI4py
	MPI4py.pickle.__init__(pickle.dumps, pickle.loads)
	from pathlib import Path

	## Set up file paths and names for results and settings ##
	## Do not edit for standard replication material ## 

	# Name of the model that is estimated
	estimation_name = 'Final_MSR2_v4'
	#estimation_name = 'Preliminary_all_v31'

	# Folder for settings in home 
	settings_folder = 'settings/'
	#scr_path = '/g/data/pv33/edu_model_temp/' + estimation_name
	#scr_path = '/scratch/pv33/edu_model_temp/' + estimation_name

	# Folder where the main estimation results will be saved 
	results_path = 'results/' + estimation_name

	# Path to generate testing plots for all treatment groups separately 
	plot_path_test = '/test_plots'

	## End of file paths and names for results and settings ##
	
	## Begin main code ##
	# generate scr path (direcrory where estimates live)
	scr_path = 'settings/estimates/' + estimation_name
	Path(scr_path).mkdir(parents=True, exist_ok=True)
	Path(scr_path + '/baseline').mkdir(parents=True, exist_ok=True)
	
	# Make MPI world 
	world = MPI4py.COMM_WORLD

	# Load the data and sort and map the moments 
	moments_data = pd.read_csv('{}moments_clean.csv'\
					.format(settings_folder))
	moments_data_mapped = edu_model.map_moments(moments_data, settings_folder)



	# Load generic model settings 
	with open("{}settings.yml".format(settings_folder), "r") as stream:
		edu_config = yaml.safe_load(stream)


	models_names_all = edu_config['treatment_groups'].keys()
	# Assign model tau group according to each processor according to its rank
	#print(models_names_all)
	model_name = list(models_names_all)[world.rank]


	# Load estimated moments dor tau group assigned to processor 
	sampmom = pickle.load(open(scr_path + "/estimates_{}.pickle"\
						.format(model_name),"rb"))

	# To run the model with the latest estimated moments, set the covariance
	# of the paramter distribution to zero so the mean is drawn 
	param_means = sampmom[0]
	param_cov = np.zeros(np.shape(sampmom[1]))
	U = np.load(scr_path+'/'+ 'U.npy')
	U_z = np.load(scr_path+'/'+ 'U_z.npy')

	# Run model (note, param_random_bounds serves no purpose here, 
	# TODO: it should be deprecated)
	param_random_bounds = {}
	with open('{}random_param_bounds.csv'.format(settings_folder), newline='') as pscfile:
		reader_ran = csv.DictReader(pscfile)
		for row in reader_ran:
			param_random_bounds[row['parameter']] = np.float64([row['LB'],
																row['UB']])
	
	print("Solving model for treatment group {}".format(model_name))

	# Run the model and generate the moments on each processor
	edu_model_instance = edu_model.EduModelParams(model_name,
							edu_config[model_name],
							U,
							U_z,
							random_draw = True,
							uniform = False,
							param_random_means = param_means, 
							param_random_cov = param_cov, 
							random_bounds = param_random_bounds)

	

	moments_sim = generate_study_pols(edu_model_instance.og)

	# Gather all the moments in the master node (rank 0)
	moments_sim_all = world.gather(moments_sim, root = 0)
	param_all = world.gather(param_means, root = 0)
	param_all_cov = world.gather(sampmom[1], root = 0)
	sampmom[0] = param_means
	print("Finished generating moments for treatment group {}".format(model_name))

	# Master node processes the plots and tables 
	if world.rank == 0:
		
		moments_sim_all  =  np.array(moments_sim_all)
		param_all = np.array(param_all)
		param_all_cov = np.array(param_all_cov) 
		moments_data_all = np.empty(np.shape(moments_sim_all))

		# Standard errors are SDs of the population distribution of the parameters 
		std_errs = np.empty((8, len(param_all[0,:])))
		for i in range(len(std_errs)):
			std_errs[i,:] = np.sqrt(np.diag(param_all_cov[i]))

		# re-format data moments so in the same format as sim moments, combined
		for i, keys in zip(np.arange(8),list(moments_data_mapped.keys())):
			moments_data_all[i] = moments_data_mapped[keys]['data_moments']

		# Save the simulated baselines in the same folder as the estimates 
		# note if the baseline is re-simulated, then the previous baseline is overwritten
		np.save(settings_folder + '/estimates/{}/baseline/moments_data_all.npy'\
	  			.format(estimation_name),moments_data_all)
		np.save(settings_folder + '/estimates/{}/baseline/moments_sim_all.npy'\
	  			.format(estimation_name),moments_sim_all)
		
		# Generate new data-frames for plotting 
		# Each item in moment list is a 11 x 36 array
		# The rows are are 
		list_moments = extract_moment_names(settings_folder + 'moments_varnames.yml')

		# Paramameters to be plotted as named in yml file
		# order here agrees with random_param_bounds file 
		param_names_old = [
					'beta_bar',
					'rho_beta',
					'sigma_beta',
					'delta',
					'zeta_star',
					'sigma_zeta',
					'zeta_hstar',
					'sigma_hzeta',
					'alpha',
					'lambda_E',
					'gamma_1',
					'gamma_2',
					'gamma_3',
					'sigma_M',
					'kappa_1',
					'kappa_2',
					'kappa_3',
					'kappa_4',
					'd',
					'varphi_sim']
		
		# Re-order the paramters in order of estimates table for paper 
		param_names_new = [
				'alpha',
				'beta_bar',
				'rho_beta',
				'sigma_beta',
				'delta',
				'kappa_3',
				'kappa_4',
				'kappa_1',
				'kappa_2',
				'd',
				'gamma_3',
				'gamma_1',
				'gamma_2',
				'sigma_M',
				'varphi_sim',
				'lambda_E',
				'zeta_star',
				'sigma_zeta',
				'zeta_hstar',
				'sigma_hzeta']
		
		# Row names for estimats table
		table_row_names = [
			"Course grade utility weight",
			#"Hyperbolic discount factor",
			"\hspace{0.4cm}Discount factor mean",
			"\hspace{0.4cm}Discount factor persistence",
			"\hspace{0.4cm}Discount factor std. deviation",
			"Hyperbolic discount factor",
			#"Study effectiveness for knowledge creation",
			"\hspace{0.4cm}Time solving MCQs",
			"\hspace{0.4cm}Time earning happiness units",
			"\hspace{0.4cm}Time answering SAQs",
			"\hspace{0.4cm}Time studying the textbook",
			"Knowledge stock depreciation",
			"\hspace{0.4cm}Solving MCQs",
			"\hspace{0.4cm}Answering SAQs",
			"\hspace{0.4cm}Studying the textbook",
			"Study elasticity of substitution",
			"Knowledge effectiveness of study output",
			#"Final exam ability",
			"Final exam difficulty parameter",
			"\hspace{0.4cm}Real exam ability mean",
			"\hspace{0.4cm}Real exam ability std. deviation",
			"\hspace{0.4cm}Perceived exam ability mean",
			"\hspace{0.4cm}Perceived exam ability std. deviation",
			]

		#Latex symbols for table row names, prefix r for raw string,
		#  or else causes unicode error
		table_row_symbols = [
			r"$\alpha$",
			r"$\bar{\beta}$",
			r"$\rho_{\beta}$",
			r"$\sigma_{\beta}$",
			r"$\delta$",
			r"$e^{mcq}$",
			r"$e^{sim}$",
			r"$e^{saq}$",
			r"$e^{book}$",
			r"$d$",
			r"$\gamma^{mcq}$",
			r"$\gamma^{saq}$",
			r"$\gamma^{book}$",
			r"$1/(1-\rho)$",
			r"$\vartheta$",
			r"$\lambda^E$",
			r"$\xi$",
			r"$\sigma_{\varepsilon^\xi{^*}}$",
			r"$\xi^*$",
			r"$\sigma_{\varepsilon^\xi}$"
		]

		# Get moments names from data 
		group_list = list(moments_data_mapped.keys())

		# Generate the LatTex table for the estimats
		# Note the treatment group names are hard-coded in the 
		# table generating functions in util.tabulate_functions 
		# ordering of estimates in table conforms to the ordering 
		# of parameter sets in param_all

		make_estimates_tables(param_all, std_errs, param_names_new, param_names_old, 
							table_row_names, table_row_symbols, results_path, compile=True)

		# Pull out the exam results data 
		data_exam_table = moments_data_all[:,10,0:2]
		sim_exam_table = moments_sim_all[:,10,0:2]

		# Make the Latex table for the exam results 
		make_grades_table(data_exam_table, sim_exam_table, results_path, compile = True)

		# Optional: plot all the test plots 
		#plot_results_all(moments_sim_all,moments_data_all, list_moments, group_list, plot_path_test)