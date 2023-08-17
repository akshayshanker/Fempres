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
from pathlib import Path

#warnings.filterwarnings('ignore')

def plot_results_all(moments_sim_all,
				 moments_data_all,
				 variable_list,
				 group_list, plot_path):

	""" Plots results for visual diagnostics (not final paper)

	Paramters
	---------
	

	Returns
	-------

	Todo
	----
	- Deprecate plotting functions in plot_profiles. replace with tabulate_functions module. 
	"""

	
	line_names=['data', 'sim']
	linestyles=["-","-"]
	col_dict = {'data': 'black', 'sim':'gray'}
	markers=['x', 'o']
	ID = np.random.randint(0,9999)
	plot_path = "plot_path/plot_test_{}/".format(ID)
	Path(plot_path).mkdir(parents=True, exist_ok=True)
	# loop through each of the variables in the list
	for i, name in zip(np.arange(len(variable_list)),variable_list):
		# create a plot for this variable
		fig, ax = plt.subplots(4,2)
		ax = ax.flatten()
		#loop through each of the group
		for j, group_id in zip(np.arange(len(group_list)),group_list):
			xs = np.arange(1,12)
			ys = moments_data_all[j,:,i]
			p = ax[j].plot(xs, ys, marker=markers[0], color=col_dict['data'], linestyle=linestyles[0],
						label=line_names[0], linewidth=2)
			ys = moments_sim_all[j,:,i]
			p = ax[j].plot(xs, ys, marker=markers[1], color=col_dict['sim'], linestyle=linestyles[1],
			label=line_names[1], linewidth=2)

			ax[j].set_title(group_id)
			ax[j].spines['top'].set_visible(False)
			ax[j].spines['right'].set_visible(False)

		handles, labels = ax[7].get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper center', ncol=2)
		fig.tight_layout()

		fig.savefig("plot_path/plot_test_{}/{}.png".format(ID, name), transparent=True)

	return 

def plot_results_paper(moments_sim_all,
						moments_data_all,
						variable_list,
						variable_list_names,
						group_list_names, 
						group_list_diffs):

	line_names =['data', 'sim']
	linestyles=["-","-"]
	col_dict = {'data': 'black', 'sim':'gray'}
	markers=['x', 'o']
	ID = np.random.randint(0,9999)
	plot_path = "results/plots/plot_paper_{}/".format(ID)
	Path(plot_path).mkdir(parents=True, exist_ok=True)

	for i, name in zip(np.arange(len(variable_list)),variable_list):
		fig, ax1 = plt.subplots(3,2,gridspec_kw={"height_ratios":[0.01,1,1]})
		ax = ax1.flatten()[2:]

		for k in np.arange(len(group_list_names)):
			xs = np.arange(1,12)
			ys = moments_data_all[int(group_list_diffs[k][0]),:,i] - moments_data_all[int(group_list_diffs[k][1]),:,i]
			p = ax[k].plot(xs, ys, marker=markers[0], color=col_dict['data'], linestyle=linestyles[0],
							label=line_names[0], linewidth=2)
			ys = moments_sim_all[int(group_list_diffs[k][0]),:,i] - moments_sim_all[int(group_list_diffs[k][1]),:,i]
			p = ax[k].plot(xs, ys, marker=markers[1], color=col_dict['sim'], linestyle=linestyles[1],
								label=line_names[1], linewidth=2)
			ax[k].set_title(group_list_names[k].format(i+1), fontsize = 10)
			ax[k].spines['top'].set_visible(False)
			ax[k].spines['right'].set_visible(False)
			ax[k].ticks[[1,2,3,4,5,6,7,8,9,10,11]]
			ax[k].set_xlabel('Week')

		ax1 = ax1.flatten()
		ax1[0].axis("off")
		ax1[0].set_title('Males'.format(i+1), fontweight='bold')
		ax1[1].axis("off")
		ax1[1].set_title('Females'.format(i+1), fontweight='bold')

		handles, labels = ax[3].get_legend_handles_labels()
		#ax[0].legend(loc='upper left', ncol=2)
		ax[0].legend(handles, labels, loc='upper left', ncol=1)
		fig.tight_layout()
		fig.suptitle(variable_list_names[i])
		fig.subplots_adjust(hspace=0.5, bottom=0.1)
		fig.subplots_adjust(top=0.88)
		fig.savefig("results/plots/plot_paper_{}/{}.png".format(ID, name + '_Jan18'), transparent=True)

	return 

def plot_results_paper2(moments_sim_all,
						moments_data_all,
						variable_list,
						variable_list_names,
						group_list_names, 
						group_list_diffs, est_name):

	line_names = ['data: gr3', 'sim: gr3','data: gr2/1', 'sim: gr2/1']

	linestyles=["-","-"]
	col_dict = {'data': 'black', 'sim':'gray'}
	markers=['x', 'o']
	ID = np.random.randint(0,9999)
	plot_path = "results/plots/plot_paper2_{}_{}/".format(est_name,ID)
	Path(plot_path).mkdir(parents=True, exist_ok=True)

	for i, name in zip(np.arange(len(variable_list)),variable_list):
		fig, ax1 = plt.subplots(3,2,gridspec_kw={"height_ratios":[0.00001,1,1]})
		ax = ax1.flatten()[2:]

		for k in np.arange(len(group_list_names)):
			xs = np.arange(1,12)
			ys = moments_data_all[int(group_list_diffs[k][0]),:,i] 
			p = ax[k].plot(xs, ys, marker=markers[0], color=col_dict['data'], linestyle=linestyles[0],
							label=line_names[0], linewidth=2)
			ys = moments_data_all[int(group_list_diffs[k][1]),:,i]
			p = ax[k].plot(xs, ys, marker=markers[1], color=col_dict['data'], linestyle=linestyles[0],
				label=line_names[2], linewidth=2)

			ys = moments_sim_all[int(group_list_diffs[k][0]),:,i]
			p = ax[k].plot(xs, ys, marker=markers[0], color=col_dict['sim'], linestyle=linestyles[1],
								label=line_names[1], linewidth=2)

			ys = moments_sim_all[int(group_list_diffs[k][1]),:,i]
			p = ax[k].plot(xs, ys, marker=markers[1], color=col_dict['sim'], linestyle=linestyles[1],
								label=line_names[3], linewidth=2)


			ax[k].set_title(group_list_names[k].format(i+1), fontsize = 10)
			ax[k].spines['top'].set_visible(False)
			ax[k].spines['right'].set_visible(False)
			ax[k].set_xticks([1,2,3,4,5,6,7,8,9,10,11]) 
			


			if variable_list_names[i] == 'Book hours' or variable_list_names[i] == 'Time earning  happines units':
				ax[k].set_ylim(0,60)
				ax[k].set_yticks(np.arange(0, 60+1, 10.0))


			if variable_list_names[i] == 'MCQ hours' or variable_list_names[i] == 'SAQ hours':
				ax[k].set_ylim(0,20)
				ax[k].set_yticks(np.arange(0, 20+1, 10.0))

			if variable_list_names[i] == 'SAQ hours':
				print(ys) 


			if variable_list_names[i] == 'SAQ hours':
				ax[k].set_ylim(0,5)
				ax[k].set_yticks(np.arange(0, 5+1, 2.0))


			if variable_list_names[i] == 'MCQ attempts' or variable_list_names[i] == 'Book page views':
				ax[k].set_ylim(0,900)
				ax[k].set_yticks(np.arange(0,900+1, 200.0))

	
			if variable_list_names[i] == 'SAQ attempts':
				ax[k].set_ylim(0,20)
				ax[k].set_yticks(np.arange(0, 20+1, 4.0))


			if variable_list_names[i] == 'Happines points':
				ax[k].set_ylim(0,350)
				ax[k].set_yticks(np.arange(0, 350+1, 100.0))

		ax1 = ax1.flatten()
		ax1[0].axis("off")
		ax1[0].set_title('Males'.format(i+1), fontweight='bold')
		ax1[1].axis("off")
		ax1[1].set_title('Females'.format(i+1), fontweight='bold')
		ax[-1].set_xlabel('Week')
		ax[-2].set_xlabel('Week')

		handles, labels = ax[3].get_legend_handles_labels()
		fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0),
          fancybox=True, ncol=2)

		fig.tight_layout()
		fig.suptitle(variable_list_names[i])
		fig.subplots_adjust(hspace=0.7, bottom=0.1)
		fig.subplots_adjust(top=0.88)
		fig.subplots_adjust(bottom=0.2)
		fig.savefig("results/plots/plot_paper2_{}_{}/{}.png".format(est_name,ID, name + '_final'), transparent=True)

	return 


def array_to_ordered_list(array, param_names_new, param_names_old):

	"""
	What does this func do? Andrew?
	# INPUT: Numpy array containing parameter estimates (for either all the means or all the S.Es)
	# OUTPUT: List of lists with 8 elements in order of columns in the pdf
	# eg. 0 = Male president, Undisclosed gender, Males
	#     7 = Female president, disclosed gender, Females

	Parameters
	----------
	array : 
	param_names_new :
	param_names_old :

	Returns
	------

	Todo
	----
	Document this
	"""
	
	#Convert numpy array to a list of lists
	A_list = array.tolist()

	#Reorder parameter values according to param_names_new and store in new list of lists
	A_list_new = []
	dict_old = collections.OrderedDict(zip(param_names_old, A_list[0])) 
	dict_new = collections.OrderedDict.fromkeys(param_names_new)

	for lst in A_list:
		dict_old = collections.OrderedDict(zip(param_names_old, lst)) 
		dict_new = collections.OrderedDict.fromkeys(param_names_new)
		for key in dict_new.keys():
			dict_new[key] = dict_old[key]
		A_list_new.append(list(dict_new.values()))

	#Reorder elements of list to match male-female column orders in the table
	mf_order = [0, 1, 2, 3, 4, 5, 6, 7]
	A_list_new[:] = [A_list_new[i] for i in mf_order]
	return A_list_new


def ordered_lists_to_df(mean_list, se_list, table_row_names, table_row_symbols, pres = "male"):
	"""
	#INPUT: The means and standard error lists returned from "array_to_ordered_list"
	#and the row/symbol name lists
	#OUTPUT: Pandas data frame"""
	#Stop long strings being cut off with ...
	pd.set_option('display.max_colwidth', None)

	#Interlace the mean/S.E estimates in same order as table
	param_list = []

	#Fill param_list with estimates from mean_list/se_list, indices a:b, b not included
	a = 0
	b = 4
	if pres == "female":
		a += 4
		b += 4
	
	for i in range(a, b):
		param_list.append(mean_list[i])
		param_list.append(se_list[i])

	#Combine the numbers with row names and symbols. 
	param_list.insert(0, table_row_symbols)
	param_list.insert(0, table_row_names)

	df = pd.DataFrame(param_list).transpose()
	return df

def df_to_tex_basic(df): 
    #Format moments to be 3d.p
    moment_indices = [2,4,6,8]
    #Format standard errors to be 3d.p with scientific notation
    se_indices = [3,5,7,9]
    df[moment_indices] = df[moment_indices].applymap("{0:.3f}".format)
    df[se_indices] = df[se_indices].applymap("{0:.3E}".format)
    
    #Convert df to tex code, 3 decimal places, no row numbering
    table_tex = df.to_latex(
    header = False,
    #float_format="{:0.3f}".format, 
    column_format="lcccccccccc",
    escape = False, #stops reading of latex command symbols as text (eg. $ -> \$, \ -> \backslash)
    index = False)
    return table_tex


def tex_add_environment(table_tex, pres = "male"):
	#Add code before/after tabular environment
	#Add tabular environment/landscape to table_tex string
	table_env_start = [
		r"\begin{landscape}",
		r"\begin{table}[htbp]",
		r"\caption{\textsc{Parameter estimates}}\label{table:estimates_male_pres}",
		r"\centering"
	]
	if pres == "female":
		table_env_start[2] = r"\caption{\textsc{Parameter estimates}}\label{table:estimates_female_pres}"
	table_env_end = [
		r"\end{table}",
		r"\end{landscape}"
	]
	table_tex = "\n".join(table_env_start) + \
		"\n" + \
		table_tex + \
		"\n".join(table_env_end)
	table_tex = table_tex.replace(r"\bottomrule", r"\hline" + "\n" + r"\hline")
	return table_tex

def tex_add_col_headings(table_tex, pres = "male"):
	#Add code for multi-level column headings
	#Add multi-level column headings to table tex string
	headings = [
		r"& & \multicolumn{4}{c}{Undisclosed Gender} &\multicolumn{4}{c}{Disclosed Gender}\\",
		r"\cmidrule(l{0.2cm}r{0.3cm}){3-6}\cmidrule(l{0.2cm}r{0.3cm}){7-10}",
		r"& & \multicolumn{2}{c}{Males} &\multicolumn{2}{c}{Females} & \multicolumn{2}{c}{Males} &\multicolumn{2}{c}{Females}\\",
		r"\cmidrule(l{0.2cm}r{0.3cm}){3-4}\cmidrule(l{0.2cm}r{0.3cm}){5-6}\cmidrule(l{0.2cm}r{0.3cm}){7-8} \cmidrule(l{0.2cm}r{0.3cm}){9-10}",
		r"& & Estimates  & S.E.  & Estimates  & S.E. & Estimates  & S.E. & Estimates  & S.E. & \tabularnewline",
		r" &  &  (1) & (2) & (3)  & (4) & (5) & (6) & (7) & (8) \\",
		r"\hline",
		r"\\",
		r"\textit{\textbf{Panel A: Male President}} & & & & & & & & & \\",
	]
	if pres == "female":
		headings[-1] = r"\textit{\textbf{Panel B: Female President}} & & & & & & & & & \\"
	table_tex = table_tex.replace(r"\toprule", r"\hline" + "\n" + r"\hline" + "\n".join(headings) + "\n")
	return table_tex

def tex_add_row_headings(table_tex):
	#Add extra (multi-level) row headings:
	row_headings = [
		r"Exponential discount factor & & & & & & & & & \\" + "\n",
		r"\hline" + "\n" + r"Effort cost in relation to & & & & & & & & & \\" + "\n",
		r"\hline" + "\n" + r"Effectiveness in knowledge creation in relation to & & & & & & & & & \\" + "\n",
		r"\hline" + "\n" + r"Final exam ability & & & & & & & & & \\" + "\n"

	]
	row_heading_locations = [
		"\hspace{0.4cm}Discount factor mean",
		"\hspace{0.4cm}Time solving MCQs",
		"\hspace{0.4cm}Solving MCQ",
		"\hspace{0.4cm}Real exam ability mean"
	]

	for i in range(0, 4):
		table_tex = table_tex.replace(row_heading_locations[i], 
									  row_headings[i] + row_heading_locations[i], 1)
	return table_tex

#Wrapper function for complete transformation of df to tex table
def df_to_tex_basic(df): 
    #Format moments to be 3d.p
    moment_indices = [2,4,6,8]
    #Format standard errors to be 3d.p with scientific notation
    se_indices = [3,5,7,9]
    df[moment_indices] = df[moment_indices].applymap("{0:.3f}".format)
    df[se_indices] = df[se_indices].applymap("{0:.3f}".format)
    
    #Convert df to tex code, 3 decimal places, no row numbering
    table_tex = df.to_latex(
    header = False,
    #float_format="{:0.3f}".format, 
    column_format="lcccccccccc",
    escape = False, #stops reading of latex command symbols as text (eg. $ -> \$, \ -> \backslash)
    index = False)
    return table_tex

def df_to_tex_complete(df, pres = "male"):
	table_tex = df_to_tex_basic(df)
	table_tex = tex_add_environment(table_tex, pres)
	table_tex = tex_add_col_headings(table_tex, pres)
	table_tex = tex_add_row_headings(table_tex)
	return table_tex

def make_estimates_tables(mean_array, se_array, param_names_new, param_names_old, table_row_names, table_row_symbols, results_path, compile = False):
	#Given arrays of mean/S.E. estimates, and new vs old order of parameters, makes two tables.
	#If compile = True, then adds a preamble to tex so that output file can directly compile in LaTeX
	#Convert arrays to lists ready to be processed into dataframes.
	Path(results_path + '/tables').mkdir(parents=True, exist_ok=True)
	mean_list = array_to_ordered_list(mean_array, param_names_new, param_names_old)
	se_list = array_to_ordered_list(se_array, param_names_new, param_names_old)

	#Convert mean_list and se_list to male/female president dataframes
	df_male_pres = ordered_lists_to_df(mean_list, se_list, table_row_names, table_row_symbols, pres = "male")
	df_female_pres = ordered_lists_to_df(mean_list, se_list, table_row_names, table_row_symbols, pres = "female")

	#Convert both dataframes to tex code
	tex_male_pres = df_to_tex_complete(df_male_pres, pres = "male")
	tex_female_pres = df_to_tex_complete(df_female_pres, pres = "female")

	tex_code_final = tex_male_pres + "\n\n\n" + tex_female_pres 
	if compile:
		preamble = [
			r"\documentclass[12pt]{article}",
			r"\usepackage{geometry}",
			r"\geometry{verbose,letterpaper,tmargin=1in,bmargin=1in ,lmargin=1in,rmargin=1in,headheight=0in,headsep=0in,footskip=1.5\baselineskip}",
			r"\usepackage{array}",
			r"\usepackage{booktabs}",
			r"\usepackage{caption}",
			r"\usepackage{multirow}",
			r"\usepackage{pdflscape}",
			r"\begin{document}",
			r"\restoregeometry",
			r"\newgeometry{left=1cm, right=2cm, top=1cm, bottom=0.5cm}",
		]
		end = r"\end{document}"
		tex_code_final = "\n".join(preamble) + tex_code_final + "\n" + end
	
	#Print tex to the file table.tex. File will be created if it does not exist.
	f = open(results_path + "/tables/estimates.tex", "w+")
	f.write(tex_code_final)
	f.close()
	return
	
"""
	Functions for creating table12.tex
"""
#Reshuffle the rows and columns of data/sim_array into correct format
#Returns a numpy array ready to be converted to pandas dataframe
def order_arrays(data_array, sim_array):
	male_indices = [0, 2, 4, 6]
	female_indices = [1, 3, 5, 7]

	#Unroll arrays into vectors
	female_data = data_array[female_indices, :].ravel(order = 'F')
	male_data = data_array[male_indices, :].ravel(order = 'F')
	female_sim = sim_array[female_indices, :].ravel(order = 'F')
	male_sim = sim_array[male_indices, :].ravel(order = 'F')

	#Concatenate into numpy array
	combined_array = np.c_[male_data, male_sim, female_data, female_sim]
	return combined_array


#Convert array to dataframe and then to tex code but without multi-level headings.
def array_to_tex_basic(combined_array):
	df = pd.DataFrame(combined_array)
	pd.set_option('display.max_colwidth', None)

	#Add row names, add extra 1,2 at end to distinguish when adding multi-level headings
	#The trailing 1/2 will be removed in the function tex12_add_row_headings
	row_names = [
		r"\hspace{0.4cm}Male President, Undisclosed Gender1",
		r"\hspace{0.4cm}Male President, Disclosed Gender",
		r"\hspace{0.4cm}Female President, Undisclosed Gender",
		r"\hspace{0.4cm}Female President, Disclosed Gender",
		r"\hspace{0.4cm}Male President, Undisclosed Gender2",
		r"\hspace{0.4cm}Male President, Disclosed Gender",
		r"\hspace{0.4cm}Female President, Undisclosed Gender",
		r"\hspace{0.4cm}Female President, Disclosed Gender"
	]
	df.insert(0, "labels", row_names)
	#Add empty column for spacing
	df.insert(3, "filler1", '')
	table_tex = df.to_latex(
	header = False,
	float_format="{:0.3f}".format, 
	column_format="lccccc",
	escape = False, #stops reading of latex command symbols as text (eg. $ -> \$, \ -> \backslash)
	index = False)
	return table_tex

#Same functionality as tex_add_environment but for table 12
def tex12_add_environment(table_tex):
	table_env_start = [
		r"\begin{table}[htbp]",
		r"\caption{\textsc{Moment fit}}\label{table:fit}",
		r"\centering"
	]
	table_tex = "\n".join(table_env_start) + \
		"\n" + \
		table_tex + \
		"\n" + \
		r"\end{table}"
	table_tex = table_tex.replace(r"\bottomrule", r"\hline")
	return table_tex

#Same functionality as tex_add_col_headings but for table 12
def tex12_add_col_headings(table_tex):
	headings = [
		r"\hline",
		r"\hline",	
		r"\multicolumn{1}{c}{} & \multicolumn{2}{c}{Male Moments} &  & \multicolumn{2}{c}{Female Moments}\tabularnewline",
		r"\cline{2-6}",
		r"\multicolumn{1}{c}{} & Data  & Sim.  &  & Data  & Sim. \tabularnewline",
		r"\hline"
	]
	table_tex = table_tex.replace(r"\toprule", "\n".join(headings) + "\n")
	return table_tex

#Same functionality as tex_add_row_headings but for table 12
def tex12_add_row_headings(table_tex):
	row_headings = [
		r"\textit{Final exam grade (out of 55)} &  & &  &  & \tabularnewline" + "\n",
		r"\textit{Overall course grade (out of 100)} &  & &  &  & \tabularnewline" + "\n"
	]
	row_heading_locations = [
		r"\hspace{0.4cm}Male President, Undisclosed Gender1",
		r"\hspace{0.4cm}Male President, Undisclosed Gender2"
	]
	for i in range(0, 2):
		table_tex = table_tex.replace(row_heading_locations[i], 
									  row_headings[i] + row_heading_locations[i][:-1], 1)
	return table_tex

#Makes table 12, writes to file "table12.tex"
def make_grades_table(data_array, sim_array, results_path, compile = False):
	combined_array = order_arrays(data_array, sim_array)
	table_tex = array_to_tex_basic(combined_array)
	table_tex = tex12_add_environment(table_tex)
	table_tex = tex12_add_col_headings(table_tex)
	table_tex = tex12_add_row_headings(table_tex)

	Path(results_path + '/tables').mkdir(parents=True, exist_ok=True)

	if compile:
		preamble = [
			r"\documentclass[12pt]{article}",
			r"\usepackage{geometry}",
			r"\geometry{verbose,letterpaper,tmargin=1in,bmargin=1in ,lmargin=1in,rmargin=1in,headheight=0in,headsep=0in,footskip=1.5\baselineskip}",
			r"\usepackage{array}",
			r"\usepackage{booktabs}",
			r"\usepackage{caption}",
			r"\usepackage{multirow}",
			r"\usepackage{pdflscape}",
			r"\begin{document}",
			r"\restoregeometry",
			r"\thispagestyle{empty}",
		]
		end = r"\end{document}"
		tex_code_final = "\n".join(preamble) + table_tex + "\n" + end
	
	#Print tex to the file table12.tex. File will be created if it does not exist.
	f = open(results_path + "/tables/grades.tex", "w+")
	f.write(tex_code_final)
	f.close()
	return

#TODO: COMMENT/DELETE THIS FUNCTION AFTER HAPPY WITH ORDER OF PARAMETERS IN TABLE.
#Generate place holder arrays to check if table is arranging parameters correctly.
#If correct, make_table12 should generate table with ij in each entry, i = row, j = col
#where i starts from 1.
def generate_arrays():
	data_array = np.ones((8, 2))
	sim_array = np.ones((8, 2))
	for i in range(0, 8):
		data_col = 10 if i % 2 else 30
		sim_col = 20 if i % 2 else 40
		data_array[i, 0] = int(i/2) + data_col
		data_array[i, 1] = int(i/2) + 4 + data_col 
		sim_array[i, 0] = int(i/2) + sim_col
		sim_array[i, 1] = int(i/2) + 4 + sim_col 
	return data_array, sim_array

def extract_moment_names(filepath: str) -> list:
    """
    Extracts the second-indent variable names from the provided YAML file.

    Parameters:
    - filepath (str): The path to the input YAML file.

    Returns:
    - list: List of second-indent variable names.
    """
    
    with open(filepath, 'r') as file:
        # Load the YAML content
        data = yaml.safe_load(file)

        # Extracting the second-indent variables
        second_indent_vars = []

        for item in data['moment_names']:
            if isinstance(item, dict):  # Check if the item is a dictionary
                for key, values in item.items():
                    if isinstance(values, list):  # Ensure the values are lists
                        second_indent_vars.extend(values)

        return second_indent_vars