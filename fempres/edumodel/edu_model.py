"""
This module provides the Education model class, EduModel, and a wrapper class
 to generate a draw of parameters and instantiated EduModel. 


See the main level for an example of how policy functions can be solved 
for an EduModel object.  

Classes:
-------
- EduModel(config:dict, U: array, U_z: array, param_id: str, mod_name: str):
    Create an instance of EduModel using given model primitives.

- EduModelParams(mod_name: str, param_dict: dict, U: array, U_z: array, random_bounds: array):
    A wrapper class that parameterizes EduModel based on a random
     draw of selected parameters specified in random_bounds.

Functions:
---------
- map_moments(moments_data: array):
    Convert raw moments from data into a sorted data frame.
"""


# Import packages
from util.randparam import rand_p_generator
from edumodel.edu_model_functions import edumodel_function_factory
import numpy as np
from numba import njit, prange, jit
import time
import random
import string
import dill as pickle
from sklearn.utils.extmath import cartesian
from quantecon import tauchen
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
import copy
import pandas as pd

import warnings
from util.tabulate_functions import extract_moment_names

warnings.filterwarnings('ignore')


class EduModel:
    """
    Creates an Education Model class with all necessary parameters, 
    functions, and grids for a parameterized Education Model.

    Parameters
    ----------
    config : dict
        Contains settings, parameters (which may be randomly drawn), 
        output shares, and theta.
    U : array
        Seed for random variables during the study period.
    U_z : array
        Seed for final exam shocks.
    param_id : str
        Identifier for this parameter set and instance of the class.
    mod_name : str
        Model's name, typically indicating treatment group and gender.
    """


    def __init__(self,
                 config,
                 U, 
                 U_z,
                 param_id,
                 mod_name 
                 ):
        
        self.parameters = config['parameters']
        self.constants = config['constants']
        self.theta = config['theta']
        self.config = config

        self.__dict__.update(config['parameters'])
        self.__dict__.update(config['constants'])

        self.u_grade, self.fin_exam_grade,\
            self.S_effort_to_IM, self.u_l = edumodel_function_factory(config['parameters'],
                                            config['constants'],
                                            config['theta'],
                                            config['share_saq'],
                                            config['share_eb'],
                                            config['share_mcq'],
                                            config['share_hap'])

        # 1D grids
        self.M  = np.linspace(self.M_min, self.M_max, self.grid_size_M)
        self.Mh = np.linspace(self.Mh_min, self.Mh_max, self.grid_size_Mh)
        self.U = U
        self.U_z = U_z

        # Create 2D endogenous state-space grid
        self.MM = UCGrid((self.M_min, self.M_max, self.grid_size_M),
                         (self.Mh_min, self.Mh_max, self.grid_size_Mh))

        self.MM_gp = nodes(self.MM)

        # Individual stochastic processes 
        lnr_beta_bar = np.log(1/self.beta_star - 1)
        
        y_bar = lnr_beta_bar*(1-self.rho_beta)
        
        self.lnr_beta_mc = tauchen(self.rho_beta,
                                   self.sigma_beta,
                                   b = y_bar,
                                   n = self.grid_size_beta)

        self.beta_hat = 1/(1+np.exp(self.lnr_beta_mc.state_values))

        self.P_beta = self.lnr_beta_mc.P
        self.beta_stat = self.lnr_beta_mc.stationary_distributions[0]


        self.zeta_mc = tauchen(0,
                                   self.sigma_zeta,
                                   b = self.zeta_star,
                                   n = self.grid_size_zeta)
        self.zeta_hat = np.exp(self.zeta_mc.state_values)
        self.P_zeta = self.zeta_mc.P
        self.zeta_stat = self.zeta_mc.stationary_distributions[0]


        # Perceived actual exam grading ability shock (normally distributed)
        # means and sd and rho is for the (e_c) process 
        self.zetah_mc = tauchen(0,
                                   self.sigma_hzeta,
                                   b=self.zeta_hstar,
                                   n=self.grid_size_zeta)
        self.zeta_hhat = np.exp(self.zetah_mc.state_values)
        self.P_zetah = self.zetah_mc.P
        self.zetah_stat = self.zetah_mc.stationary_distributions[0]


        # Study ability shock (log-normally distributed)
        # means and sd and rho is for the log(e_s) process 
        self.es_mc = tauchen(self.rho_es,
                                   self.sigma_es,
                                   b = self.es_star,
                                   n = self.grid_size_es)
        self.es_hat = np.exp(self.es_mc.state_values)/\
                            np.dot(np.exp(self.es_mc.state_values),\
                                    self.es_mc.stationary_distributions[0])
        self.P_es = self.es_mc.P
        self.es_stat = self.es_mc.stationary_distributions[0]


        # Combine all shock values into cartesian product 
        self.Q_shocks_ind = cartesian([np.arange(len(self.beta_hat)),\
                                        np.arange(len(self.es_hat))])

        # Build joint beta and es shock transition matrix
        self.EBA_P = np.zeros((len(self.beta_hat), len(self.es_hat),\
                                int(len(self.beta_hat)*len(self.es_hat))))

        sizeEBA =   int(len(self.beta_hat)*
                                len(self.es_hat))

        for j in self.Q_shocks_ind:

            EBA_P_temp = cartesian([self.P_beta[j[0]],
                                    self.P_es[j[1]]])

            self.EBA_P[j[0], j[1], :]\
                = EBA_P_temp[:, 0]*EBA_P_temp[:, 1]

        self.EBA_P2 = self.EBA_P.reshape((sizeEBA, sizeEBA))

        # Generate final period T expected continuation value
        VF_UC = np.zeros((len(self.zeta_hhat)*len(self.M)*len(self.Mh)))

        T_state_all = cartesian([self.zeta_hhat, self.M, self.Mh])

        for i in range(len(T_state_all)):
            zetah = T_state_all[i,0]
            m = T_state_all[i,1]
            mh = T_state_all[i,2]
            exam_mark = self.fin_exam_grade(zetah,m)
            utilT = self.u_grade(exam_mark, mh)

            VF_UC[i] = utilT

        # Condition this back
        VF_UC_1 = VF_UC.reshape((len(self.zeta_hat),len(self.M),len(self.Mh)))

        self.VT  = np.zeros((len(self.M), len(self.Mh)))

        for i_m in range(len(self.M)):
            for i_mh in range(len(self.Mh)):
                self.VT[i_m, i_mh] = np.dot(VF_UC_1[:,i_m, i_mh], self.P_zeta[0])


def map_moments(moments_data, settings_path):

    """
    Convert an array of data moments into a sorted nested dictionary 
    structure where each key represents a specific group by week.

    Parameters
    ----------
    moments_data : dataframe 
        Input data where each row corresponds to a 
            specific group (defined by M x F x RCT) in a particular week.

    Returns
    -------
    moments_grouped_sorted : dict
        A nested dictionary where:
        - Level 0 key is the group ID (e.g., tau_00, tau_01...).
        - Level 1 key 'data_moments' provides data moments for that group.
        The corresponding value is a dataframe with weeks as rows 
            and moment names as columns.

    Notes
    -----

    Todo
    ----
    """


    # Get the group names 
    group_list = moments_data['group'].unique()

    # Make a function to turn the group names into a list
    def gen_group_list(somelist):
        return {x: {} for x in somelist}

    moments_grouped_sorted = gen_group_list(group_list)

    # List the moments names (col names in the data frame)
    # and order them in the same order as the word doc 
    # the order of the list should be the same as in sim
    # moments mapped in gen_moments
    list_moments = extract_moment_names(settings_path + 'moments_varnames.yml')


    # For each group, create an empty array of sorted moments 
    for keys in moments_grouped_sorted:
        moments_grouped_sorted[keys]['data_moments'] = np.empty((11,55))

    # Map the moments to the array with cols as they are ordered
    # in list_moments for each group
    for i in range(len(list_moments)):
        for key in moments_grouped_sorted:
            moments_data_for_gr = moments_data[moments_data['group'] == key]
            moments_grouped_sorted[key]['data_moments'][:,i] \
                = moments_data_for_gr[list_moments[i]]
    # Return the sorted moments for each group
    return moments_grouped_sorted


class EduModelParams:

    """
    This is a wrapper class that parameterizes the 
    `EduModel` class. Each instance contains a 
    parameterized `EduModel` object, identified by a unique `param_id`.

    Parameters
    ----------
    mod_name : str
        Model's name.
    param_dict : dict
        Model parameters and primitives.
    U : array
        Seed for study period variables.
    U_z : array
        Seed for final exam shocks.
    random_draw : bool
        If True, draw random values from `random_bounds`.
    random_bounds : dict
        Parameters to randomize and their bounds.
    param_random_means : array
        Mean values for random parameters. Should match the order 
        in `random_bounds`.
    param_random_cov : array
        Covariance matrix for parameter sampling.
    Uniform : bool
        If True, draw from a Uniform distribution.

    Notes
    -----
    - The derived object is a parameterized `EduModel`.
    - `EduModelParams` generates the unique `param_id`.
    
    """

    def __init__(self,
                 mod_name, # name of model 
                 param_dict, 
                 U,        # beta shocks uniform draw
                 U_z,       # zeta shocks uniform draw
                 random_draw = False,   # True iff parameters generated randomly
                 random_bounds = None,  # Parameter bounds for random draws 
                 param_random_means = None,  # mean of random param distribution
                 param_random_cov = None, # cov of random param distribution
                 uniform = False):  # True iff  draw is uniform

        # Generate a random ID for this param draw 
        self.param_id = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=6))+'_'+time\
                            .strftime("%Y%m%d-%H%M%S") + '_'+mod_name
        self.mod_name = mod_name

        # If random draw is false, assign paramters from paramter dictionary pre-sets 
        if random_draw == False:
            param_deterministic = param_dict['parameters']
            parameters_draws = rand_p_generator(param_deterministic,
                                    random_bounds,
                                    deterministic = 1,
                                    initial = uniform)

            self.og = EduModel(param_dict, U, U_z, self.param_id,\
                             mod_name=mod_name)
            self.parameters = parameters_draws
            self.parameters['param_id'] = self.param_id

        if random_draw == True:
            param_deterministic = param_dict['parameters']

            parameters_draws = rand_p_generator(param_deterministic,
                                                random_bounds,
                                                deterministic = 0,
                                                initial = uniform,
                                                param_random_means = param_random_means,
                                                param_random_cov = param_random_cov)
            param_dict_new = copy.copy(param_dict)
            param_dict_new['parameters'] = parameters_draws

            self.og = EduModel(param_dict_new, U, U_z,\
                                 self.param_id, mod_name=mod_name)

            self.parameters = parameters_draws
            self.parameters['param_id'] = self.param_id



if __name__ == "__main__":

    # Below script are tests and for demonstration purposes
    # and are not for replication material

    # Generate  instance of LifeCycleParams class with
    # an instance of DB LifeCycle model and DC LS model
    import yaml
    import csv
    import pandas as pd
    from solve_policies.studysolver import generate_study_pols
    from pathlib import Path

    # Folder contains settings (grid sizes and non-random params)
    # and random param bounds
    settings = 'settings/'
    # Name of model
    model_name = 'tau_31'
    estimation_name = 'Final_MSR2_v4'
    # Path for scratch folder (will contain latest estimated means)
    scr_path = 'settings/estimates/' + estimation_name

    with open("{}settings.yml".format(settings), "r") as stream:
        edu_config = yaml.safe_load(stream)

    param_random_bounds = {}
    with open('{}random_param_bounds.csv'.format(settings), newline='') as pscfile:
        reader_ran = csv.DictReader(pscfile)
        for row in reader_ran:
            param_random_bounds[row['parameter']] = np.float64([row['LB'],
                                                                row['UB']])

    estimates = pickle.load(open(scr_path + "/estimates_{}.pickle"\
                        .format(model_name),"rb"))

    
    # Generate random points for beta and es
    U = np.random.rand(edu_config['baseline_lite']['parameters']['S'],
                        edu_config['baseline_lite']['parameters']['N'],\
                        edu_config['baseline_lite']['parameters']['T'],2)

    # Generate random points for ability and percieved ability 
    U_z = np.random.rand(edu_config['baseline_lite']['parameters']['S'],
                        edu_config['baseline_lite']['parameters']['N'],2)


    moments_data = pd.read_csv('{}moments_clean.csv'\
                    .format(settings))


    # Instantiate the wrapper class with a parameter draw
    # below is a fixed parameter draw by setting the covariance matrix 
    # to zero 

    edu_model = EduModelParams('test',
                                edu_config[model_name],
                                U,
                                U_z,
                                random_draw = True,
                                uniform = False,
                                param_random_means = estimates[0], 
                                param_random_cov = np.zeros(np.shape(estimates[1])), 
                                random_bounds = param_random_bounds)
    # the EduModel class instance is the og attribute
    # to the wrapper class 
    # Use the study solver model's `generate_study_pols' function to 
    # produce policy functions for this instance by backward iteration 

    policies = generate_study_pols(edu_model.og)




