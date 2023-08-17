"""
This module estimates the EduModel using the Simulated Method of Moments. 
Minimization is executed using the Cross-Entropy method (refer to Kroese et al).

Functions:
----------
- gen_RMS
- load_tm1_iter
- iter_SMM
- gen_param_moments
- map_moments

For the main block, it's essential to execute with MPI. Use 8 times the number 
of cores for each group in Tau.

Execution Example (on Gadi's standard compute node):

```bash
module load python3/3.7.4
module load openmpi/4.0.2
mpiexec -n 14400 python3 -m mpi4py smm.py Preliminary_all_v50 Preliminary_all_v50 False False 8'''

In the above example, a new estimation name is started and 
the Preliminary_all_v50 no preset is used and no saved results are loaded. 
The model is estimated for all 8 treatment groups with each group getting
14400/8 = 1800 CPU cores. 

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
import sys
import pandas as pd

# Import education model modules
import edumodel.edu_model as edu_model
from edumodel.studysolver import generate_study_pols
from util.tabulate_functions import extract_moment_names

warnings.filterwarnings('ignore')

def ssd(diff):
    """
    Returns sum of square.\

    Parameters
    ---------
    diff: float
        vector of differences

    Returns
    -------

    ssd: float
        sum of squares 

    """

    

    return np.dot( dif, dif )


def gen_RMS(edu_model,\
            moments_data, \
            moment_weights,\
            use_weights = False):
    
    """
    Simulate the model using a given parameter vector, and compute the error 
    between the simulated moments of a parameterized EduModel instance 
    and the sorted, mapped data moments.

    Parameters
    ----------
    edu_model : EduModel object
        The education model to be simulated.
    moments_data : array
        Sorted moments for a specific group.
    moment_weights : MxM array
        Moment weight matrix, where M represents the number of moments.
    use_weights : bool
        If True, uses provided weights. Otherwise, uses identity matrix.

    Returns
    -------
    error : float64
        Error of the SMM objective.

    Notes
    ----
    """

    # Solve model and generate moments 
    moments_sim = generate_study_pols(edu_model.og)

    # List moments into 1-D array (note order maintained)
    moments_sim_array = np.array(np.ravel(moments_sim))
    moments_data_array = np.array(np.ravel(moments_data))
    moments_sim_array[np.where(np.isnan(moments_sim_array))] = 1e10

    # Cacluate absolute deviation 
    deviation = (moments_sim_array\
                [~np.isnan(moments_data_array)]\
                  - moments_data_array\
                  [~np.isnan(moments_data_array)])
    
    norm  = np.sum(np.square(moments_data_array[~np.isnan(moments_data_array)]))
    N_err = len(deviation)

    # RMSE of absolute deviation 
    RMSE = 1-np.sqrt((1/N_err)*np.sum(np.square(deviation))/norm)

    # Relative deviation (percentage)
    deviation_r = (moments_sim_array\
                    [~np.isnan(moments_data_array)]\
                        - moments_data_array\
                            [~np.isnan(moments_data_array)])\
                        /moments_data_array[~np.isnan(moments_data_array)]
    
    moments_data_nonan = moments_data_array[~np.isnan(moments_data_array)]

    
    if use_weights == False:
        error = 1-ssd(deviation_r)/N_err

    else:
        error = 1- np.dot(np.dot(deviation_r.T,moment_weights), deviation_r)/N_err

    return error, deviation_r


def load_tm1_iter(tau_world, est_name, model_name, \
                    preset_estimation, load_saved = True,\
                    load_preset =True):
    """ Initializes array of best performer and least best in the 
        elite set (see notationin Kroese et al)


    Notes
    -----
    If t>0, load_preset is true, then loads model tau_00 for all gender 0 groups 
    from preset_estimation. And loads model tau_01 for all gender 1 groups 
    from preset_estimation
    """ 

    load_name1 = model_name

    if load_preset == True:
        load_name = load_name1
        est_name_load = preset_estimation
    else:
        load_name = model_name
        est_name_load = est_name

    S_star,gamma_XEM    = np.full(d+1,0), np.full(d+1,0)
    
    if load_saved == True:
        if tau_world.rank == 0:
            print("Loading preset {} from {} for model {}_{}."\
                .format(load_name,est_name_load, est_name,model_name))
        
        sampmom = pickle.load(open("/scratch/pv33/edu_model_temp/{}/latest_sampmom.smms"\
                .format(est_name_load + '/'+ load_name),"rb"))
    else:
        if tau_world.rank == 0:
            print("Generating uniform initial values for model {}_{}"\
                .format(est_name,model_name))
        sampmom = [0,0]

    return gamma_XEM, S_star,t, sampmom

def iter_SMM(config,                # Configuration settings for the model name
             model_name,            # Name of model (tau group name)
             U, U_z,
             moment_weights,
             param_random_bounds,   # Bounds for parameter draws
             sampmom,               # t-1 parameter means
             moments_data,          # Data moments
             gamma_XEM,             # Lower elite performer
             S_star,                # Upper elite performer
             t,                     # Iteration number
             tau_world,             # Communicate class for groups
             N_elite):              # Number of elite draws

    """
    Initializes parameters and the EduModel model across draws and performs 
    a single iteration of the SMM. Returns an updated sampling distribution
    for the parameters.

    Parameters
    ----------
    config : dict
        Configuration settings for the model.
    model_name : str
        Group ID (RCT x gender).
    U : NxTx2 array
        Shocks for beta and e_s.
    U_z : Nx2 array
        Shocks related to exam ability.
    moment_weights : array
        Weighting matrix.
    gamma_XEM : array
        Error for the lower elite performer.
    S_star : array
        Error for the upper elite performer.
    t : int
        Iteration number.
    tau_world : communicator class
        MPI sub-groups for RCT x gender groups.
    N_elite : int
        Number of elite draws.

    Returns
    -------
    number_N : int
        Number of elite draws
    means : array
        Means of the new sampling distribution of parameters.

    cov : matrix
        Covariance matrix of new sampling distribution of parameters.

    gamma_XEM : list
        List with bottom error of elite draws

    S_star : flaot
        List with top error of elite draws

    error_gamma : float
        difference between each iteration error associated with gamma.

    error_S : float
        difference between each iteration error associated with S_star.

    elite_indices[0] : str
        ID of the elite index (first element).

    """

    # Draw uniform parameters for the first iteration 
    if t==0:
        uniform = True
    else:
        uniform = False

    if tau_world.rank == 0:
        print("...generating sampling moments")

    edu_model_instance = edu_model\
                    .EduModelParams(model_name,\
                                config,
                                U, U_z, 
                                random_draw = True,
                                random_bounds = param_random_bounds, 
                                param_random_means = sampmom[0], 
                                param_random_cov = sampmom[1], 
                                uniform = uniform)

    parameters = edu_model_instance.parameters

    if tau_world.rank == 0:
        print("Random Parameters drawn, distributng iteration {} \
                    for tau_group {}".format(t,model_name))
    if t==0:
        use_weights= False
    else:
        use_weights= False
    RMS,deviation_r =  gen_RMS(edu_model_instance, moments_data,moment_weights, use_weights = use_weights)

    errors_ind = [edu_model_instance.param_id, np.float64(RMS)]

    # Gather parameters and corresponding errors from all ranks 
    tau_world.Barrier()

    indexed_errors = tau_world.gather(errors_ind, root = 0)

    parameter_list = tau_world.gather(parameters, root = 0)

    moments_list = tau_world.gather(deviation_r, root = 0)

    # tau_world master does selection of elite parameters and drawing new means 
    if tau_world.rank == 0:
        indexed_errors = \
            np.array([item for item in indexed_errors if item[0]!='none'])
        parameter_list\
             = [item for item in parameter_list if item is not None]

        moment_errors\
            = np.array([item for item in moments_list if item is not None])
 
        parameter_list_dict = dict([(param['param_id'], param)\
                             for param in parameter_list])
        
        errors_arr = np.array(indexed_errors[:,1]).astype(np.float64)


        error_indices_sorted = np.take(indexed_errors[:,0],\
                                     np.argsort(-errors_arr))

        errors_arr_sorted = np.take(errors_arr, np.argsort(-errors_arr))

        number_N = len(error_indices_sorted)
        
        elite_errors = errors_arr_sorted[0: N_elite]
        elite_indices = error_indices_sorted[0: N_elite]

        
        # now get the elite moment_errors 
        weights = np.exp((elite_errors - np.min(elite_errors))\
                        / (np.max(elite_errors)\
                            - np.min(elite_errors)))
        
        gamma_XEM = np.append(gamma_XEM,elite_errors[-1])
        S_star = np.append(S_star,elite_errors[0])

        error_gamma = gamma_XEM[d +t] \
                        - gamma_XEM[d +t -1]
        error_S = S_star[int(d +t)]\
                        - S_star[int(d +t -1)]

        means, cov = gen_param_moments(parameter_list_dict,\
                                param_random_bounds,\
                                elite_indices, weights)

        print("...generated and saved sampling moments")
        print("...time elapsed: {} minutes".format((time.time()-start)/60))

        return number_N, [means, cov], gamma_XEM, S_star, error_gamma,\
                                             error_S, elite_indices[0]
    else:
        return 1 

def gen_param_moments(parameter_list_dict,\
                        param_random_bounds,\
                         selected,\
                         weights):

    """
    Estimate parameters of a sampling distribution.

    Parameters
    ----------
    parameter_list_dict : dict
        Dictionary containing all randomly drawn parameters with ID keys
        for all draws. 
    param_random_bounds : dict
        Bounds for parameter draws.
    selected : 2D-array
        Set of elite parameter IDs. 
    weights : array
        Weighting matrix smm for the parameters.

    Returns
    -------
    means : array
        Means of the elite parameters.
    cov : matrix
        Covariance matrix of the elite parameters.
    """

    # create empty list to be filled with the elite parameters 
    sample_params = []

    for i in range(len(selected)):
        rand_params_i = []
        for key in param_random_bounds.keys():
            rand_params_i.append(\
                float(parameter_list_dict[selected[i]][key]))
        
        sample_params.append(rand_params_i)

    sample_params = np.array(sample_params)
    means = np.average(sample_params, axis = 0, weights = weights)
    cov = np.cov(sample_params, rowvar = 0, aweights = weights)

    return means, cov



def map_moments(moments_data, settings_path):

    """
    Convert an array of data moments into a sorted nested dictionary 
    structure where each key represents a specific group by week.

    Parameters
    ----------
    moments_data : dataframe 
        Input data where each row corresponds to a 
            specific group (defined by M x F x RCT) in a particular week.
    settings_path: string
        Path to the folder where the list of moments used in the SMM is stored. 

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
    - The returned nested dictionary has a specific structure for easy interpretation.

    Todo
    ----
    - Ensure that the `moments_data` dataframe includes a column for the group id.
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


if __name__ == "__main__":

    from mpi4py import MPI as MPI4py
    MPI4py.pickle.__init__(pickle.dumps, pickle.loads)
    from pathlib import Path
    import sys
    import ast

    # Read in system arguments 
    # if t= 0, then a new estimation is started
    # and all estimates for the estimate name are reset
    # else if t is an integer >0, then the estimation 
    # picks up saved estimates as initial value 
    t = np.int(sys.argv[1])
    # name of estimation 
    estimation_name = '{}'.format(sys.argv[2])
    # name of another estimation results  
    # that is required as an initial value 
    preset_estimation = '{}'.format(sys.argv[3])
    # load_preseet = True if start from preset_estimation 
    # if False, then start from uniform prior
    load_preset = ast.literal_eval(sys.argv[4])
    # Load saved if estimates from pre-set
    # or previous results to be loaded
    # if t>0, the load_saved is automatically True 
    load_saved = ast.literal_eval(sys.argv[5])
    # Number of treatment group 
    number_tau_groups = np.int(sys.argv[6])

    # Generate paths
    # Folder for settings (where final estimates are saved) 
    # Also declate declare scratch path. 
    # Ensure scratch path has sufficient memory for saving 
    # monte carlo draws. 
    settings_folder = 'settings/'
    scr_path = '/scratch/pv33/edu_model_temp/' + '/' + estimation_name
    
    Path(scr_path).mkdir(parents=True, exist_ok=True)
    results_path = 'settings/' + estimation_name
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # Cross-Entropy estimation parameters  
    tol = 1E-3
    N_elite = 40
    d = 3
    
    # Load MPI communicator class
    world = MPI4py.COMM_WORLD

    # Create communicators. Each processor sub-group gets a tau group
    world_size = world.Get_size()
    world_rank = world.Get_rank()
    blocks_layer_1 = number_tau_groups
    block_size_layer_1 = world_size/blocks_layer_1
    color_layer_1 = int(world_rank/block_size_layer_1)
    key_layer1 = int(world_rank%block_size_layer_1)
    tau_world = world.Split(color_layer_1,key_layer1)
    tau_world_rank = tau_world.Get_rank()

    # Load the data and sort and map the moments 
    moments_data = pd.read_csv('{}moments_clean.csv'\
                    .format(settings_folder))
    moments_data_mapped = map_moments(moments_data, settings_folder)

    # Assign model tau group according to each core according to processor color
    # Note that if tau group is 1, then only the first treatment group i.e. tau_00
    # will be estimated. 
    # if a specific treatment group needs to be estiamted, then specify as in
    # by example: 
    # model_name = ['tau_20', 'tau_30'][world.rank]
    # where number_tau_groups can be set to 2 

    model_name = list(moments_data_mapped.keys())[color_layer_1]
    Path(scr_path + '/' + model_name).mkdir(parents=True, exist_ok=True)

    # Load settings 
    with open("{}settings.yml".format(settings_folder), "r") as stream:
        edu_config = yaml.safe_load(stream)

    param_random_bounds = {}
    with open('{}random_param_bounds.csv'.format(settings_folder), newline='') as pscfile:
        reader_ran = csv.DictReader(pscfile)
        for row in reader_ran:
            param_random_bounds[row['parameter']] = np.float64([row['LB'],
                                                                row['UB']])
    
    # The error vectors for the estimation
    if world_rank == 0:
        if t == 0 or t==1:
            # Generate random points for beta and es
            U = np.random.rand(edu_config['baseline_lite']['parameters']['S'],
                        edu_config['baseline_lite']['parameters']['N'],\
                        edu_config['baseline_lite']['parameters']['T'],2)

            # Generate random points for ability and percieved ability 
            U_z = np.random.rand(edu_config['baseline_lite']['parameters']['S'],
                        edu_config['baseline_lite']['parameters']['N'],2)

            scr_path2 = '/scratch/pv33/edu_model_temp/' + '/' + estimation_name
            Path(scr_path2).mkdir(parents=True, exist_ok=True)
            np.save(scr_path2+'/'+ 'U.npy',U)
            np.save(scr_path2+'/'+ 'U_z.npy',U_z)
        else:
            pass 
    else:
        pass 

    world.Barrier() 
    U = np.load(scr_path+'/'+ 'U.npy')
    U_z = np.load(scr_path+'/'+ 'U_z.npy')

    with open('{}random_param_bounds.csv'\
    .format(settings_folder), newline='') as pscfile:
        reader_ran = csv.DictReader(pscfile)
        for row in reader_ran:
            param_random_bounds[row['parameter']] = np.float64([row['LB'],\
                row['UB']])

    start = time.time()

    # Initialize the SMM error grid
    if t == 0:
        load_saved = False
    else:
        load_saved = True
    gamma_XEM, S_star,t, sampmom = load_tm1_iter(tau_world,estimation_name,\
                                                    model_name,\
                                                     preset_estimation,
                                                     load_saved= load_saved,\
                                                     load_preset = load_preset)
    error = 1 
    
    moment_weights = 0
    
    # Iterate on the SMM
    while error > tol:
        iter_return = iter_SMM(edu_config[model_name],
                                 model_name,
                                 U, U_z, 
                                 moment_weights,
                                 param_random_bounds,
                                 sampmom,
                                 moments_data_mapped[model_name]['data_moments'],
                                 gamma_XEM,
                                 S_star,
                                 t, tau_world,N_elite) 
        tau_world.Barrier() 

        if tau_world.rank == 0:
            number_N, sampmom, gamma_XEM, S_star, error_gamma, error_S, top_ID = iter_return
            # Error is set to deviation in RMSE of bottom of elite draw parameter set 
            error_cov = np.abs(error_gamma)

            
            pickle.dump(gamma_XEM,open("/scratch/pv33/edu_model_temp/{}/{}/gamma_XEM.smms"\
                        .format(estimation_name, model_name),"wb"))
            pickle.dump(S_star,open("/scratch/pv33/edu_model_temp/{}/{}/S_star.smms"\
                        .format(estimation_name, model_name),"wb"))
            
            pickle.dump(t,open("/scratch/pv33/edu_model_temp/{}/{}/t.smms"\
                        .format(estimation_name, model_name),"wb"))
            pickle.dump(sampmom,open("/scratch/pv33/edu_model_temp/{}/{}/latest_sampmom.smms"\
                        .format(estimation_name, model_name),"wb"))
            pickle.dump(top_ID, open("/scratch/pv33/edu_model_temp/{}/{}/topid.smms"\
                        .format(estimation_name, model_name),"wb"))
            
            print("...iteration {} on {} models,elite_gamma error are {} and elite S error are {}"\
                        .format(t, number_N, error_gamma, error_S))
            
            print("...cov error is {}."\
                    .format(np.abs(np.max(sampmom[1]))))
            error = error_cov

        else:
            sampmom = None
            gamma_XEM = None
            S_star = None

        t = t+1
        tau_world.Barrier()
        sampmom = tau_world.bcast(sampmom, root = 0)
        moment_weights = tau_world.bcast(moment_weights, root = 0)
        gamma_XEM = tau_world.bcast(gamma_XEM, root = 0) 
        S_star = tau_world.bcast(S_star, root = 0)
        gc.collect()

    tau_world.Barrier()
    if tau_world.rank == 0:
        # Save final estimates in results folder in home
        pickle.dump(sampmom, open(results_path + "/estimates_{}.pickle"\
                        .format(model_name),"wb"))
