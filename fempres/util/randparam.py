"""
Module contains function to generate random parameters for the model.

Functions:
---------
rand_p_generator : function
    Function generates list of randomly drawn parameters 
    and an ID for the parameters

"""

import numpy as np

def rand_p_generator(param_deterministic,
                    param_random_bounds, # dictionary containing random parameters 
                    param_random_means = 0,  # array of means of sampling distribution
                    param_random_cov = 0,  # array of means of sampling distribution
                    deterministic = 1,   # flag to make all parmaeters determinstic 
                    initial = 1, #  
                    ):

    """ Function generates list of randomly drawn parameters and an ID for the parameters
        returns the parameter dictionary with new parameters and ID. 

    Parameters
    ----------
    param_deterministic : dictionary
        dictionary containing all parameters
    param_random_bounds : dictionary
        dictionary containing bounds for random parameters
    param_random_means : array
        array of means of sampling distribution
    param_random_cov : array
        array of covariance matrix of sampling distribution
    deterministic : int
        flag to make all parmaeters determinstic
    initial : int
        if initial then draws from uniform disitrubution within bounds

    Returns
    -------
    parameters : dictionary 
        dictionary containing the randomly generated parameters
    
    Notes
    -----
    - The chocie of which parameters are randomized in based on the list of parameters
     Pontained in the dictionary "pram_bounds"
    - parameters in param_bounds will override any deterministic parameters in parameters
    - If you want to make a parameter deterministic, remove it from the param_bounds list 
    - When deterministic flag is set to 1, all parameters are kept as they are in the 
        param_deterministic dictionary
    - To 'draw' from the means of estimates, set deterministic to 0 and make the 
        covariance matrix zero. 

    Todo
    ----
    - Make noise injection and option in the function
    - Currently gamma sum constraint is hard-coded, make this a flexible constraint
    """

    parameters = {}

    # first pull out all the parameters that are deterministic
    for key in param_deterministic:
        parameters[key] = param_deterministic[key]

    random_param_bounds_ar = \
        np.array([(bdns) for key, bdns in param_random_bounds.items()] ) 

    # generate random sample
    param_random_cov = np.array(param_random_cov)
    random_draw = np.random.uniform(0, 1)
    
    # set noise injection rate if requried 
   # if random_draw< 0.1 and initial==0:
    #    np.fill_diagonal(param_random_cov, param_random_cov.diagonal()*3)
    #if random_draw< 0.15 and random_draw> 0.1  and initial==0:
    #    initial == 1

    # Flag that is turned to True when a random draw is within the bounds
    in_range = False
    if deterministic == 0 and initial == 0:
        while in_range == False:
            draws = np.random.multivariate_normal(param_random_means, param_random_cov)
            for i,key in zip(np.arange(len(draws)),param_random_bounds.keys()):
                parameters[key]  = draws[i]

            # constraint ensures gamma parameters sum to less than 1 and that CES parameter is less than 1
            if np.sum(draws < random_param_bounds_ar[:,0]) + np.sum(draws > random_param_bounds_ar[:,1]) == 0 and\
                parameters['gamma_1'] + parameters['gamma_2'] + parameters['gamma_3'] < 1 and np.abs(parameters['sigma_M'] -1)>.01:
                in_range = True
            else:
                pass

    if deterministic == 0 and initial == 1:
        while in_range == False:

            for key in param_random_bounds:
                parameters[key]  = np.random.uniform(param_random_bounds[key][0], param_random_bounds[key][1])

            if parameters['gamma_1'] + parameters['gamma_2'] + parameters['gamma_3'] < 1 and np.abs(parameters['sigma_M'] -1)>.01:
                in_range = True

    # Generate an ID for the paramerter draw 
    parameters['ID'] = np.random.randint(0,999999999999)

    return parameters