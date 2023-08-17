
# Repository for 'The Power of Knowing a Woman is In Charge: Lessons From a Randomized Experiment'
by Dobrescu, Motta, and Shanker (2023)

## Setting Up Locally

1. Clone the repository:
```bash
git clone https://github.com/akshayshankere/Fempres.git
```

2. Set up the primary configuration with `config.yaml` located at the repository root. The primary configuration consists of:
```yaml
OpenMPI:
  version: 4.0.2
Python:
  version: 3.7.4
``` 

3. Install the necessary Python dependencies:
```bash
pip install -r requirements.txt
```

## Overview

This repository holds the code to simulate a university student's study behavior model and estimate its parameters using the simulated method of moments.

- Model representation: `EduModel` class which comprises parameters, shocks, and functions depicting student behaviors.
  
For a hands-on example, refer to the code snippet below. More details can be found in the corresponding docstrings and scripts.

```python
# Import relevant packages (as shown in gen_baseline.py)
# Declare all file paths (as shown in main block of gen_baseline.py)

from edumodel import EduModelParams, EduModel  # Import the EduModel and its wrapper class

# Create a dictionary from the settings file
with open("{}settings.yml".format(settings), "r") as stream:
    edu_config = yaml.safe_load(stream)

# Define the treatment group and name of estimates for model instantiation
model_name = 'tau_31'
estimation_name = 'Final_MSR2_v4'

# Generate or load shock seed vectors named U and U_z
# (Refer to gen_baseline.py or edumodel.py for details)

# Load the estimates for this treatment group
estimates_file_path = f"{scr_path}/estimates_{model_name}.pickle"
with open(estimates_file_path, "rb") as f:
    estimates = pickle.load(f)

# Create an instance of the wrapper class
edu_model = EduModelParams(
    model_name,  # Model name (typically the treatment group ID)
    edu_config[model_name],  # Primitive parameters
    U,  # Shock seed
    U_z,  # Shock seed for exam shocks
    random_draw=True,  # Update settings with estimated parameters
    uniform=False,  # Do not draw new parameters from uniform distribution
    param_random_means=estimates[0],  # Mean of the sampling distribution estimates
    param_random_cov=np.zeros(np.shape(estimates[1])),  # Set covariance to zero for point estimates
    random_bounds=param_random_bounds  # Parameters replaced by estimated params
)

# Import the operator that solves policy functions and generates average moments
from study_solver import generate_study_pols

# Generate simulated moments
moments_simulated = generate_study_pols(edu_model.og)

# Note: moments_simulated is an array of moments for the given treatment group.
# The first index is week and the second index is the variable name,
# as ordered in the moments_varnames.yml file in the settings folder.

```

For understanding the solution procedure, delve into the documentation for `generate_study_pols` and the functions inside `study_solver.py`.

For a comprehensive list of variable definitions, consult `settings.yml`.

## Paper Replication 

### Plot Replication
To reproduce the paper's baseline and counterfactual plots:
```bash
python3 plot_profiles.py
```

- Plots are saved in `results/plots` under a date-specific folder.
- The version of estimates in `plot_profiles.py` is `Final_MSR2_v4`.
- Plots are generated using saved simulation data in `settings/estimates/..`

### Baseline Simulation

Simulating the paper's baselines requires 8 CPUs and MPI:
```bash
mpiexec -n 8  python3 -m mpi4py gen_baseline.py
```
Ensure the desired estimation name is set in the main body of `gen_baseline.py` so that the correct estimates are read and correct file names are used when simulation results are saved.

New simulations will overwrite the previous ones in `settings/estimates` for the specified estimation name. 

### Counterfactual Simulations

Generate counterfactuals using:
```bash
mpiexec -n 18  python3 -m mpi4py gen_counterfactuals.py
```
### Simulated Method of Moments 

Following script estimates parameters using the simulated method of moments using 5760 CPU cores:
```bash
mpiexec -n 5760  python3 -m mpi4py smm.py 1 test_1 test_1 True True 8
```
The mpiexec command assumes a cluster with 5760 cores and names the estimates as `test_1` (see main body of script for system input definitions).

Example HPC PBS job scripts are provided in `bashscripts` to configure a suitable cluster. 

Before running `smm.py`, ensure to specify the appropriate scratch drive on your cluster in the main block of `smm.py`:


    # Generate paths
    # Folder for settings (where final estimates are saved) 
    # Also declare scratch path. 
    # Ensure scratch path has sufficient memory for saving 
    # monte carlo draws. 
    settings_folder = 'settings/'
    scr_path = '/scratch/pv33/edu_model_temp/' + '/' + estimation_name
    

Cross entropy hyper parameters can also be specified in the main block of `smm.py`:

    # Cross-Entropy estimation parameters  
    tol = 1E-3
    N_elite = 40
    d = 3
The scratch drive should have at least 10GB of available space to allow for SMM IO operations.

## Repository Structure

```
.
├── fempres
│   ├── bashscripts
│   │   └── *Job scripts for model simulation and SMM on cluster*
│   ├── edumodel
│   │   ├── edu_model_functions.py  # Model's behavioral functions
│   │   ├── edu_model.py            # Main model class
│   │   └── studysolver.py          # Function to solve & generate a model's time series
│   ├── settings
│   │   ├── Calibrations.xlsx       # Parameters in Excel format
│   │   ├── estimates               # Estimates and simulated profiles folder
│   │   ├── Format moments          # Input STATA data and re-shape script
│   │   ├── moments_clean.csv       # Data moments
│   │   ├── moments_varnames.yml    # Moment variable names for SMM
│   │   ├── random_param_bounds.csv # Parameter randomization bounds
│   │   └── settings.yml            # Variable definitions & parameter values
│   ├── gen_baseline.py             # Script to generate baselines
│   ├── gen_counterfactuals.py      # Counterfactuals generation script
│   ├── plot_profiles.py            # Plot profiles script
│   ├── results                     # Results and plots directory
│   ├── smm.py                      # Cross entropy estimation
│   ├── tabest.py                   # On-the-fly estimate tabulation
│   └── util                        # Utility and convenience functions
└── readme.md
```


## Naming Conventions

### Parameters
- All model parameters are defined in `settings.yml`.
- Parameters estimated via simulated method of moments are listed in `random_param_bounds.csv`.
  
For deeper insights into model instantiation and parameter selection, refer to the `EduModelParams` function documentation.

### Models & Estimates

- `estimate_name` refers to estimates from an SMM run.
- `model_name` or `mod_name` references the treatment group.
- `Param_ID` marks a particular instance of `EduModelParams`.

## Notes

- Ensure order of treatment groups in moments_clean.csv matches the order of treatment groups in `settings.yml`


