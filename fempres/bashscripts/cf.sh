#!/bin/bash
#PBS -l ncpus=18
#PBS -l mem=192GB
#PBS -q normal
#PBS -P pv33
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/pv33
#PBS -l wd


cd $HOME
cd Fempres
cd fempres  


module load python3/3.7.4
module load openmpi/4.0.2

mpiexec -n 18  python3 -m mpi4py gen_counterfactuals.py
mpiexec -n 8  python3 -m mpi4py gen_baseline.py
python3 plot_profiles.py