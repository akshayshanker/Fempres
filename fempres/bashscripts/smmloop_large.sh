#!/bin/bash
#PBS -l ncpus=14400
#PBS -l mem=57600GB
#PBS -q normal
#PBS -P pv33
#PBS -l walltime=5:00:00
#PBS -l storage=scratch/pv33
#PBS -l wd

module load python3/3.7.4
module load openmpi/4.0.7

cd $HOME
cd Fempres
cd fempres  

mpiexec -n 14400  python3 -m mpi4py smm.py 1 Preliminary_all_v50_test Preliminary_all_v31 True True 8
 

