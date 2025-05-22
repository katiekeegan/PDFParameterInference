#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J ParameterLearning
#SBATCH --mail-user=kkeega3@emory.edu
#SBATCH --mail-type=ALL
#SBATCH -t 8:00:00
#SBATCH -A m1266

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
module load python
module load pytorch/2.0.1
python params_learning.py