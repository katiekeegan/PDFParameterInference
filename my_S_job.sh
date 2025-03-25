#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -q regular
#SBATCH -J myjob
#SBATCH --mail-user=kkeega3@emory.edu
#SBATCH --mail-type=ALL
#SBATCH -t 16:00:00
#SBATCH -A m1266

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Load necessary modules
module load python
module load pytorch/2.0.1

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
srun -n 8 -c 32 --cpu_bind=cores -G 8 --gpu-bind=none  ./experiments_S.sh