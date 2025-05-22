#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J QuantOm
#SBATCH --mail-user=kkeega3@emory.edu
#SBATCH --mail-type=ALL
#SBATCH -t 8:00:00
#SBATCH -A m1266

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
# # Dynamically set MASTER_ADDR to the first node in the allocation
# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# MASTER_PORT=12355
# # Export required env variables
# export MASTER_ADDR MASTER_PORT
# export WORLD_SIZE=4

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
module load python
module load pytorch/2.0.1
python cl.py