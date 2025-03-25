#!/bin/bash
#SBATCH --job-name=pytorch_ddp       # Job name
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=4           # Number of tasks per node (4 GPUs per node)
#SBATCH --gpus-per-node=4             # Number of GPUs per node
#SBATCH --cpus-per-task=6             # Number of CPU cores per task
#SBATCH --time=04:00:00               # Time limit (2 hours)
#SBATCH --partition=gpu               # Partition to use
#SBATCH --output=ddp_output_%j.log    # Standard output and error log

# Load necessary modules (adjust as needed)
module load cuda/11.3.1
module load python
module load pytorch/1.10

# Set the master node's IP address and port for communication
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # Get the first node
MASTER_PORT=12355  # Set a specific port for communication

# Define the total number of GPUs (world size)
WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_PER_NODE))

# Run the distributed training command on each node
srun --nodes=$SLURM_NNODES --ntasks=$WORLD_SIZE \
     torchrun --nnodes=$SLURM_NNODES \
              --nproc_per_node=$SLURM_GPUS_PER_NODE \
              --node_rank=$SLURM_PROCID \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
              train.py
