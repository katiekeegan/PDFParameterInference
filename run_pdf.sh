#!/bin/bash

module load python
module load pytorch/2.0.1

# Run the Python script
srun --nodes=2 --ntasks-per-node=4 --cpus-per-task=4 --gpus-per-task=1 \
     --export=ALL,MASTER_ADDR=<hostname_of_node0>,MASTER_PORT=12355 \
     python cl.py