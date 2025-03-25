#!/bin/bash

lrs=(0.0001 0.001 0.01)

for lr in "${lrs[@]}"
    do python gan_3.py --lr "$lr" &
done

# Wait for all background processes to finish
wait

echo "All tasks completed."