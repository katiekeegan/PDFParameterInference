#!/bin/bash

lambdas=(0.05 0.28571429 0.57142857 0.85714286 1.14285714 1.42857143 1.71428571 2.0 2.28571429 2.57142857 2.85714286 3.14285714 3.42857143 3.71428571 4.0)

for lmbd in "${lambdas[@]}"
    do python gan_poisson.py --lmbd "$lmbd" &
done

# Wait for all background processes to finish
wait

echo "All tasks completed."