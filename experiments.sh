#!/bin/bash

lrs=(0.0001 0.001 0.01)

for i in {0..3}
    do python gan_3.py --seed "$i" &
done

# Wait for all background processes to finish
wait

echo "All tasks completed."