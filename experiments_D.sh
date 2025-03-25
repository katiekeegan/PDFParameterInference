#!/bin/bash

Ds=(1 5 10)

for D in "${Ds[@]}"
    do python gan.py --D-steps "$D" &
done

# Wait for all background processes to finish
wait

echo "All tasks completed."