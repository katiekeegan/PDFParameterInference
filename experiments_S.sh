#!/bin/bash

Ss=(1 4 16)

for S in "${Ss[@]}"
    do python gan.py --S-steps "$S" &
done

# Wait for all background processes to finish
wait

echo "All tasks completed."