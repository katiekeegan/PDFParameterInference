#!/bin/bash

Dsteps=(1 5 10)
Gsteps=(1 5 10)
Ssteps=(1 5 10)

for Gstep in "${Gsteps[@]}"
    do
    for Sstep in "${Ssteps[@]}"
        do
        for Dstep in "${Dsteps[@]}"
            do python gan.py --G-steps "$Gstep" --D-steps "$Dstep" --S-steps "$Sstep" &
        done
    done
done

# Wait for all background processes to finish
wait

echo "All tasks completed."