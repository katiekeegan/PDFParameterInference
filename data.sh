#!/bin/bash

datasizes = (1024 10240 102400)

for datasize in "${datasizes[@]}"
    do python gan.py --seed "$i" & --n-true-events "$datasize" &
done

# Wait for all background processes to finish
wait

echo "All tasks completed."