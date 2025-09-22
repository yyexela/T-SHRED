#!/bin/bash

for dir in "results" "scripts" "src" "notebooks" "configs" "bash" "pickles"; do
    rsync -az "/home/alexey/Git/T-SHRED/$dir" vector:"/home/alexey/Git/T-SHRED/"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to sync $dir to vector"
    fi
    rsync -az vector:"/home/alexey/Git/T-SHRED/$dir" "/home/alexey/Git/T-SHRED/"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to sync $dir from vector"
    fi
done

