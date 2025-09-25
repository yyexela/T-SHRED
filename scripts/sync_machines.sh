#!/bin/bash

# Check if hostname is kutz-lambda-matrix
if [ "$(hostname)" == "kutz-lambda-matrix" ]; then
    target_machine="vector"
else
    target_machine="matrix"
fi

for dir in "results" "scripts" "src" "notebooks" "configs" "bash" "pickles" ".vscode"; do
    rsync -az "/home/alexey/Git/T-SHRED/$dir" $target_machine:"/home/alexey/Git/T-SHRED/"
    #echo "rsync -az \"/home/alexey/Git/T-SHRED/$dir\" $target_machine:\"/home/alexey/Git/T-SHRED/\""
    if [ $? -ne 0 ]; then
        echo "Error: Failed to sync $dir to $target_machine"
    fi
    #rsync -az $target_machine:"/home/alexey/Git/T-SHRED/$dir" "/home/alexey/Git/T-SHRED/"
    #echo "rsync -az $target_machine:\"/home/alexey/Git/T-SHRED/$dir\" \"/home/alexey/Git/T-SHRED/\""
    if [ $? -ne 0 ]; then
        echo "Error: Failed to sync $dir from $target_machine"
    fi
done

