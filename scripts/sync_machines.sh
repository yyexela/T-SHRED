#!/bin/bash

# This script synchronizes the files between the local machine and the target machine.

# Check hostname to set up target
if [ "$(hostname)" == "kutz-lambda-matrix" ]; then
    target_machine="vector"
else
    target_machine="matrix"
fi

#for f in "results" "scripts" "src" "notebooks" "configs" "bash" ".vscode" ".git" "Makefile"; do
for f in "results" "scripts" "src" "notebooks" "configs" "bash" "pickles" ".vscode" ".git" "Makefile"; do
    # Sync to target machine
    #echo "Running: rsync -az \"/home/alexey/Git/T-SHRED/$f\" $target_machine:\"/home/alexey/Git/T-SHRED/\""
    rsync -az "/home/alexey/Git/T-SHRED/$f" $target_machine:"/home/alexey/Git/T-SHRED/"
    to_result=$?
    
    # Sync from target machine
    #echo "Running: rsync -az $target_machine:\"/home/alexey/Git/T-SHRED/$f\" \"/home/alexey/Git/T-SHRED/\""
    rsync -az $target_machine:"/home/alexey/Git/T-SHRED/$f" "/home/alexey/Git/T-SHRED/"
    from_result=$?

    # Check results and print appropriate message
    if [ $to_result -ne 0 ]; then
        echo "Error: Failed to sync '$f' to $target_machine"
    elif [ $from_result -ne 0 ]; then
        echo "Error: Failed to sync '$f' from $target_machine"
    else
        echo "Success: Synced '$f' with $target_machine"
    fi
done

