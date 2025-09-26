#!/bin/bash

# Check hostname to set up target
if [ "$(hostname)" == "kutz-lambda-matrix" ]; then
    target_machine="vector"
else
    target_machine="matrix"
fi

for dir in "results" "scripts" "src" "notebooks" "configs" "bash" "pickles" "logs" ".vscode" ".git"; do
    # Sync to target machine
    #echo "Running: rsync -az \"/home/alexey/Git/T-SHRED/$dir\" $target_machine:\"/home/alexey/Git/T-SHRED/\""
    rsync -az "/home/alexey/Git/T-SHRED/$dir" $target_machine:"/home/alexey/Git/T-SHRED/"
    to_result=$?
    
    # Sync from target machine
    #echo "Running: rsync -az $target_machine:\"/home/alexey/Git/T-SHRED/$dir\" \"/home/alexey/Git/T-SHRED/\""
    rsync -az $target_machine:"/home/alexey/Git/T-SHRED/$dir" "/home/alexey/Git/T-SHRED/"
    from_result=$?
    
    # Check results and print appropriate message
    if [ $to_result -ne 0 ]; then
        echo "Error: Failed to sync '$dir/' to $target_machine"
    elif [ $from_result -ne 0 ]; then
        echo "Error: Failed to sync '$dir/' from $target_machine"
    else
        echo "Success: Synced '$dir/' with $target_machine"
    fi
done

