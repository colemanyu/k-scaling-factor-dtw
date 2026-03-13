#!/bin/bash

# Parameters
dataset="GunPoint"
Ps=(2 3 4)
ls=(1.25 1.50 1.75 2.00)
methods=(0 1)

# Get the absolute path to the current directory (code/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Iterate over P and dist_method to create sessions (3 * 2 = 6 sessions)
# Inside each session, iterate over l values (4 runs sequentially)
for P in "${Ps[@]}"
do
  for method in "${methods[@]}"
  do
    # Create a unique session name
    session_name="search_${dataset}_P${P}_m${method}"
    
    echo "Creating tmux session: $session_name"
    
    # Create a new detached session
    tmux new-session -d -s "$session_name"
    
    # Send commands to the session
    tmux send-keys -t "$session_name" "bash" C-m
    
    # 1. Activate the conda environment (assuming 'ksfdtw' as used in the other script)
    tmux send-keys -t "$session_name" "conda activate ksfdtw" C-m
    
    # 2. Change to the code directory
    tmux send-keys -t "$session_name" "cd $SCRIPT_DIR" C-m
    
    # 3. Queue commands for each l value
    for l in "${ls[@]}"
    do
        cmd="python 06_searching_tables_3_4.py $dataset $P $l $method"
        echo "  Queuing: $cmd"
        tmux send-keys -t "$session_name" "$cmd" C-m
    done
    
    echo "Started processing P=$P, method=$method in session $session_name"
  done
done

echo "All sessions created. Use 'tmux ls' to view them."
