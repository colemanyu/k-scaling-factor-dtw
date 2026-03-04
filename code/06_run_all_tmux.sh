#!/bin/bash

# List of datasets
datasets=(
  "Adiac"
  "CBF"
  "ECG200"
  "Epilepsy"
  "GunPoint"
  "MedicalImages"
  "Plane"
  "PowerCons"
  "SonyAIBORobotSurface1"
  "SwedishLeaf"
)

# Get the absolute path to the current directory (code/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

for dataset in "${datasets[@]}"
do
  # Create a unique session name
  session_name="pairwise_${dataset}"
  
  echo "Creating tmux session: $session_name for dataset: $dataset"
  
  # Create a new detached session
  tmux new-session -d -s "$session_name"
  
  # Send commands to the session
  # 1. Activate the conda environment
  tmux send-keys -t "$session_name" "conda activate ksfdtw" C-m
  
  # 2. Change to the code directory
  tmux send-keys -t "$session_name" "cd $SCRIPT_DIR" C-m
  
  # 3. Run the Python script
  tmux send-keys -t "$session_name" "python 06_searching_pairwise.py $dataset" C-m
  
  echo "Started processing $dataset in session $session_name"
done

echo "All sessions created. Use 'tmux ls' to view them."
