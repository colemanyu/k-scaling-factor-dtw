#!/bin/bash

# List of datasets to process
datasets=(
    "GunPoint"
)

# Lists of P values
P_targets=(2 3 4)
Ps=(2 3 4)

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    echo "Running searching speedup for dataset: $dataset"
    
    # Iterate over P_target and P
    for P_target in "${P_targets[@]}"; do
        for P in "${Ps[@]}"; do
            echo "Running with P_target=$P_target and P=$P"
            python 06_searching_table_6.py "$dataset" "$P_target" "$P"
        done
    done
    
    echo "Finished $dataset"
    echo "----------------------------------------"
done
