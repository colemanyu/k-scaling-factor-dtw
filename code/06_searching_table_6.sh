#!/bin/bash

# List of datasets to process
datasets=(
    "GunPoint"
)

# Lists of P values
P_dataset_s=(2 3 4)
P_given_s=(2 3 4)

# Iterate over each dataset
for dataset in "${datasets[@]}"; do
    echo "Running searching speedup for dataset: $dataset"
    
    # Iterate over P_target and P
    for P_dataset in "${P_dataset_s[@]}"; do
        for P_given in "${P_given_s[@]}"; do
            echo "Running with P_dataset=$P_dataset and P_given=$P_given"
            python 06_searching_table_6.py "$dataset" "$P_dataset" "$P_given"
        done
    done
    
    echo "Finished $dataset"
    echo "----------------------------------------"
done
