#!/bin/bash

# List of datasets to process
datasets=(
    "SonyAIBORobotSurface1"
    "ECG200"
    "MedicalImages"
    "CBF"
    "SwedishLeaf"
    "Plane"
    "PowerCons"
    "GunPoint"
    "Adiac"
    "Epilepsy"
)

# Iterate over each dataset and run the script
for dataset in "${datasets[@]}"; do
    echo "Running searching speedup for dataset: $dataset"
    # python 06_searching_table_7_8_9.py "$dataset"
    python 06_searching_table_10.py "$dataset"
    echo "Finished $dataset"
    echo "----------------------------------------"
done
