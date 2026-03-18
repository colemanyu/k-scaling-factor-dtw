#!/bin/bash

# Configuration
P_values=(2 3 4)
# P_values=(2)
l_values=(1.25 1.50 1.75 2.00)
# l_values=(1.50)
dist_methods=(0 1)
# dist_methods=(1)
# function_used="psdtw_prime_parallel_bsf" # 0.96
function_used="psdtw_prime_parallel_bsf_lb" # 0.94
# function_used="psdtw_prime_parallel_bsf_lb2" # 0.94
# function_used="psdtw_prime_parallel_bsf_lb3" # 0.92

echo "Starting experiments for function: $function_used"

for P in "${P_values[@]}"; do
    for l in "${l_values[@]}"; do
        for dist_method in "${dist_methods[@]}"; do
            echo "--------------------------------------------------"
            echo "Running with P=$P, l=$l, dist_method=$dist_method"
            python 06_searching_figure_11.py --P "$P" --l "$l" --function_used "$function_used" --dist_method "$dist_method"
        done
    done
done

echo "All experiments completed."
