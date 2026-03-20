#!/bin/bash

# Configuration
# P_values=(2 3 4)
P_values=(3)
# l_values=(1.25 1.50 1.75 2.00)
l_values=(1.50)
dist_methods=(0 1)
functions_to_run=("psdtw_prime_parallel_bsf" "psdtw_prime_parallel_bsf_lb" "psdtw_prime_parallel_bsf_lb2" "psdtw_prime_parallel_bsf_lb3")
# functions_to_run=("psdtw_prime_parallel_bsf_lb3")

for function_used in "${functions_to_run[@]}"; do
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
done

echo "All experiments completed."
