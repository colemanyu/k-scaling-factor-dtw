#!/bin/bash

# Configuration
P_values=(2 3 4)
# P_values=(3)
l_values=(1.25 1.50 1.75 2.00)
# l_values=(1.50)
dist_methods=(0 1)
functions_to_run=("psdtw_prime_parallel_bsf" "psdtw_prime_parallel_bsf_lb" "psdtw_prime_parallel_bsf_lb2" "psdtw_prime_parallel_bsf_lb3")
# functions_to_run=("psdtw_prime_parallel_bsf_lb3")

for P in "${P_values[@]}"; do
    for l in "${l_values[@]}"; do
        for dist_method in "${dist_methods[@]}"; do
            for function_used in "${functions_to_run[@]}"; do
                echo "Starting experiments for function: $function_used"
                echo "--------------------------------------------------"
                echo "Running with P=$P, l=$l, dist_method=$dist_method"
                if [ "$function_used" == "psdtw_prime_parallel_bsf_lb3" ]; then
                    python 06_searching_figure_11_lb3.py --P "$P" --l "$l" --function_used "$function_used" --dist_method "$dist_method"
                else
                    python 06_searching_figure_11.py --P "$P" --l "$l" --function_used "$function_used" --dist_method "$dist_method"
                fi
            done
        done
    done
done


echo "All experiments completed."
