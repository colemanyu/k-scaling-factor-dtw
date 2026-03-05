# %% [markdown]
# # Import Libraries

# %%
import numpy as np
import pandas as pd
from numba import njit, prange
import math

import matplotlib.pyplot as plt
import os
import sys
from aeon.utils.numba.general import z_normalise_series_2d
from aeon.distances import get_distance_function
import time

# %%
from ksfdtw.distance_measures import (
    euclidean_distance as ksfdtw_euclidean_distance,
    dtw as ksfdtw_dtw,
    usdtw_prime as usdtw_prime,
    psdtw_prime_vanilla as psdtw_prime_vanilla,
    psdtw_prime_parallel as psdtw_prime_parallel,
    psdtw_prime_parallel_bsf as psdtw_prime_parallel_bsf,
    psdtw_prime_parallel_bsf_lb as psdtw_prime_parallel_bsf_lb,
    psdtw_prime_parallel_bsf_lb2 as psdtw_prime_parallel_bsf_lb2,
    cut_based_distance as cut_based_distance,
)
from ksfdtw.utils import precision_at_k, nearest_neighbor_search
from aeon.distances import (
    euclidean_distance as aeon_euclidean_distance,
    squared_distance as aeon_squared_distance,
    dtw_distance as aeon_dtw_distance,
)

# %% [markdown]
# # Import Dataset

# %%
# The ten datasets used in the experiments:

# SonyAIBORobotSurface1
# ECG200
# MedicalImages
# CBF
# SwedishLeaf
# Plane
# PowerCons
# GunPoint
# Adiac
# Epilepsy

if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
else:
    dataset_name = "SonyAIBORobotSurface1"
# Target set
P = 3
l = 1.50
data = np.load(
    f"../data_processed/{dataset_name}_P{P}_uniform.npz",
    allow_pickle=True,
)
# Use Train or Test set or Both?
trans_uniform_concatenated = data["X_train_trans_uniform_concatenated"]
# trans_uniform_concatenated = data["X_test_trans_uniform_concatenated"]
# trans_uniform_concatenated = np.concatenate((data["X_train_trans_uniform_concatenated"], data["X_test_trans_uniform_concatenated"]), axis=0)

# Query set
# P = 3
data = np.load(
    f"../data_processed/{dataset_name}_P{P}_l{l:.2f}_random.npz",
    allow_pickle=True,
)
# Use Train or Test set or Both?
trans_random_concatenated = data["X_train_trans_random_concatenated"]
# trans_random_concatenated = data["X_test_trans_random_concatenated"]
# trans_random_concatenated = np.concatenate((data["X_train_trans_random_concatenated"], data["X_test_trans_random_concatenated"]), axis=0)


# %% [markdown]
# ## Z-normalise the transformed series

# %%
trans_uniform_concatenated = z_normalise_series_2d(trans_uniform_concatenated)
trans_random_concatenated = z_normalise_series_2d(trans_random_concatenated)

# %%
# Create indices (assuming 0 to N-1)
indices = np.arange(len(trans_uniform_concatenated)).reshape(-1, 1)

# Append indices
trans_uniform_concatenated = np.hstack((trans_uniform_concatenated, indices))
trans_random_concatenated = np.hstack((trans_random_concatenated, indices))

# Shuffle Transformed Uniform (Target Set)
np.random.seed(42)
np.random.shuffle(trans_uniform_concatenated)

# Shuffle Transformed Random (Query Set)
np.random.seed(42)
np.random.shuffle(trans_random_concatenated)

print("Shuffled data with index column appended.")
print("New shape:", trans_uniform_concatenated.shape)

# %% [markdown]
# ## Assign query and target sets

# %%
# Query set
query_set = trans_random_concatenated

# Target set
target_set = trans_uniform_concatenated
if len(query_set) != len(target_set):
    raise ValueError("query_set and target_set have different sizes!")

# %% [markdown]
# ## Plot graph

# %%
instance_idx = 0

# %% [markdown]
# # Searching with Cut-based distance

# %% [markdown]
# ## Import computed cuts

# %%
dist_method = 0  # 0 for ED, 1 for DTW, 15 for DTW (self-defined)
# function_used: psdtw_prime_vanilla, psdtw_prime_parallel
function_used = psdtw_prime_vanilla

# %%
data = np.load(
    f"../outputs_training_set/{dataset_name}_P{P}_l{l:.2f}_dist_method{dist_method}_{function_used.__name__}.npz",
    allow_pickle=True,
)

all_distances = np.ascontiguousarray(data["all_distances"], dtype=np.float64)
all_count_dist_calls = np.ascontiguousarray(
    data["all_count_dist_calls"], dtype=np.float64
)
all_cuts = np.ascontiguousarray(data["all_cuts"], dtype=np.float64)

# %%
# Results of PSD
print(
    f"{data["precision_at_1"]:.2f}",
    f"{data["precision_at_3"]:.2f}",
    end=" & ",
    # f"{data["precision_at_5"]:.2f}",
    # f"{data["precision_at_7"]:.2f}",
)
# print(
#     f"{precision_at_1 / len(query_set):.2f} & {precision_at_3 / len(query_set):.2f}",
#     end=" & ",
# )
print()
print("Elapsed time:", data["elapsed_time"])

total_count_dist_calls = 0
for r in all_count_dist_calls:
    total_count_dist_calls += np.sum(r)
print("Total distance measure calls: " + str(total_count_dist_calls))
original_elapsed_time = data["elapsed_time"]
original_total_count_dist_calls = total_count_dist_calls

# %% [markdown]
# ## Compute Cut-based distances

# %%
# some bug due to shuffling, so we just use the original total count dist calls for now.

# %%
# # 0: aeon_squared_distance, 1: aeon_dtw_distance, 2: aeon_adtw_distance, 3: aeon_ddtw_distance, 4: aeon_erp_distance, 5: aeon_edr_distance
# # 6: aeon_lcss_distance, 7: aeon_manhattan_distance, 8: aeon_minkowski_distance, 9: aeon_msm_distance, 10: aeon_sbd_distance
# # 11: aeon_shape_dtw_distance, 12: aeon_twe_distance, 13: aeon_wddtw_distance, 14: aeon_wdtw_distance
# # for i in range(0, 15):
# for i in [1, 2, 3, 11, 13, 14]:
#     # print("dist_method: " + str(i))
#     dist_method = i
#     precision_at_1, precision_at_3, precision_at_5, precision_at_7 = 0, 0, 0, 0
#     for i in range(0, len(query_set)):
#         query_vec = query_set[i][:-1]
#         query_idx = query_set[i][-1]
#         target_idxs = target_set[:, -1]
#         true_match_pos = np.where(target_idxs == query_idx)[0][0]

#         distances = np.array(
#             [
#                 cut_based_distance(
#                     query_vec,
#                     target_set[j][:-1], # Strip index
#                     0.1,
#                     l,
#                     P,
#                     dist_method=dist_method,
#                     # cuts=all_cuts[i][j],
#                     cuts=all_cuts[i][j],
#                 )
#                 for j in range(0, len(target_set))
#             ]
#         )
#         precision_at_1 += precision_at_k(distances, true_match_pos, 1)
#         precision_at_3 += precision_at_k(distances, true_match_pos, 3)
#         # precision_at_5 += precision_at_k(distances, true_match_pos, 5)
#         # precision_at_7 += precision_at_k(distances, true_match_pos, 7)
#     print(
#         f"{precision_at_1 / len(query_set):.2f} & {precision_at_3 / len(query_set):.2f}",
#         end=" & ",
#     )
#     # print(
#     #     f"{precision_at_1 / len(query_set):.2f}",
#     #     f"{precision_at_3 / len(query_set):.2f}",
#     # )

# %% [markdown]
# # Nearest neighbor search with bsf

# %%
r = 0.0
l = 1.5
P = 3
dist_method = 0
function_used = psdtw_prime_parallel_bsf

# %%
# Warmup for numba
start = time.time()
nearest_neighbor_search(query_set[0][:-1], target_set[:, :-1], r=r, l=l,  P=P, dist_method=dist_method, dist_func=function_used)
end = time.time()
elapsed_time = end - start
print(elapsed_time)

# %%
print("Starting nearest neighbor search over the entire query set...")
print(dataset_name)
all_count_dist_calls = []
start = time.time()
precision_at_1 = 0
for i in range(0, len(query_set)):
    query_vec = query_set[i][:-1]
    target_vecs = target_set[:, :-1]
    best_idx, bsf, total_dist_calls = nearest_neighbor_search(query_vec, target_vecs, r=r, l=l,  P=P, dist_method=dist_method, dist_func=function_used)
    all_count_dist_calls.append(total_dist_calls)
    
    # Check correctness using index
    if target_set[best_idx, -1] == query_set[i, -1]:
        precision_at_1 += 1
    print(i, end=" ")
print()
print(
    f"{precision_at_1 / len(query_set):.2f}",
    end=" & ",
)
end = time.time()
elapsed_time = end - start
print()
print("Elapsed time: " + str(elapsed_time))
print("Average Elapsed time: " + str(elapsed_time / len(query_set)))

total_count_dist_calls = 0
for r in all_count_dist_calls:
    total_count_dist_calls += np.sum(r)
print("Total distance measure calls: " + str(total_count_dist_calls))
new_elapsed_time = elapsed_time
new_total_count_dist_calls = total_count_dist_calls

# %% [markdown]
# # Calculate % distance calls pruned and Speedup

# %%
pruned_dist_calls = original_total_count_dist_calls - new_total_count_dist_calls
pruned_dist_calls_percentage = pruned_dist_calls / original_total_count_dist_calls * 100
print(f"Pruned distance measure calls: {pruned_dist_calls} ({pruned_dist_calls_percentage:.2f}%)")
# speed up
speed_up = original_elapsed_time / new_elapsed_time
print(f"Speed up: {speed_up:.2f}x")

# %% [markdown]
# # End

# %%
import datetime

print(f"This notebook was last run end-to-end on: {datetime.datetime.now()}\n")
###
###
###


