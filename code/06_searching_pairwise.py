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

# SonyAIBORobotSurface1 (Test)
# ECG200 (Train)
# MedicalImages (Train)
# CBF (Train)
# SwedishLeaf (Train)
# Plane (Train)
# PowerCons (Train)
# GunPoint (Train)
# Adiac (Train)
# Epilepsy (Train)

if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
else:
    dataset_name = "SonyAIBORobotSurface1"
print(f"Using dataset: {dataset_name}")
# Target set
P = 3
l = 1.50
data = np.load(
    f"../data_processed/{dataset_name}_P{P}_uniform.npz",
    allow_pickle=True,
)
# Use Train or Test set?
trans_uniform_concatenated = data["X_train_trans_uniform_concatenated"]
# trans_uniform_concatenated = data["X_test_trans_uniform_concatenated"]

# Query set
# P = 3
data = np.load(
    f"../data_processed/{dataset_name}_P{P}_l{l:.2f}_random.npz",
    allow_pickle=True,
)
# Use Train or Test set?
trans_random_concatenated = data["X_train_trans_random_concatenated"]
# trans_random_concatenated = data["X_test_trans_random_concatenated"]

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
np.random.seed(43)
np.random.shuffle(trans_uniform_concatenated)

# Shuffle Transformed Random (Query Set)
np.random.seed(43)
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
# # Searching with distance measures provided in aeon

# %% [markdown]
# ## Precision@k

# %% [markdown]
# Compute $P@k$ for querying $Q \in$ `query_set` using `method_name` on `target_set`

# %%
# https://www.aeon-toolkit.org/en/latest/api_reference/distances.html
# https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.distances.get_distance_function.html
dist_funcs = {
    "squared": get_distance_function("squared"),  # ED
    "dtw": lambda Q, C: get_distance_function("dtw")(Q, C, window=0.1),
    "adtw": lambda Q, C: get_distance_function("adtw")(Q, C, window=0.1),
    "ddtw": lambda Q, C: get_distance_function("ddtw")(Q, C, window=0.1),
    # "erp": lambda Q, C: get_distance_function("erp")(Q, C, window=0.1),
    # "edr": lambda Q, C: get_distance_function("edr")(Q, C, window=0.1),
    # "lcss": lambda Q, C: get_distance_function("lcss")(Q, C, window=0.1),
    # "manhattan": get_distance_function("manhattan"),
    # "minkowski": get_distance_function("minkowski"),
    # "msm": lambda Q, C: get_distance_function("msm")(Q, C, window=0.1),
    # "sbd": get_distance_function("sbd"),
    "shape_dtw": lambda Q, C: get_distance_function("shape_dtw")(Q, C, window=0.1),
    # "twe": lambda Q, C: get_distance_function("twe")(Q, C, window=0.1),
    "wddtw": lambda Q, C: get_distance_function("wddtw")(Q, C, window=0.1),
    "wdtw": lambda Q, C: get_distance_function("wdtw")(Q, C, window=0.1),
}

# %%
for dist_name, dist_func in dist_funcs.items():
    # start = time.time()
    precision_at_1, precision_at_3, precision_at_5, precision_at_7 = 0, 0, 0, 0
    for i in range(0, len(query_set)):
        # Strip index
        query_feat = query_set[i][:-1]
        query_idx = query_set[i][-1]
        
        # Target set has index at -1, features up to -1.
        # Strip target features for distance calculation
        distances = np.array([dist_func(query_feat, x[:-1]) for x in target_set])

        # Find true match index (where target index matches query index)
        # Note: target_set[:, -1] creates a view if contiguous, or copy if not. 
        # Since we just need to search, it's fine.
        true_match_pos = np.where(target_set[:, -1] == query_idx)[0][0]

        precision_at_1 += precision_at_k(distances, true_match_pos, 1)
        precision_at_3 += precision_at_k(distances, true_match_pos, 3)
        precision_at_5 += precision_at_k(distances, true_match_pos, 5)
        precision_at_7 += precision_at_k(distances, true_match_pos, 7)
    print(
        f"{precision_at_1 / len(query_set):.2f} & {precision_at_3 / len(query_set):.2f}",
        end=" & ",
    )
    # end = time.time()
    # elapsed_time = end - start
    # print("Elapsed time for overall distance computation: " + str(elapsed_time))

# %% [markdown]
# # Searching with PSED, PSDTW

# %%
# psdtw_prime_vanilla, psdtw_prime_parallel, psdtw_prime_parallel_bsf, psdtw_prime_parallel_bsf_lb, psdtw_prime_parallel_bsf_lb2
dist_method = 0  # 0 for ED, 1 for DTW, 15 for DTW (self-defined)
function_used = psdtw_prime_vanilla

dist_func_pp = lambda Q, C: function_used(
        Q, C, l=l, r=0.0, P=P, dist_method= dist_method
    )
dist_func_p = lambda Q, C: dist_func_pp(Q, C)

# %%
def dist_func(Q, C):
    dist, _, _ = dist_func_p(Q, C)
    return dist

# %%
# Warmup for numba
dist_func(
    trans_uniform_concatenated[instance_idx][:-1],
    trans_random_concatenated[instance_idx][:-1],
)
start = time.time()
dist_func(
    trans_uniform_concatenated[instance_idx][:-1],
    trans_random_concatenated[instance_idx][:-1],
)
end = time.time()
elapsed_time = end - start
print("Elapsed time for a single distance computation: " + str(elapsed_time))

# %% [markdown]
# ## Precision@k

# %%
all_distances = []
all_count_dist_calls = []
all_cuts = []
start = time.time()
precision_at_1, precision_at_3, precision_at_5, precision_at_7 = 0, 0, 0, 0
for i in range(0, len(query_set)):
    results = [dist_func_p(query_set[i][:-1], x[:-1]) for x in target_set] # Strip index
    dist_arr, count_dist_calls_arr, cuts_arr = zip(*results)
    distances = np.array(dist_arr)

    # store per-iteration results
    all_distances.append(distances)
    all_count_dist_calls.append(count_dist_calls_arr)
    all_cuts.append(cuts_arr)

    # Find true match position
    query_idx = query_set[i][-1]
    target_idxs = target_set[:, -1]
    true_match_pos = np.where(target_idxs == query_idx)[0][0]

    precision_at_1 += precision_at_k(distances, true_match_pos, 1)
    precision_at_3 += precision_at_k(distances, true_match_pos, 3)
    # precision_at_5 += precision_at_k(distances, true_match_pos, 5)
    # precision_at_7 += precision_at_k(distances, true_match_pos, 7)
print(
    f"{precision_at_1 / len(query_set):.2f} & {precision_at_3 / len(query_set):.2f}",
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
total_count_dist_calls_original = total_count_dist_calls

# %%
os.makedirs("../outputs", exist_ok=True)
np.savez(
    f"../outputs/{dataset_name}_P{P}_l{l:.2f}_dist_method{dist_method}_{function_used.__name__}.npz",
    all_distances=np.array(all_distances, dtype=object),
    all_count_dist_calls=np.array(all_count_dist_calls, dtype=object),
    all_cuts=np.array(all_cuts, dtype=object),
    precision_at_1=precision_at_1 / len(query_set),
    precision_at_3=precision_at_3 / len(query_set),
    precision_at_5=precision_at_5 / len(query_set),
    precision_at_7=precision_at_7 / len(query_set),
    elapsed_time=elapsed_time,
)
