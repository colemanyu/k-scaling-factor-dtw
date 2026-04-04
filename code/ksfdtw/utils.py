import numpy as np
import math
from numba import njit

@njit
def nearest_neighbor_interpolation(ts, L):
    """
    Parameters
    ----------
    ts : time series
    L : desired length of output time series
    """
    k = len(ts)
    result = np.empty(L, dtype=ts.dtype)
    for j in range(L):
        idx = int(np.ceil((j + 1) * k / L)) - 1  # 0-based to 1-based to 0-based
        result[j] = ts[idx]
    return result

def precision_at_k(distances, true_index, k):
    # Get the indices of the top-k smallest distances
    top_k_indices = sorted(range(len(distances)), key=lambda x: distances[x])[:k]

    # Check if the true match is among them
    return 1 if true_index in top_k_indices else 0

@njit
def nearest_neighbor_search(query, dataset, r, l, P, dist_method, dist_func):
    """
    query: shape (m,)
    dataset: shape (N, n)  (assuming equal length for simplicity, or list of arrays)
    """
    bsf = np.inf
    best_idx = -1
    total_dist_calls = 0
    
    for k in range(len(dataset)):
        candidate = dataset[k]
        
        # Pass the current BSF into the distance function
        dist, count_dist_calls, cuts = dist_func(query, candidate, r, l, P, dist_method, bsf)
        
        total_dist_calls += count_dist_calls
        
        # If we found a closer match, update BSF
        if dist < bsf:
            bsf = dist
            best_idx = k
            # print(f"New best found at index {k}: {bsf}")
            
    return best_idx, bsf, total_dist_calls

@njit
def construct_sorted_windows(Q, C, r, l, L_gmax):
    m = len(Q)
    n = len(C)
    assert m == n, "m should be equal to n"

    r_int = int(r * L_gmax)
    windows_sorted = []
    for i in range(1, n + 1):
        idx_start = int(max(1, math.ceil(i/l) - r_int)) - 1
        idx_end = int(min(math.floor(i * l) + r_int, m)) - 1
        window = Q[::-1][idx_start:idx_end+1]
        windows_sorted.append(np.sort(window))
    return windows_sorted

@njit
def nearest_neighbor_search_lb3(Q, dataset, r, l, P, dist_method, dist_func):
    """
    query: shape (m,)
    dataset: shape (N, n)  (assuming equal length for simplicity, or list of arrays)
    """
    bsf = np.inf
    best_idx = -1
    total_dist_calls = 0

    m = len(Q)
    n = len(dataset[0])  # Just to get the length for C

    l_root = math.sqrt(l)
    L_Q_gavg = m / P
    # L_Q_gmin = int(math.ceil(L_Q_gavg / l_root))
    L_Q_gmax = int(math.floor(L_Q_gavg * l_root))
    L_C_gavg = n / P
    # L_C_gmin = int(math.ceil(L_C_gavg / l_root))
    L_C_gmax = int(math.floor(L_C_gavg * l_root))
    L_gmax = max(L_Q_gmax, L_C_gmax)

    sorted_windows = construct_sorted_windows(Q, dataset[0], r, l, L_gmax)
    
    for k in range(len(dataset)):
        candidate = dataset[k]
        
        # Pass the current BSF into the distance function
        dist, count_dist_calls, cuts = dist_func(Q, candidate, r, l, P, dist_method, bsf, sorted_windows)
        
        total_dist_calls += count_dist_calls
        
        # If we found a closer match, update BSF
        if dist < bsf:
            bsf = dist
            best_idx = k
            # print(f"New best found at index {k}: {bsf}")
            
    return best_idx, bsf, total_dist_calls