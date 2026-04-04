import numpy as np
from numba import njit, prange
import math
from .utils import nearest_neighbor_interpolation

from aeon.distances import (
    euclidean_distance as aeon_euclidean_distance,
    dtw_distance as aeon_dtw_distance,
    adtw_distance as aeon_adtw_distance,
    ddtw_distance as aeon_ddtw_distance,
    erp_distance as aeon_erp_distance,
    edr_distance as aeon_edr_distance,
    lcss_distance as aeon_lcss_distance,
    manhattan_distance as aeon_manhattan_distance,
    minkowski_distance as aeon_minkowski_distance,
    msm_distance as aeon_msm_distance,
    sbd_distance as aeon_sbd_distance,
    shape_dtw_distance as aeon_shape_dtw_distance,
    squared_distance as aeon_squared_distance,
    twe_distance as aeon_twe_distance,
    wddtw_distance as aeon_wddtw_distance,
    wdtw_distance as aeon_wdtw_distance,
)

@njit
def euclidean_distance(x, y):
    """
    Calculate Euclidean distance between two sequences using numba.

    Parameters:
    x, y: array-like sequences of the same length

    Returns:
    float: Euclidean distance
    """
    if len(x) != len(y):
        raise ValueError("Sequences must have the same length")

    distance = 0.0
    for i in range(len(x)):
        distance += (x[i] - y[i]) ** 2

    return distance
@njit
def dtw(Q, C, window=None):
    m, n = len(Q), len(C)

    # Convert fractional window to absolute value
    if window is None:
        r = max(m, n)  # No constraint if window is None
    else:
        r = int(window * max(m, n))  # Convert fraction to absolute value

    if abs(n - m) > r:
        raise ValueError("abs(n-m) > r!")

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(max(1, i - r), min(n, i + r) + 1):
            cost = (Q[i - 1] - C[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[m, n]


@njit
def usdtw_prime(Q, C, L, r, dist_method):
    Q_scaled = nearest_neighbor_interpolation(Q, L)
    C_scaled = nearest_neighbor_interpolation(C, L)
    if dist_method == 0:
        return aeon_squared_distance(Q_scaled, C_scaled)
    elif dist_method == 1:
        return aeon_dtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 2:
        return aeon_adtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 3:
        return aeon_ddtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 4:
        return aeon_erp_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 5:
        return aeon_edr_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 6:
        return aeon_lcss_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 7:
        return aeon_manhattan_distance(Q_scaled, C_scaled)
    elif dist_method == 8:
        return aeon_minkowski_distance(Q_scaled, C_scaled)
    elif dist_method == 9:
        return aeon_msm_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 10:
        return aeon_sbd_distance(Q_scaled, C_scaled)
    elif dist_method == 11:
        return aeon_shape_dtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 12:
        return aeon_twe_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 13:
        return aeon_wddtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 14:
        return aeon_wdtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 15:
        return dtw(Q_scaled, C_scaled, window=r)
    else:
        raise ValueError("Invalid distance method!")
        
@njit
def usdtw(Q, C, r, l, L, dist_method=0):
    m, n = len(Q), len(C)
    best_so_far = np.inf
    best_k = -1
    for k in range(math.ceil(m / l), min(math.floor(l * m), n) + 1):
        C_prefix = C[:k]
        dist = usdtw_prime(Q, C_prefix, L, r, dist_method)
        if dist < best_so_far:
            best_so_far = dist
            best_k = k
    return best_so_far, best_k

@njit
def psdtw_prime_vanilla(Q, C, r, l, P, dist_method, bsf=np.inf):
    # print("psdtw_prime_vanilla")
    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    # Only support m == n for now in our test set
    assert m == n, "m should be equal to n"
    
    l_root = math.sqrt(l)
    # g: global
    L_Q_gavg = m / P
    L_Q_gmin = int(math.ceil(L_Q_gavg / l_root))
    L_Q_gmax = int(math.floor(L_Q_gavg * l_root))
    L_C_gavg = n / P
    L_C_gmin = int(math.ceil(L_C_gavg / l_root))
    L_C_gmax = int(math.floor(L_C_gavg * l_root))
    # Used for upsampling 
    L_gmax = max(L_Q_gmax, L_C_gmax)

    # Minimum cost to align the **first i** elements of Q (i.e., Q[:i]) and the **first j** elements of C (i.e., C[:j]) using **P** exactly segments
    # Each segment satisfies the length constraint
    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    for p in range(1, P + 1):
        # p segments in Q take at least (L_Q_gmin * p) points
        for i in range(L_Q_gmin * p, min(L_Q_gmax * p, m) + 1):
            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                i_prime = i - L_Q
                if i_prime < 0:
                    continue
                Q_segment = Q[i_prime:i][::-1] # |Q_segement| = L_Q

                L_C_min = max(L_C_gmin, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_C_gmax)

                # p segments in C take at least (L_C_gmin * p) points
                for j in range(L_C_gmin * p, min(L_C_gmax * p, n) + 1):
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        if j_prime < 0:
                            continue
                        D_cost = D[i_prime, j_prime, p - 1]
                        if np.isinf(D_cost):
                            continue
                        if D_cost > D[i][j][p]:  # D[i][j][p] stores the best_so_far
                            continue
                        C_segment = C[j_prime:j][::-1] # |C_segment| = L_C
                        
                        dist_cost = usdtw_prime(
                            Q_segment,
                            C_segment,
                            r=r,
                            L=L_gmax, # For upsampling
                            dist_method=dist_method,
                        )
                        count_dist_calls += 1

                        # D[i, j, p] = min(D[i, j, p], D_cost + dist_cost)
                        cur_cost = D_cost + dist_cost # cur: current
                        if cur_cost < D[i, j, p]:
                            D[i, j, p] = cur_cost
                            D_cut[i, j, p, 0] = i_prime
                            D_cut[i, j, p, 1] = j_prime
    cuts = np.zeros((P, 4), dtype=np.int64)
    i, j, p = m, n, P
    while p > 0:
        i_prime = D_cut[i, j, p, 0]
        j_prime = D_cut[i, j, p, 1]
        cuts[p - 1, 0] = i_prime
        cuts[p - 1, 1] = i
        cuts[p - 1, 2] = j_prime
        cuts[p - 1, 3] = j
        i, j, p = i_prime, j_prime, p - 1
    return D[m, n, P], count_dist_calls, cuts

@njit(parallel=True)
def psdtw_prime_parallel(Q, C, r, l, P, dist_method, bsf=np.inf):
    # print("psdtw_prime_parallel")
    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    assert m == n, "m should be equal to n"
    
    l_root = math.sqrt(l)
    L_Q_gavg = m / P
    L_Q_gmin = int(math.ceil(L_Q_gavg / l_root))
    L_Q_gmax = int(math.floor(L_Q_gavg * l_root))
    L_C_gavg = n / P
    L_C_gmin = int(math.ceil(L_C_gavg / l_root))
    L_C_gmax = int(math.floor(L_C_gavg * l_root))
    L_gmax = max(L_Q_gmax, L_C_gmax)

    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    # Outer loop 'p' must remain sequential because it depends on p-1
    for p in range(1, P + 1):

        # Parallelize the 'i' loop.
        for i in prange(L_Q_gmin * p, min(L_Q_gmax * p, m) + 1):
            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                i_prime = i - L_Q
                if i_prime < 0:
                    continue
                Q_segment = Q[i_prime:i][::-1] # |Q_segement| = L_Q

                L_C_min = max(L_C_gmin, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_C_gmax)

                for j in range(L_C_gmin * p, min(L_C_gmax * p, n) + 1):
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        if j_prime < 0:
                            continue
                        D_cost = D[i_prime, j_prime, p - 1]
                        if np.isinf(D_cost):
                            continue
                        if D_cost > D[i, j, p]: # D[i][j][p] stores the best_so_far
                            continue
                        C_segment = C[j_prime:j][::-1] # |C_segment| = L_C
                        
                        dist_cost = usdtw_prime(
                            Q_segment,
                            C_segment,
                            r=r,
                            L=L_gmax, # For upsampling
                            dist_method=dist_method,
                        )
                        
                        count_dist_calls += 1
                        
                        # D[i, j, p] = min(D[i, j, p], D_cost + dist_cost)
                        cur_cost = D_cost + dist_cost # cur: current
                        if cur_cost < D[i, j, p]:
                            D[i, j, p] = cur_cost
                            D_cut[i, j, p, 0] = i_prime
                            D_cut[i, j, p, 1] = j_prime

    cuts = np.zeros((P, 4), dtype=np.int64)
    i, j, p = m, n, P
    while p > 0:
        i_prime = D_cut[i, j, p, 0]
        j_prime = D_cut[i, j, p, 1]
        cuts[p - 1, 0] = i_prime
        cuts[p - 1, 1] = i
        cuts[p - 1, 2] = j_prime
        cuts[p - 1, 3] = j
        i, j, p = i_prime, j_prime, p - 1
        
    return D[m, n, P], count_dist_calls, cuts

@njit(parallel=True)
def psdtw_prime_parallel_bsf(Q, C, r, l, P, dist_method, bsf=np.inf):
    # print("psdtw_prime_parallel_bsf")
    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    assert m == n, "m should be equal to n"
    
    l_root = math.sqrt(l)
    L_Q_gavg = m / P
    L_Q_gmin = int(math.ceil(L_Q_gavg / l_root))
    L_Q_gmax = int(math.floor(L_Q_gavg * l_root))
    L_C_gavg = n / P
    L_C_gmin = int(math.ceil(L_C_gavg / l_root))
    L_C_gmax = int(math.floor(L_C_gavg * l_root))
    L_gmax = max(L_Q_gmax, L_C_gmax)

    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    # Outer loop 'p' must remain sequential because it depends on p-1
    for p in range(1, P + 1):
        
        # Parallelize the 'i' loop.
        for i in prange(L_Q_gmin * p, min(L_Q_gmax * p, m) + 1):
            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                i_prime = i - L_Q
                if i_prime < 0:
                    continue
                Q_segment = Q[i_prime:i][::-1] # |Q_segement| = L_Q

                L_C_min = max(L_C_gmin, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_C_gmax)

                for j in range(L_C_gmin * p, min(L_C_gmax * p, n) + 1):
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        if j_prime < 0:
                            continue
                        D_cost = D[i_prime, j_prime, p - 1]
                        if np.isinf(D_cost):
                            continue
                        if D_cost > bsf:
                            continue
                        if D_cost > D[i, j, p]: # D[i][j][p] stores the best_so_far
                            continue
                        C_segment = C[j_prime:j][::-1] # |C_segment| = L_C
                        
                        dist_cost = usdtw_prime(
                            Q_segment,
                            C_segment,
                            r=r,
                            L=L_gmax, # For upsampling
                            dist_method=dist_method,
                        )
                        
                        count_dist_calls += 1
                        
                        cur_cost = D_cost + dist_cost
                        if cur_cost < D[i, j, p]:
                            D[i, j, p] = cur_cost
                            D_cut[i, j, p, 0] = i_prime
                            D_cut[i, j, p, 1] = j_prime

    cuts = np.zeros((P, 4), dtype=np.int64)
    i, j, p = m, n, P
    while p > 0:
        i_prime = D_cut[i, j, p, 0]
        j_prime = D_cut[i, j, p, 1]
        cuts[p - 1, 0] = i_prime
        cuts[p - 1, 1] = i
        cuts[p - 1, 2] = j_prime
        cuts[p - 1, 3] = j
        i, j, p = i_prime, j_prime, p - 1
        
    return D[m, n, P], count_dist_calls, cuts


# # Added compiler directives for maximum speed
# @njit(fastmath=True, boundscheck=False, cache=True)
# def delta(d, window_sorted):
#     """
#     Calculates the squared distance between d and the closest point in window_sorted.
#     Assumes window_sorted is a non-empty, sorted numpy array.
#     """
#     # Fast boundary checks
#     if d <= window_sorted[0]:
#         diff = d - window_sorted[0]
#         return diff * diff
    
#     if d > window_sorted[-1]:
#         diff = d - window_sorted[-1]
#         return diff * diff

#     # Binary search
#     idx = np.searchsorted(window_sorted, d)
    
#     # Mathematical optimization: 
#     # Calculate the positive differences to the left and right neighbors
#     diff1 = window_sorted[idx] - d
#     diff2 = d - window_sorted[idx - 1]
    
#     # Find the smaller difference using a simple ternary (often compiles to a branchless CSEL instruction)
#     best_diff = diff1 if diff1 < diff2 else diff2
    
#     # Square only once
#     return best_diff * best_diff

@njit
def delta(d, window_sorted):
    """
    Parameters
    ----------
    d : float
        Data point.
    window_sorted : array_like, shape (k,)
        Sorted data sequence.

    Returns
    -------
    float
        Squared distance between d and the closest point in window_sorted.
    """
    if window_sorted.size == 0:
        return 0.0  # should not happen if constructed correctly

    # Check the necessity of binary search
    # Fast boundary checks
    if d <= window_sorted[0]:
        diff = d - window_sorted[0]
        return diff * diff
    if d > window_sorted[-1]:
        diff = d - window_sorted[-1]
        return diff * diff

    # Do binary search if needed
    idx = np.searchsorted(window_sorted, d)
    d1 = d - window_sorted[idx]
    d2 = d - window_sorted[idx - 1]
    return min(d1 * d1, d2 * d2)
@njit
def insert_to_sorted(arr, val):
    idx = np.searchsorted(arr, val)
    res = np.empty(len(arr) + 1, dtype=arr.dtype) # res: result
    res[:idx] = arr[:idx]
    res[idx] = val
    res[idx+1:] = arr[idx:]
    return res

@njit(parallel=True)
def psdtw_prime_parallel_bsf_lb(Q, C, r, l, P, dist_method, bsf=np.inf):

    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    assert m == n, "m should be equal to n"
    
    l_root = math.sqrt(l)
    L_Q_gavg = m / P
    L_Q_gmin = int(math.ceil(L_Q_gavg / l_root))
    L_Q_gmax = int(math.floor(L_Q_gavg * l_root))
    L_C_gavg = n / P
    L_C_gmin = int(math.ceil(L_C_gavg / l_root))
    L_C_gmax = int(math.floor(L_C_gavg * l_root))
    L_gmax = max(L_Q_gmax, L_C_gmax)

    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    # Outer loop 'p' must remain sequential because it depends on p-1
    for p in range(1, P + 1):
        
        # Parallelize the 'i' loop.
        for i in prange(L_Q_gmin * p, min(L_Q_gmax * p, m) + 1):
            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                i_prime = i - L_Q
                if i_prime < 0:
                    continue
                Q_segment = Q[i_prime:i][::-1] # |Q_segement| = L_Q

                L_C_min = max(L_C_gmin, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_C_gmax)

                r_int = int(r * max(len(Q_segment), L_C_max))

                # windows_sorted = [np.empty(0, dtype=Q.dtype)]
                # windows_sorted.pop()

                windows_sorted = []

                # for k in range(0, L_C_max):
                #     k_1 = k + 1
                #     idx_start_1 = int(max(1, math.ceil(k_1/l)-r_int))
                #     idx_end_1 = int(min(math.floor(k_1 * l) + r_int, len(Q_segment)))
                #     window = Q_segment[idx_start_1-1:idx_end_1-1+1]
                #     windows_sorted.append(np.sort(window))

                for k in range(1, L_C_max + 1):
                    idx_start = int(max(1, math.ceil(k / l) - r_int)) - 1 # 1-based to 0-based index
                    idx_end = int(min(math.floor(k * l) + r_int, len(Q_segment))) - 1 # 1-based to 0-based index
                    window = Q_segment[idx_start : idx_end + 1] # Inclusive of idx_end
                    windows_sorted.append(np.sort(window))

                for j in range(L_C_gmin * p, min(L_C_gmax * p, n) + 1):
                    lb = 0.0
                    is_lb_initialized = False
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        if j_prime < 0:
                            continue
                        D_cost = D[i_prime, j_prime, p - 1]

                        if np.isinf(D_cost):
                            continue
                        if D_cost > bsf:
                            continue
                        if D_cost > D[i, j, p]: # D[i][j][p] stores the best_so_far
                            continue

                        C_segment = C[j_prime:j][::-1] # |C_segment| = L_C

                        if not is_lb_initialized:
                            for k in range(0, L_C):
                                if k == 0:
                                    lb = (Q_segment[0] - C_segment[0]) ** 2
                                else:
                                    lb += delta(C_segment[k], windows_sorted[k])
                                if lb > D[i, j, p]:
                                    break
                            is_lb_initialized = True
                        else:
                            lb += delta(C_segment[L_C - 1], windows_sorted[L_C - 1])
                        if lb > D[i, j, p]:
                            break
                        # Use lb with the last point to further tighten the bound
                        lb_check = lb - delta(C_segment[L_C - 1], windows_sorted[L_C - 1]) + (Q_segment[-1] - C_segment[-1]) ** 2

                        if D_cost + lb_check > D[i, j, p]:
                            continue    
                        
                        dist_cost = usdtw_prime(
                            Q_segment,
                            C_segment,
                            r=r,
                            L=L_gmax, # For upsampling
                            dist_method=dist_method,
                        )
                        
                        count_dist_calls += 1
                        
                        cur_cost = D_cost + dist_cost
                        if cur_cost < D[i, j, p]:
                            D[i, j, p] = cur_cost
                            D_cut[i, j, p, 0] = i_prime
                            D_cut[i, j, p, 1] = j_prime

    cuts = np.zeros((P, 4), dtype=np.int64)
    i, j, p = m, n, P
    while p > 0:
        i_prime = D_cut[i, j, p, 0]
        j_prime = D_cut[i, j, p, 1]
        cuts[p - 1, 0] = i_prime
        cuts[p - 1, 1] = i
        cuts[p - 1, 2] = j_prime
        cuts[p - 1, 3] = j
        i, j, p = i_prime, j_prime, p - 1
        
    return D[m, n, P], count_dist_calls, cuts

@njit(parallel=True)
def psdtw_prime_parallel_bsf_lb2(Q, C, r, l, P, dist_method, bsf=np.inf):

    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    assert m == n, "m should be equal to n"
    
    l_root = math.sqrt(l)
    L_Q_gavg = m / P
    L_Q_gmin = int(math.ceil(L_Q_gavg / l_root))
    L_Q_gmax = int(math.floor(L_Q_gavg * l_root))
    L_C_gavg = n / P
    L_C_gmin = int(math.ceil(L_C_gavg / l_root))
    L_C_gmax = int(math.floor(L_C_gavg * l_root))
    L_gmax = max(L_Q_gmax, L_C_gmax)

    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    # Outer loop 'p' must remain sequential because it depends on p-1
    for p in range(1, P + 1):
        
        # Parallelize the 'i' loop.
        for i in prange(L_Q_gmin * p, min(L_Q_gmax * p, m) + 1):
            cached_r_int = -1
            windows_sorted = []

            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                i_prime = i - L_Q
                if i_prime < 0:
                    continue
                Q_segment = Q[i_prime:i][::-1] # |Q_segement| = L_Q

                L_C_min = max(L_C_gmin, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_C_gmax)

                r_int = int(r * max(len(Q_segment), L_C_max))

                if cached_r_int == r_int:
                    for k in range(1, L_C_max + 1):
                        if (k - 1) < len(windows_sorted):
                            prev_idx_end = int(min(math.floor(k * l) + r_int, L_Q - 1)) - 1
                            new_idx_end = int(min(math.floor(k * l) + r_int, L_Q)) - 1

                            if new_idx_end > prev_idx_end:
                                windows_sorted[k-1] = insert_to_sorted(
                                    windows_sorted[k-1],
                                    Q_segment[new_idx_end]
                                )
                        else:
                            # Not computed yet, build from scratch and append
                            idx_start = int(max(1, math.ceil(k / l) - r_int)) - 1 # 1-based to 0-based index
                            idx_end = int(min(math.floor(k * l) + r_int, len(Q_segment))) - 1 # 1-based to 0-based index
                            window = Q_segment[idx_start : idx_end + 1] # Inclusive of idx_end
                            windows_sorted.append(np.sort(window))
                else:
                    # Clear and rebuild if r_int changes
                    windows_sorted.clear()
                    for k in range(1, L_C_max + 1):
                        idx_start = int(max(1, math.ceil(k / l) - r_int)) - 1 # 1-based to 0-based index
                        idx_end = int(min(math.floor(k * l) + r_int, len(Q_segment))) - 1 # 1-based to 0-based index
                        window = Q_segment[idx_start : idx_end + 1] # Inclusive of idx_end
                        windows_sorted.append(np.sort(window))
                cached_r_int = r_int

                for j in range(L_C_gmin * p, min(L_C_gmax * p, n) + 1):
                    lb = 0.0
                    is_lb_initialized = False
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        if j_prime < 0:
                            continue
                        D_cost = D[i_prime, j_prime, p - 1]

                        if np.isinf(D_cost):
                            continue
                        if D_cost > bsf:
                            continue
                        if D_cost > D[i, j, p]: # D[i][j][p] stores the best_so_far
                            continue

                        C_segment = C[j_prime:j][::-1] # |C_segment| = L_C
                        
                        if not is_lb_initialized:
                            for k in range(0, L_C):
                                if k == 0:
                                    lb = (Q_segment[0] - C_segment[0]) ** 2
                                else:
                                    lb += delta(C_segment[k], windows_sorted[k])
                                if lb > D[i, j, p]:
                                    break
                            is_lb_initialized = True
                        else:
                            lb += delta(C_segment[L_C - 1], windows_sorted[L_C - 1])
                        if lb > D[i, j, p]:
                            break

                        # Use lb with the last point to further tighten the bound
                        lb_check = lb - delta(C_segment[L_C - 1], windows_sorted[L_C - 1]) + (Q_segment[-1] - C_segment[-1]) ** 2

                        if D_cost + lb_check > D[i, j, p]:
                            continue
                        
                        dist_cost = usdtw_prime(
                            Q_segment,
                            C_segment,
                            r=r,
                            L=L_gmax, # For upsampling
                            dist_method=dist_method,
                        )
                        
                        count_dist_calls += 1
                        
                        cur_cost = D_cost + dist_cost
                        if cur_cost < D[i, j, p]:
                            D[i, j, p] = cur_cost
                            D_cut[i, j, p, 0] = i_prime
                            D_cut[i, j, p, 1] = j_prime

    cuts = np.zeros((P, 4), dtype=np.int64)
    i, j, p = m, n, P
    while p > 0:
        i_prime = D_cut[i, j, p, 0]
        j_prime = D_cut[i, j, p, 1]
        cuts[p - 1, 0] = i_prime
        cuts[p - 1, 1] = i
        cuts[p - 1, 2] = j_prime
        cuts[p - 1, 3] = j
        i, j, p = i_prime, j_prime, p - 1
        
    return D[m, n, P], count_dist_calls, cuts


# @njit(parallel=True)
# def psdtw_prime_parallel_bsf_lb3(Q, C, r, l, P, dist_method, bsf=np.inf):
#     # I am a placeholder for further optimization.
#     return -1

@njit(parallel=True)
def psdtw_prime_parallel_bsf_lb3(Q, C, r, l, P, dist_method, bsf=np.inf, sorted_windows=None):
    
    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    assert m == n, "m should be equal to n"
    # print(len(sorted_windows))
    
    l_root = math.sqrt(l)
    L_Q_gavg = m / P
    L_Q_gmin = int(math.ceil(L_Q_gavg / l_root))
    L_Q_gmax = int(math.floor(L_Q_gavg * l_root))
    L_C_gavg = n / P
    L_C_gmin = int(math.ceil(L_C_gavg / l_root))
    L_C_gmax = int(math.floor(L_C_gavg * l_root))
    L_gmax = max(L_Q_gmax, L_C_gmax)

    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    # Outer loop 'p' must remain sequential because it depends on p-1
    for p in range(1, P + 1):
        
        # Parallelize the 'i' loop.
        for i in prange(L_Q_gmin * p, min(L_Q_gmax * p, m) + 1):
            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                i_prime = i - L_Q
                if i_prime < 0:
                    continue
                Q_segment = Q[i_prime:i][::-1] # |Q_segement| = L_Q

                L_C_min = max(L_C_gmin, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_C_gmax)

                for j in range(L_C_gmin * p, min(L_C_gmax * p, n) + 1):
                    lb = 0.0
                    is_lb_initialized = False
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        if j_prime < 0:
                            continue
                        D_cost = D[i_prime, j_prime, p - 1]
                        if np.isinf(D_cost):
                            continue
                        if D_cost > bsf:
                            continue
                        if D_cost > D[i, j, p]: # D[i][j][p] stores the best_so_far
                            continue
                        C_segment = C[j_prime:j][::-1] # |C_segment| = L_C

                        if not is_lb_initialized:
                            for k in range(0, L_C):
                                if k == 0:
                                    lb = (Q_segment[0] - C_segment[0]) ** 2
                                else:
                                    lb += delta(C_segment[k], sorted_windows[j-k])
                                if lb > D[i, j, p]:
                                    break
                            is_lb_initialized = True
                        else:
                            lb += delta(C_segment[L_C - 1], sorted_windows[j -(L_C - 1)]) # "L_C - 1" = "0 + (L_C - 1)"
                        if lb > D[i, j, p]:
                            break
                        # Use lb with the last point to further tighten the bound
                        lb_check = lb - delta(C_segment[L_C - 1], sorted_windows[j -(L_C - 1)]) + (Q_segment[-1] - C_segment[-1]) ** 2

                        if D_cost + lb_check > D[i, j, p]:
                            continue
                        
                        dist_cost = usdtw_prime(
                            Q_segment,
                            C_segment,
                            r=r,
                            L=L_gmax, # For upsampling
                            dist_method=dist_method,
                        )
                        
                        count_dist_calls += 1
                        
                        cur_cost = D_cost + dist_cost
                        if cur_cost < D[i, j, p]:
                            D[i, j, p] = cur_cost
                            D_cut[i, j, p, 0] = i_prime
                            D_cut[i, j, p, 1] = j_prime

    cuts = np.zeros((P, 4), dtype=np.int64)
    i, j, p = m, n, P
    while p > 0:
        i_prime = D_cut[i, j, p, 0]
        j_prime = D_cut[i, j, p, 1]
        cuts[p - 1, 0] = i_prime
        cuts[p - 1, 1] = i
        cuts[p - 1, 2] = j_prime
        cuts[p - 1, 3] = j
        i, j, p = i_prime, j_prime, p - 1
        
    return D[m, n, P], count_dist_calls, cuts

@njit
def cut_based_distance(Q, C, r, l, P, dist_method, cuts):
    m = len(Q)
    L_avg = m / P
    l_root = math.sqrt(l)
    L_max = min(int(math.floor(L_avg * l_root)), m)
    # print(cuts.shape)
    dist = 0.0
    for cut in cuts:
        # print(cut[0], cut[1], cut[2], cut[3])
        dist_cost = usdtw_prime(
            Q[cut[0] : cut[1]][::-1],
            C[cut[2] : cut[3]][::-1],
            L=L_max,
            r=r,
            dist_method=dist_method,
        )
        # dist_cost = distance_measure_prime(
        #     Q[cut[0] : cut[1]],
        #     C[cut[2] : cut[3]],
        #     r=r,
        #     dist_method=dist_method,
        # )
        dist += dist_cost
    return dist