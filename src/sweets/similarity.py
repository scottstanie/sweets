"""Module for getting the cosine similarity of a stack of images.

This is a translation of the original MATLAB code by Ke Wang, which can be found here:
https://github.com/UT-Radar-Interferometry-Group/psps/blob/a15d458817fe7d06a6edaa0b3208ea78bc4782e7/src/cpp/similarity.cpp
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def median(v: np.ndarray, counter: int) -> float:
    """
    Calculate the median of the array up to a given counter.

    Parameters:
    -----------
    v : np.ndarray
        The input array.
    counter : int
        The number of elements to consider from the input array.

    Returns:
    --------
    float
        Median of the array up to the counter.
    """
    n = counter // 2
    sorted_v = np.partition(v, n)
    return sorted_v[n]


@jit(nopython=True)
def scan_array(rdmin: int, rdmax: int) -> np.ndarray:
    """
    Scan an array with given minimum and maximum radius values.

    Parameters:
    -----------
    rdmin : int
        Minimum radius value.
    rdmax : int
        Maximum radius value.

    Returns:
    --------
    np.ndarray
        Resulting indices after scanning.
    """
    x, y, p, flag = 0, 0, 0, 0
    visited = np.zeros((rdmax, rdmax), dtype=bool)
    indices = []

    visited[0, 0] = True
    for r in range(1, rdmax):
        x = r
        y = 0
        p = 1 - r
        if r > rdmin:
            indices.extend([[r, 0], [-r, 0], [0, r], [0, -r]])
        visited[r, 0] = True
        visited[0, r] = True
        flag = 0
        while x > y:
            if flag == 0:
                y += 1
                if p <= 0:
                    p += 2 * y + 1
                else:
                    x -= 1
                    p += 2 * y - 2 * x + 1
            else:
                flag -= 1

            if x < y:
                break
            while not visited[x - 1, y]:
                x -= 1
                flag += 1
            visited[x, y] = True
            visited[y, x] = True
            if r > rdmin:
                indices.extend([[x, y], [-x, -y], [x, -y], [-x, y]])
                if x != y:
                    indices.extend([[y, x], [-y, -x], [y, -y], [-y, x]])
            if flag > 0:
                x += 1

    return np.array(indices)


@jit(nopython=True, parallel=True)
def median_similarity(
    stack: np.ndarray, ps: np.ndarray, N: int, rdmin: int, rdmax: int
) -> np.ndarray:
    """
    Compute the median similarity for given parameters.

    Parameters:
    -----------
    stack : np.ndarray
        Input multi-dimensional array.
    ps : np.ndarray
        A 2D mask array.
    N : int
        Maximum number of values to use.
    rdmin : int
        Minimum radius value.
    rdmax : int
        Maximum radius value.

    Returns:
    --------
    np.ndarray
        Resulting median similarity values.
    """
    nifg, nrow, ncol = stack.shape[0], ps.shape[0], ps.shape[1]
    sim_median = np.zeros((nrow, ncol))
    indices = scan_array(rdmin, rdmax)
    nindices = len(indices)

    for r0 in prange(nrow):
        for c0 in prange(ncol):
            if not ps[r0, c0]:
                continue
            counter = 0
            simvec = np.zeros(N)
            for i in range(nindices):
                r, c = r0 + indices[i, 0], c0 + indices[i, 1]
                if 0 <= r < nrow and 0 <= c < ncol and ps[r, c]:
                    sim = np.sum(np.cos(stack[:, r0, c0] - stack[:, r, c]))
                    simvec[counter] = sim / nifg
                    counter += 1
                    if counter >= N:
                        break
            sim_median[r0, c0] = median(simvec, counter)

    return sim_median


@jit(nopython=True)
def max_similarity(
    stack: np.ndarray,
    ps0: np.ndarray,
    sim_th: float,
    N: int,
    rdmin: int,
    rdmax: int,
    maxiter: int,
    nonps_cal_flag: bool,
) -> np.ndarray:
    nifg, nrow, ncol = stack.shape[0], ps0.shape[0], ps0.shape[1]
    sim_max = np.zeros((nrow, ncol))
    ps_prev = np.zeros((nrow, ncol), dtype=np.bool_)
    ps = np.copy(ps0)
    indices = scan_array(rdmin, rdmax)
    nindices = len(indices)
    rd2 = np.full((nrow, ncol), rdmax * rdmax, dtype=np.int32)

    for it in range(maxiter):
        print(f"{nrow} {ncol}")
        print(f"calculating max similarity iterate {it}")

        if it > 0:
            for i in prange(nrow):
                for j in prange(ncol):
                    ps_prev[i, j] = ps[i, j]
                    ps[i, j] = sim_max[i, j] >= sim_th
        else:
            for r0 in prange(nrow):
                for c0 in prange(ncol):
                    if (
                        ps[r0, c0]
                        and not nonps_cal_flag
                        or not ps[r0, c0]
                        and nonps_cal_flag
                    ):
                        continue

                    npts = 0
                    for i in range(nindices):
                        r, c = r0 + indices[i, 0], c0 + indices[i, 1]
                        if 0 <= r < nrow and 0 <= c < ncol:
                            if not ps[r, c] and not nonps_cal_flag:
                                continue
                            npts += 1
                            sim = (
                                np.sum(np.cos(stack[:, r0, c0] - stack[:, r, c])) / nifg
                            )
                            sim_max[r0, c0] = max(sim, sim_max[r0, c0])
                            if npts >= N:
                                break

        if nonps_cal_flag:
            break

        newps_counter = 0
        for r0 in prange(nrow):
            for c0 in prange(ncol):
                if ps_prev[r0, c0] or not ps[r0, c0]:
                    continue

                newps_counter += 1
                for i in range(nindices):
                    r, c = r0 + indices[i, 0], c0 + indices[i, 1]
                    if 0 <= r < nrow and 0 <= c < ncol:
                        sim = np.sum(np.cos(stack[:, r0, c0] - stack[:, r, c])) / nifg
                        sim_max[r, c] = max(sim, sim_max[r, c])

        if newps_counter == 0:
            break

    return sim_max
