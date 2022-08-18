# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False


cimport cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.stdlib cimport malloc, free


INT = np.int64
ctypedef np.int64_t INT_t
FLOAT = np.float64
ctypedef np.float64_t FLOAT_t


DEF MIN_DIST = 4
DEF UNPAIR = -1


cdef void build_table(double[:, ::1] probs, double[:, ::1] memo):
    cdef int i, j, k, l, L
    cdef double unpairi, unpairj, pairij, bifurc
    L = probs.shape[0]
    for k in range(1, L):
        for i in range(L - k):
            j = i + k
            if j - i < MIN_DIST:
                continue
            unpairi = memo[i+1, j]
            unpairj = memo[i, j-1]
            pairij = memo[i+1, j-1] + probs[i, j] # canon(seq[i] + seq[j])
            bifurc = 0
            for l in range(i, j):
                bifurc = max(bifurc, memo[i, l] + memo[l+1, j])
            memo[i, j] = max(unpairi, unpairj, pairij, bifurc)


cdef void trace_table(double[:, ::1] memo,
                      double[:, ::1] probs,
                      long[:, ::1] pairs,
                      int i,
                      int j):
    cdef int k
    if i >= j:
        pass
    elif memo[i, j] == memo[i+1, j]:
        trace_table(memo, probs, pairs, i+1, j)
    elif memo[i, j] == memo[i, j-1]:
        trace_table(memo, probs, pairs, i, j-1)
    elif memo[i, j] == memo[i+1, j-1] + probs[i, j]: # canon(seq[i] + seq[j]):
        pairs[i, j] = pairs[j, i] = 1
        trace_table(memo, probs, pairs, i+1, j-1)
    else:
        for k in range(i+1, j-1):
            if memo[i, j] == memo[i, k] + memo[k+1, j]:
                trace_table(memo, probs, pairs, i, k)
                trace_table(memo, probs, pairs, k+1, j)
                break


cpdef np.ndarray[INT_t, ndim=2] nussinov(np.ndarray[FLOAT_t, ndim=2] probs):
    cdef double[:, ::1] memo, probs_view = probs
    cdef long[:, ::1] pairs_view
    cdef int i, j, L

    probs_view = probs
    L = probs.shape[0]
    memo = np.zeros([L, L], dtype=FLOAT)
    pairs = np.zeros([L, L], dtype=INT)
    pairs_view = pairs

    build_table(probs_view, memo)
    trace_table(memo, probs_view, pairs_view, 0, L - 1)

    return pairs
