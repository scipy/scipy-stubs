# type-tests for `sparse/csgraph/_shortest_path.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

import scipy.sparse as sparse
from scipy.sparse.csgraph import bellman_ford, dijkstra, floyd_warshall, johnson, minimum_spanning_tree, shortest_path, yen

###

type _ScalarType = np.float32
_csr_arr: sparse.csr_array[_ScalarType, tuple[int, int]]

###

# bellman_ford

assert_type(bellman_ford(_csr_arr), onp.Array2D[np.float64])
assert_type(bellman_ford(_csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])

# minimum_spanning_tree

assert_type(minimum_spanning_tree(_csr_arr), sparse.csr_array[_ScalarType, tuple[int, int]])

# shortest_path

assert_type(shortest_path(_csr_arr), onp.Array2D[np.float64])
assert_type(shortest_path(_csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])

# floyd_warshall

assert_type(floyd_warshall(_csr_arr), onp.Array2D[np.float64])
assert_type(floyd_warshall(_csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])

# dijkstra

assert_type(dijkstra(_csr_arr), onp.Array2D[np.float64])
assert_type(dijkstra(_csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])
assert_type(dijkstra(_csr_arr, min_only=True), onp.Array1D[np.float64])
assert_type(
    dijkstra(_csr_arr, True, None, True, False, np.inf, min_only=True),
    tuple[onp.Array1D[np.float64], onp.Array1D[np.int32], onp.Array1D[np.int32]],
)

# johnson

assert_type(johnson(_csr_arr), onp.Array2D[np.float64])
assert_type(johnson(_csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])

# yen

assert_type(yen(_csr_arr, 0, 1, 2), onp.Array1D[np.float64])
assert_type(yen(_csr_arr, 0, 1, 2, return_predecessors=True), tuple[onp.Array1D[np.float64], onp.Array2D[np.int32]])
