# Type tests for scipy.sparse.csgraph
from typing import assert_type

import numpy as np
import optype.numpy as onp

import scipy.sparse as sparse
from ._types import ScalarType, csr_arr
from scipy.sparse.csgraph import (
    bellman_ford,
    breadth_first_order,
    breadth_first_tree,
    depth_first_order,
    depth_first_tree,
    dijkstra,
    floyd_warshall,
    minimum_spanning_tree,
    shortest_path,
)

assert_type(breadth_first_order(csr_arr, 0, return_predecessors=True), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

assert_type(breadth_first_order(csr_arr, 0, True, False), onp.Array1D[np.int32])
assert_type(breadth_first_order(csr_arr, 0), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

assert_type(breadth_first_tree(csr_arr, 0), sparse.csr_array[ScalarType, tuple[int, int]])

assert_type(depth_first_order(csr_arr, 0, return_predecessors=True), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

assert_type(depth_first_order(csr_arr, 0, True, False), onp.Array1D[np.int32])
assert_type(depth_first_order(csr_arr, 0), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

assert_type(depth_first_tree(csr_arr, 0), sparse.csr_array[ScalarType, tuple[int, int]])

assert_type(bellman_ford(csr_arr), onp.Array2D[np.float64])
assert_type(bellman_ford(csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])

assert_type(minimum_spanning_tree(csr_arr), sparse.csr_array[ScalarType, tuple[int, int]])

assert_type(shortest_path(csr_arr), onp.Array2D[np.float64])
assert_type(shortest_path(csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])

assert_type(floyd_warshall(csr_arr), onp.Array2D[np.float64])
assert_type(floyd_warshall(csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])

assert_type(dijkstra(csr_arr), onp.Array2D[np.float64])
assert_type(dijkstra(csr_arr, return_predecessors=True), tuple[onp.Array2D[np.float64], onp.Array2D[np.int32]])
assert_type(dijkstra(csr_arr, min_only=True), onp.Array1D[np.float64])
assert_type(
    dijkstra(csr_arr, True, None, True, False, np.inf, min_only=True),
    tuple[onp.Array1D[np.float64], onp.Array1D[np.int32], onp.Array1D[np.int32]],
)
