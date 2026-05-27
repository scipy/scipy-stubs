from typing import assert_type

import numpy as np
import optype.numpy as onp

import scipy.sparse as sparse
from scipy.sparse.csgraph import (
    construct_dist_matrix,
    csgraph_from_dense,
    csgraph_from_masked,
    csgraph_masked_from_dense,
    csgraph_to_dense,
    csgraph_to_masked,
    reconstruct_path,
)

csr_int: sparse.csr_array[np.int32, tuple[int, int]]
csr_float: sparse.csr_array[np.float64, tuple[int, int]]

int_graph: onp.Array2D[np.int32]
float_graph: onp.Array2D[np.float64]
masked_int_graph: onp.MArray2D[np.int32]
masked_float_graph: onp.MArray2D[np.float64]

pred_int: onp.Array2D[np.int32]
pred_float: onp.Array2D[np.float64]

assert_type(csgraph_from_dense(int_graph), sparse.csr_array[np.int32, tuple[int, int]])
assert_type(csgraph_from_dense(float_graph), sparse.csr_array[np.float64, tuple[int, int]])

assert_type(csgraph_from_masked(masked_int_graph), sparse.csr_array[np.int32, tuple[int, int]])
assert_type(csgraph_from_masked(masked_float_graph), sparse.csr_array[np.float64, tuple[int, int]])

assert_type(csgraph_masked_from_dense(int_graph), onp.MArray2D[np.int32])
assert_type(csgraph_masked_from_dense(float_graph), onp.MArray2D[np.float64])

assert_type(csgraph_to_dense(csr_int), onp.Array2D[np.int32])
assert_type(csgraph_to_dense(csr_float), onp.Array2D[np.float64])

assert_type(csgraph_to_masked(csr_int), onp.MArray2D[np.int32])
assert_type(csgraph_to_masked(csr_float), onp.MArray2D[np.float64])

assert_type(reconstruct_path(int_graph, pred_int), sparse.csr_array[np.int32, tuple[int, int]])
assert_type(reconstruct_path(float_graph, pred_float), sparse.csr_array[np.float64, tuple[int, int]])

assert_type(construct_dist_matrix(int_graph, pred_int), onp.Array2D[np.int32])
assert_type(construct_dist_matrix(float_graph, pred_float), onp.Array2D[np.float64])
