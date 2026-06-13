# type-tests for `sparse/csgraph/_tools.pyi`

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

###

_csr_int: sparse.csr_array[np.int32, tuple[int, int]]
_csr_float: sparse.csr_array[np.float64, tuple[int, int]]

_int_graph: onp.Array2D[np.int32]
_float_graph: onp.Array2D[np.float64]
_masked_int_graph: onp.MArray2D[np.int32]
_masked_float_graph: onp.MArray2D[np.float64]

_pred_int: onp.Array2D[np.int32]
_pred_float: onp.Array2D[np.float64]

###

# csgraph_from_dense

assert_type(csgraph_from_dense(_int_graph), sparse.csr_array[np.int32, tuple[int, int]])
assert_type(csgraph_from_dense(_float_graph), sparse.csr_array[np.float64, tuple[int, int]])

# csgraph_from_masked

assert_type(csgraph_from_masked(_masked_int_graph), sparse.csr_array[np.int32, tuple[int, int]])
assert_type(csgraph_from_masked(_masked_float_graph), sparse.csr_array[np.float64, tuple[int, int]])

# csgraph_masked_from_dense

assert_type(csgraph_masked_from_dense(_int_graph), onp.MArray2D[np.int32])
assert_type(csgraph_masked_from_dense(_float_graph), onp.MArray2D[np.float64])

# csgraph_to_dense

assert_type(csgraph_to_dense(_csr_int), onp.Array2D[np.int32])
assert_type(csgraph_to_dense(_csr_float), onp.Array2D[np.float64])

# csgraph_to_masked

assert_type(csgraph_to_masked(_csr_int), onp.MArray2D[np.int32])
assert_type(csgraph_to_masked(_csr_float), onp.MArray2D[np.float64])

# reconstruct_path

assert_type(reconstruct_path(_int_graph, _pred_int), sparse.csr_array[np.int32, tuple[int, int]])
assert_type(reconstruct_path(_float_graph, _pred_float), sparse.csr_array[np.float64, tuple[int, int]])

# construct_dist_matrix

assert_type(construct_dist_matrix(_int_graph, _pred_int), onp.Array2D[np.int32])
assert_type(construct_dist_matrix(_float_graph, _pred_float), onp.Array2D[np.float64])
