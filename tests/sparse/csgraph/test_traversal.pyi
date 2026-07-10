# type-tests for `sparse/csgraph/_traversal.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

import scipy.sparse as sparse
from scipy.sparse.csgraph import (
    breadth_first_order,
    breadth_first_tree,
    connected_components,
    depth_first_order,
    depth_first_tree,
)

###

type _ScalarType = np.float32
_csr_arr: sparse.csr_array[_ScalarType, tuple[int, int]]

###

# breadth_first_order

assert_type(breadth_first_order(_csr_arr, 0, return_predecessors=True), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(breadth_first_order(_csr_arr, 0, True, False), onp.Array1D[np.int32])
assert_type(breadth_first_order(_csr_arr, 0), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

# breadth_first_tree

assert_type(breadth_first_tree(_csr_arr, 0), sparse.csr_array[_ScalarType, tuple[int, int]])

# depth_first_order

assert_type(depth_first_order(_csr_arr, 0, return_predecessors=True), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(depth_first_order(_csr_arr, 0, True, False), onp.Array1D[np.int32])
assert_type(depth_first_order(_csr_arr, 0), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

# depth_first_tree

assert_type(depth_first_tree(_csr_arr, 0), sparse.csr_array[_ScalarType, tuple[int, int]])

# connected_components

assert_type(connected_components(_csr_arr), tuple[int, onp.Array1D[np.int32]])
assert_type(connected_components(_csr_arr, directed=False), tuple[int, onp.Array1D[np.int32]])
