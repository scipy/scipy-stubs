# Type tests for scipy.sparse.csgraph
from typing import assert_type

import numpy as np
import optype.numpy as onp

import scipy.sparse as sparse
from ._types import ScalarType, csr_arr
from scipy.sparse.csgraph import breadth_first_order, breadth_first_tree, depth_first_order, depth_first_tree

assert_type(breadth_first_order(csr_arr, 0, return_predecessors=True), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

assert_type(breadth_first_order(csr_arr, 0, True, False), onp.Array1D[np.int32])
assert_type(breadth_first_order(csr_arr, 0), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

assert_type(breadth_first_tree(csr_arr, 0), sparse.csr_array[ScalarType, tuple[int, int]])

assert_type(depth_first_order(csr_arr, 0, return_predecessors=True), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

assert_type(depth_first_order(csr_arr, 0, True, False), onp.Array1D[np.int32])
assert_type(depth_first_order(csr_arr, 0), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])

assert_type(depth_first_tree(csr_arr, 0), sparse.csr_array[ScalarType, tuple[int, int]])
