# type-tests for `sparse/csgraph/_flow.pyi`

from typing import Any, assert_type

import numpy as np

import scipy.sparse as sparse
from scipy.sparse.csgraph import maximum_flow

###

_csr_arr: sparse.csr_array[np.int8, tuple[int, int]]
_csr_mat: sparse.csr_matrix[np.int8]

###

# maximum_flow

_flow = maximum_flow(_csr_arr, 0, 1)
assert_type(_flow.flow_value, np.int_)
assert_type(_flow.flow, sparse.csr_array[np.int32, tuple[int, int]] | Any)
assert_type(maximum_flow(_csr_mat, 0, 1).flow_value, np.int_)
