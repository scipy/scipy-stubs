# type-tests for `sparse/csgraph/_flow.pyi`

from typing import assert_type

import numpy as np

import scipy.sparse as sparse
from scipy.sparse.csgraph import maximum_flow

###

_csr_arr: sparse.csr_array[np.int8, tuple[int, int]]

###

# maximum_flow

_flow = maximum_flow(_csr_arr, 0, 1)
assert_type(_flow.flow_value, np.int_)
assert_type(_flow.flow, sparse.csr_array[np.int32, tuple[int, int]])
