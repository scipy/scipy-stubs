from typing import assert_type

import numpy as np

import scipy.sparse as sparse
from scipy.sparse.csgraph import maximum_flow

csr_arr: sparse.csr_array[np.float64, tuple[int, int]]

flow = maximum_flow(csr_arr, 0, 1)

assert_type(flow.flow_value, int | np.int32 | np.int64)
assert_type(flow.flow, sparse.csr_array[np.float64, tuple[int, int]])
