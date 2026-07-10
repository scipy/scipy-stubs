# type-tests for `sparse/csgraph/_reordering.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

import scipy.sparse as sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee, structural_rank

###

_csr_arr: sparse.csr_array[np.float64, tuple[int, int]]
_coo_arr: sparse.coo_array[np.float64, tuple[int, int]]

###

# reverse_cuthill_mckee

assert_type(reverse_cuthill_mckee(_csr_arr), onp.Array1D[np.int32])
assert_type(reverse_cuthill_mckee(_csr_arr, symmetric_mode=True), onp.Array1D[np.int32])

# structural_rank

assert_type(structural_rank(_coo_arr), np.intp)
