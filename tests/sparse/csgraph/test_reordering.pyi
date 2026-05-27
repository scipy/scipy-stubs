from typing import assert_type

import numpy as np
import optype.numpy as onp

import scipy.sparse as sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee, structural_rank

csr_arr: sparse.csr_array[np.float64, tuple[int, int]]
coo_arr: sparse.coo_array[np.float64, tuple[int, int]]

assert_type(reverse_cuthill_mckee(csr_arr), onp.Array1D[np.int32])
assert_type(reverse_cuthill_mckee(csr_arr, symmetric_mode=True), onp.Array1D[np.int32])

assert_type(structural_rank(coo_arr), np.intp)
