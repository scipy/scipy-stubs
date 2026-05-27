from typing import assert_type

import numpy as np
import optype.numpy as onp

import scipy.sparse as sparse
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching

csr_arr: sparse.csr_array[np.float64, tuple[int, int]]

assert_type(maximum_bipartite_matching(csr_arr), onp.Array1D[np.int32 | np.intp])
assert_type(maximum_bipartite_matching(csr_arr, perm_type="column"), onp.Array1D[np.int32 | np.intp])

assert_type(min_weight_full_bipartite_matching(csr_arr), tuple[onp.Array1D[np.int32 | np.intp], onp.Array1D[np.int32 | np.intp]])
