from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import lgmres

a_f: csr_array[np.float64]
a_c: csr_array[np.complex128]
b_f: onp.Array1D[np.float64]
b_c: onp.Array1D[np.complex128]

# lgmres
assert_type(lgmres(a_f, b_f), tuple[onp.Array1D[np.float64], int])
assert_type(lgmres(a_c, b_c), tuple[onp.Array1D[np.complex128], int])
