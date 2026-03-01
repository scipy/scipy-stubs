from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import gcrotmk

a_f: csr_array[np.float64]
a_c: csr_array[np.complex128]
b_f: onp.Array1D[np.float64]
b_c: onp.Array1D[np.complex128]

# gcrotmk
assert_type(gcrotmk(a_f, b_f), tuple[onp.Array1D[np.float64], int])
assert_type(gcrotmk(a_c, b_c), tuple[onp.Array1D[np.complex128], int])
