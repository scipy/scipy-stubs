from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import expm
from scipy.sparse import csr_array
from scipy.sparse.linalg import funm_multiply_krylov

def f_c128(A: onp.Array2D[np.complex128]) -> onp.ArrayND[np.complex128]: ...

_sp_f64: csr_array[np.float64]
A_c: onp.Array2D[np.complex128]
b_f: onp.Array1D[np.float64]
b_c: onp.Array1D[np.complex128]

# funm_multiply_krylov
assert_type(funm_multiply_krylov(expm, _sp_f64, b_f), onp.Array1D[np.float64])
assert_type(funm_multiply_krylov(f_c128, A_c, b_c), onp.Array1D[np.complex128])
