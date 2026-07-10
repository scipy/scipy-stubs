from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import minres

###

_csr_i64: csr_array[np.int64]
_csr_f32: csr_array[np.float32]
_csr_f64: csr_array[np.float64]
_csr_c64: csr_array[np.complex64]
_csr_c128: csr_array[np.complex128]

_arr_i64_1d: onp.Array1D[np.int64]
_arr_f32_1d: onp.Array1D[np.float32]
_arr_f64_1d: onp.Array1D[np.float64]

###
# minres

assert_type(minres(_csr_i64, _arr_i64_1d), tuple[onp.Array1D[np.float64], int])
assert_type(minres(_csr_f32, _arr_f32_1d), tuple[onp.Array1D[np.float32], int])
assert_type(minres(_csr_f64, _arr_f64_1d), tuple[onp.Array1D[np.float64], int])
assert_type(minres(_csr_c64, _arr_f32_1d), tuple[onp.Array1D[np.complex64], int])
assert_type(minres(_csr_c128, _arr_f64_1d), tuple[onp.Array1D[np.complex128], int])
