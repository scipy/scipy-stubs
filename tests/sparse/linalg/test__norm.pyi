from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.sparse import csr_array
from scipy.sparse.linalg import norm

###

a: csr_array
a_i: csr_array[np.int64]
a_f4: csr_array[np.float32]
a_f8: csr_array[np.float64]
a_c8: csr_array[np.complex64]
a_c16: csr_array[np.complex128]

###

assert_subtype[np.float64](norm(a))
assert_subtype[onp.Array1D[np.float64]](norm(a, axis=0))

assert_type(norm(a_i), np.float64)
assert_type(norm(a_f8), np.float64)
assert_type(norm(a_c16), np.float64)
assert_type(norm(a_f8, ord="fro"), np.float64)
assert_type(norm(a_f8, axis=(0, 1)), np.float64)
assert_type(norm(a_f8, ord=1), np.float64 | Any)
assert_type(norm(a_f8, ord=None, axis=0), onp.Array1D[np.float64])
assert_type(norm(a_f8, axis=1), onp.Array1D[np.float64])
assert_type(norm(a_f8, ord=1, axis=1), onp.Array1D[np.float64 | Any])

assert_type(norm(a_f4), np.float32)
assert_type(norm(a_f4, ord="fro"), np.float32)
assert_type(norm(a_f4, ord=1), np.float64 | Any)
assert_type(norm(a_f4, ord=None, axis=0), onp.Array1D[np.float64])
assert_type(norm(a_f4, axis=1), onp.Array1D[np.float64])

assert_type(norm(a_c8), np.float32)
assert_type(norm(a_c8, ord="fro"), np.float32)
assert_type(norm(a_c8, ord=1), np.float32 | Any)
assert_type(norm(a_c8, ord=None, axis=0), onp.Array1D[np.float32])
assert_type(norm(a_c8, axis=1), onp.Array1D[np.float32])
assert_type(norm(a_c8, ord=1, axis=1), onp.Array1D[np.float32 | Any])
