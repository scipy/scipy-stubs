from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import norm

_Real: TypeAlias = np.int32 | np.int64 | np.float64

a: csr_array

assert_type(norm(a), _Real)
assert_type(norm(a, ord="fro"), _Real)
assert_type(norm(a, ord=1), _Real)
assert_type(norm(a, axis=(0, 1)), _Real)
assert_type(norm(a, ord=None, axis=0), onp.Array1D[_Real])
assert_type(norm(a, axis=1), onp.Array1D[_Real])
