from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import minres

a: csr_array[np.float64]
b: onp.Array1D[np.float64]

# minres
assert_type(minres(a, b), tuple[onp.Array1D[np.float64], int])
