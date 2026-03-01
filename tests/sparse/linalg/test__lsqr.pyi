from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import csr_array
from scipy.sparse.linalg import lsqr

a: csr_array[np.float64]
b: onp.Array1D[np.float64]

# lsqr
assert_type(
    lsqr(a, b),
    tuple[
        onp.Array1D[np.float64],
        Literal[0, 1, 2, 3, 4, 5, 6, 7],
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        onp.Array1D[np.float64],
    ],
)
