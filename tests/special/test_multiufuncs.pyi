from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import legendre_p

assert_type(legendre_p(1, 1.0), onp.Array1D[np.float64])
assert_type(legendre_p(1, np.float32(1.0)), onp.Array1D[np.float64])
assert_type(legendre_p(1, 1.0, diff_n=True), onp.Array1D[np.float64])
assert_type(legendre_p(1, 1.0, diff_n=1), onp.Array1D[np.float64])
