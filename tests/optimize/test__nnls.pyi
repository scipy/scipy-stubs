from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import nnls

###
# nnls

_a: list[list[float]]
_b: list[float]

assert_type(nnls(_a, _b), tuple[onp.ArrayND[np.float64], np.float64])
assert_type(nnls(_a, _b, maxiter=9), tuple[onp.ArrayND[np.float64], np.float64])
assert_type(nnls(_a, _b, atol=1e-8), tuple[onp.ArrayND[np.float64], np.float64])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
