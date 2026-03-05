# type-tests for `stats/_quantile.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import quantile

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_py_f_1d: list[float]

###
# quantile

assert_type(quantile(_py_f_1d, 0.5), np.float64)
assert_type(quantile(_f64_1d, 0.5), np.float64)
assert_type(quantile(_f64_nd, 0.5, axis=None), np.float64)
assert_type(quantile(_f64_2d, 0.5, keepdims=True), onp.ArrayND[np.float64])
assert_type(quantile(_f64_1d, 0.5, method="hazen"), np.float64)
