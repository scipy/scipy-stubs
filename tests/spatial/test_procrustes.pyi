# type-tests for `spatial/_procrustes.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial import procrustes

_f64_2d: onp.Array2D[np.float64]

assert_type(procrustes(_f64_2d, _f64_2d)[0], onp.Array2D[np.float64])
assert_type(procrustes(_f64_2d, _f64_2d)[1], onp.Array2D[np.float64])
assert_type(procrustes(_f64_2d, _f64_2d)[2], np.float64)
