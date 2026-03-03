# type-tests for `linalg/_cythonized_array_utils.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import bandwidth, ishermitian, issymmetric

###

f64_2d: onp.Array2D[np.float64]
c128_2d: onp.Array2D[np.complex128]

###
# bandwidth

assert_type(bandwidth(f64_2d), tuple[int, int])
assert_type(bandwidth(c128_2d), tuple[int, int])

###
# issymmetric

assert_type(issymmetric(f64_2d), bool)
assert_type(issymmetric(f64_2d, atol=1e-6), bool)
assert_type(issymmetric(f64_2d, rtol=1e-6), bool)

###
# ishermitian

assert_type(ishermitian(c128_2d), bool)
assert_type(ishermitian(c128_2d, atol=1e-6), bool)
assert_type(ishermitian(c128_2d, rtol=1e-6), bool)
