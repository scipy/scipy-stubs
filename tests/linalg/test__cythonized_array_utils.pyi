# type-tests for `linalg/_cythonized_array_utils.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.linalg import ishermitian, issymmetric

###

f64_2d: onp.Array2D[np.float64]
c128_2d: onp.Array2D[np.complex128]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###
# issymmetric

assert_type(issymmetric(f64_2d), bool)
assert_type(issymmetric(f64_2d, atol=1e-6), bool)
assert_type(issymmetric(f64_2d, rtol=1e-6), bool)
assert_type(issymmetric(_f64_3d), onp.ArrayND[np.bool])
assert_subtype[bool | Any](issymmetric(_f64_nd))

###
# ishermitian

assert_type(ishermitian(c128_2d), bool)
assert_type(ishermitian(c128_2d, atol=1e-6), bool)
assert_type(ishermitian(c128_2d, rtol=1e-6), bool)
assert_type(ishermitian(_f64_3d), onp.ArrayND[np.bool])
