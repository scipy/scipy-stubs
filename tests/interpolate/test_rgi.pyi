from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import RegularGridInterpolator, interpn

# regression test for https://github.com/scipy/scipy-stubs/issues/497
RegularGridInterpolator(np.array([], dtype=np.float64), np.array([], dtype=np.float64))

###
# interpn

_f64_1d: onp.Array1D[np.float64]
_py_c_2d: list[list[complex]]

assert_type(interpn((_f64_1d,), _f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(interpn((_f64_1d, _f64_1d), _py_c_2d, _f64_1d), onp.ArrayND[np.complex128])
