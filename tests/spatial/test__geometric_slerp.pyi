# type-tests for `spatial/_geometric_slerp.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.spatial import geometric_slerp

###

_f64: np.float64
_f64_1d: onp.Array1D[np.float64]
_float_list: list[float]

###
# geometric_slerp

assert_type(geometric_slerp(_f64_1d, _f64_1d, t=0.5), onp.Array2D[np.float64])
assert_type(geometric_slerp(_f64_1d, _f64_1d, t=_f64), onp.Array2D[np.float64])
assert_type(geometric_slerp(_f64_1d, _f64_1d, t=_f64_1d), onp.Array2D[np.float64])
assert_type(geometric_slerp(_f64_1d, _f64_1d, t=_float_list), onp.Array2D[np.float64])
