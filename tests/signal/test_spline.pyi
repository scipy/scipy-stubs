# type-tests for `signal/_spline.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import sepfir2d

###

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c64_2d: onp.Array2D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]

###

# sepfir2d
assert_type(sepfir2d(_f32_2d, _f32_1d, _f32_1d), onp.Array2D[np.float32])
assert_type(sepfir2d(_f64_2d, _f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(sepfir2d(_c64_2d, _c64_1d, _c64_1d), onp.Array2D[np.complex64])
assert_type(sepfir2d(_c128_2d, _c128_1d, _c128_1d), onp.Array2D[np.complex128])
