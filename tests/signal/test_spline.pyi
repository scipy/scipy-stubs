# type-tests for `signal/_spline.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import sepfir2d

###

_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]
_inexact_2d: onp.Array2D[np.float64 | np.complex128]

###

# sepfir2d
assert_type(sepfir2d(_f32_2d, _f32_2d, _f32_2d), onp.ArrayND[np.float32 | np.float64])
assert_type(sepfir2d(_f64_2d, _f64_2d, _f64_2d), onp.ArrayND[np.float32 | np.float64])
assert_type(sepfir2d(_c64_2d, _c64_2d, _c64_2d), onp.ArrayND[np.complex64 | np.complex128])
assert_type(sepfir2d(_c128_2d, _c128_2d, _c128_2d), onp.ArrayND[np.complex64 | np.complex128])
assert_type(sepfir2d(_inexact_2d, _inexact_2d, _inexact_2d), onp.ArrayND[np.float32 | np.float64 | np.complex64 | np.complex128])
