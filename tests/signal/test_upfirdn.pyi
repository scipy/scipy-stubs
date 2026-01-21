# type-tests for `signal/_upfirdn.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import upfirdn

###

_i64_1d: onp.Array1D[np.int64]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]

###

assert_type(upfirdn(_i64_1d, _i64_1d), onp.ArrayND[np.float64])
assert_type(upfirdn(_f16_1d, _f16_1d), onp.ArrayND[np.float32])
assert_type(upfirdn(_f32_1d, _f32_1d), onp.ArrayND[np.float32])
assert_type(upfirdn(_f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(upfirdn(_c64_1d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(upfirdn(_c128_1d, _c128_1d), onp.ArrayND[np.complex128])
