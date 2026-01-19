from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import gaussian_kde

###

_py_float_1d: list[float]
_i64_1d: onp.Array1D[np.int64]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f80_1d: onp.Array1D[np.float128]

_py_float_2d: list[list[float]]
_i64_2d: onp.Array2D[np.int64]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f80_2d: onp.Array2D[np.float128]

###

assert_type(gaussian_kde(_py_float_1d), gaussian_kde[np.float64])
assert_type(gaussian_kde(_i64_1d), gaussian_kde[np.float64])
assert_type(gaussian_kde(_f16_1d), gaussian_kde[np.float16])
assert_type(gaussian_kde(_f32_1d), gaussian_kde[np.float32])
assert_type(gaussian_kde(_f64_1d), gaussian_kde[np.float64])
assert_type(gaussian_kde(_f80_1d), gaussian_kde[np.float128])

assert_type(gaussian_kde(_py_float_2d), gaussian_kde[np.float64])
assert_type(gaussian_kde(_i64_2d), gaussian_kde[np.float64])
assert_type(gaussian_kde(_f16_2d), gaussian_kde[np.float16])
assert_type(gaussian_kde(_f32_2d), gaussian_kde[np.float32])
assert_type(gaussian_kde(_f64_2d), gaussian_kde[np.float64])
assert_type(gaussian_kde(_f80_2d), gaussian_kde[np.float128])
