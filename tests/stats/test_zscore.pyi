# type-tests for `zscore` and `gzscore` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import gzscore, zscore

_py_i_1d: list[int]
_py_i_2d: list[list[int]]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]

_bool_1d: onp.Array1D[np.bool_]
_bool_2d: onp.Array2D[np.bool_]

_i16_1d: onp.Array1D[np.int16]
_i16_2d: onp.Array2D[np.int16]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

_c64_1d: onp.Array1D[np.complex64]
_c64_2d: onp.Array2D[np.complex64]

_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]

###

# zscore

assert_type(zscore(_py_i_1d), onp.Array1D[np.float64])
assert_type(zscore(_py_f_1d), onp.Array1D[np.float64])
assert_type(zscore(_bool_1d), onp.Array1D[np.float64])
assert_type(zscore(_i16_1d), onp.Array1D[np.float64])
assert_type(zscore(_f32_1d), onp.Array1D[np.float32])
assert_type(zscore(_f64_1d), onp.Array1D[np.float64])
assert_type(zscore(_c64_1d), onp.Array1D[np.complex64])
assert_type(zscore(_c128_1d), onp.Array1D[np.complex128])

assert_type(zscore(_py_i_2d), onp.Array2D[np.float64])
assert_type(zscore(_py_f_2d), onp.Array2D[np.float64])
assert_type(zscore(_bool_2d), onp.Array2D[np.float64])
assert_type(zscore(_i16_2d), onp.Array2D[np.float64])
assert_type(zscore(_f32_2d), onp.Array2D[np.float32])
assert_type(zscore(_f64_2d), onp.Array2D[np.float64])
assert_type(zscore(_c64_2d), onp.Array2D[np.complex64])
assert_type(zscore(_c128_2d), onp.Array2D[np.complex128])

assert_type(zscore(_py_i_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(_py_f_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(_bool_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(_i16_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(_f32_1d, axis=None), onp.Array1D[np.float32])
assert_type(zscore(_f64_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(_c64_1d, axis=None), onp.Array1D[np.complex64])
assert_type(zscore(_c128_1d, axis=None), onp.Array1D[np.complex128])

assert_type(zscore(_py_i_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(_py_f_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(_bool_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(_i16_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(_f32_2d, axis=None), onp.Array2D[np.float32])
assert_type(zscore(_f64_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(_c64_2d, axis=None), onp.Array2D[np.complex64])
assert_type(zscore(_c128_2d, axis=None), onp.Array2D[np.complex128])

# gzscore

assert_type(gzscore(_py_i_1d), onp.Array1D[np.float64])
assert_type(gzscore(_py_f_1d), onp.Array1D[np.float64])
assert_type(gzscore(_bool_1d), onp.Array1D[np.float64])
assert_type(gzscore(_i16_1d), onp.Array1D[np.float64])
assert_type(gzscore(_f32_1d), onp.Array1D[np.float32])
assert_type(gzscore(_f64_1d), onp.Array1D[np.float64])
assert_type(gzscore(_c64_1d), onp.Array1D[np.complex64])
assert_type(gzscore(_c128_1d), onp.Array1D[np.complex128])

assert_type(gzscore(_py_i_2d), onp.Array2D[np.float64])
assert_type(gzscore(_py_f_2d), onp.Array2D[np.float64])
assert_type(gzscore(_bool_2d), onp.Array2D[np.float64])
assert_type(gzscore(_i16_2d), onp.Array2D[np.float64])
assert_type(gzscore(_f32_2d), onp.Array2D[np.float32])
assert_type(gzscore(_f64_2d), onp.Array2D[np.float64])
assert_type(gzscore(_c64_2d), onp.Array2D[np.complex64])
assert_type(gzscore(_c128_2d), onp.Array2D[np.complex128])

assert_type(gzscore(_py_i_1d, axis=None), onp.Array1D[np.float64])
assert_type(gzscore(_py_f_1d, axis=None), onp.Array1D[np.float64])
assert_type(gzscore(_bool_1d, axis=None), onp.Array1D[np.float64])
assert_type(gzscore(_i16_1d, axis=None), onp.Array1D[np.float64])
assert_type(gzscore(_f32_1d, axis=None), onp.Array1D[np.float32])
assert_type(gzscore(_f64_1d, axis=None), onp.Array1D[np.float64])
assert_type(gzscore(_c64_1d, axis=None), onp.Array1D[np.complex64])
assert_type(gzscore(_c128_1d, axis=None), onp.Array1D[np.complex128])

assert_type(gzscore(_py_i_2d, axis=None), onp.Array2D[np.float64])
assert_type(gzscore(_py_f_2d, axis=None), onp.Array2D[np.float64])
assert_type(gzscore(_bool_2d, axis=None), onp.Array2D[np.float64])
assert_type(gzscore(_i16_2d, axis=None), onp.Array2D[np.float64])
assert_type(gzscore(_f32_2d, axis=None), onp.Array2D[np.float32])
assert_type(gzscore(_f64_2d, axis=None), onp.Array2D[np.float64])
assert_type(gzscore(_c64_2d, axis=None), onp.Array2D[np.complex64])
assert_type(gzscore(_c128_2d, axis=None), onp.Array2D[np.complex128])
