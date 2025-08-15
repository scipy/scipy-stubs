# type-tests for `zscore` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import zscore

py_i_1d: list[int]
py_i_2d: list[list[int]]

py_f_1d: list[float]
py_f_2d: list[list[float]]

bool_1d: onp.Array1D[np.bool_]
bool_2d: onp.Array2D[np.bool_]

i16_1d: onp.Array1D[np.int16]
i16_2d: onp.Array2D[np.int16]

f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]

c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]

c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]

###

assert_type(zscore(py_i_1d), onp.Array1D[np.float64])
assert_type(zscore(py_f_1d), onp.Array1D[np.float64])
assert_type(zscore(bool_1d), onp.Array1D[np.float64])
assert_type(zscore(i16_1d), onp.Array1D[np.float64])
assert_type(zscore(f32_1d), onp.Array1D[np.float32])
assert_type(zscore(f64_1d), onp.Array1D[np.float64])
assert_type(zscore(c64_1d), onp.Array1D[np.complex64])
assert_type(zscore(c128_1d), onp.Array1D[np.complex128])

assert_type(zscore(py_i_2d), onp.Array2D[np.float64])
assert_type(zscore(py_f_2d), onp.Array2D[np.float64])
assert_type(zscore(bool_2d), onp.Array2D[np.float64])
assert_type(zscore(i16_2d), onp.Array2D[np.float64])
assert_type(zscore(f32_2d), onp.Array2D[np.float32])
assert_type(zscore(f64_2d), onp.Array2D[np.float64])
assert_type(zscore(c64_2d), onp.Array2D[np.complex64])
assert_type(zscore(c128_2d), onp.Array2D[np.complex128])

assert_type(zscore(py_i_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(py_f_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(bool_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(i16_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(f32_1d, axis=None), onp.Array1D[np.float32])
assert_type(zscore(f64_1d, axis=None), onp.Array1D[np.float64])
assert_type(zscore(c64_1d, axis=None), onp.Array1D[np.complex64])
assert_type(zscore(c128_1d, axis=None), onp.Array1D[np.complex128])

assert_type(zscore(py_i_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(py_f_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(bool_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(i16_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(f32_2d, axis=None), onp.Array2D[np.float32])
assert_type(zscore(f64_2d, axis=None), onp.Array2D[np.float64])
assert_type(zscore(c64_2d, axis=None), onp.Array2D[np.complex64])
assert_type(zscore(c128_2d, axis=None), onp.Array2D[np.complex128])
