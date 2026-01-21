# type-tests for `signal/_savitzky_golay.pyi`

from _typeshed import Incomplete
from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import savgol_coeffs, savgol_filter

###

_py_i_1d: list[int]
_py_f_1d: list[float]
_i16_1d: onp.Array1D[np.int16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_i16_2d: onp.Array2D[np.int16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]

_py_i_3d: list[list[list[int]]]
_py_f_3d: list[list[list[float]]]
_i16_3d: onp.Array3D[np.int16]
_f32_3d: onp.Array3D[np.float32]
_f64_3d: onp.Array3D[np.float64]

_i16_nd: onp.ArrayND[np.int16]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]

###

# savgol_coeffs

assert_type(savgol_coeffs(5, 3), onp.Array1D[np.float64])
assert_type(savgol_coeffs(5, 3, deriv=2), onp.Array1D[np.float64])
assert_type(savgol_coeffs(5, 3, deriv=2, delta=0.5), onp.Array1D[np.float64])
assert_type(savgol_coeffs(5, 3, deriv=2, delta=0.5, pos=1), onp.Array1D[np.float64])
assert_type(savgol_coeffs(5, 3, xp=np), Incomplete)

# savgol_filter

assert_type(savgol_filter(_py_i_1d, 10, 3), onp.Array1D[np.float64])
assert_type(savgol_filter(_py_f_1d, 10, 3), onp.Array1D[np.float64])
assert_type(savgol_filter(_i16_1d, 10, 3), onp.Array1D[np.float64])
assert_type(savgol_filter(_f32_1d, 10, 3), onp.Array1D[np.float32])
assert_type(savgol_filter(_f64_1d, 10, 3), onp.Array1D[np.float64])

assert_type(savgol_filter(_py_i_2d, 10, 3), onp.Array2D[np.float64])
assert_type(savgol_filter(_py_f_2d, 10, 3), onp.Array2D[np.float64])
assert_type(savgol_filter(_i16_2d, 10, 3), onp.Array2D[np.float64])
assert_type(savgol_filter(_f32_2d, 10, 3), onp.Array2D[np.float32])
assert_type(savgol_filter(_f64_2d, 10, 3), onp.Array2D[np.float64])

assert_type(savgol_filter(_py_i_3d, 10, 3), onp.Array3D[np.float64])
assert_type(savgol_filter(_py_f_3d, 10, 3), onp.Array3D[np.float64])
assert_type(savgol_filter(_i16_3d, 10, 3), onp.Array3D[np.float64])
assert_type(savgol_filter(_f32_3d, 10, 3), onp.Array3D[np.float32])
assert_type(savgol_filter(_f64_3d, 10, 3), onp.Array3D[np.float64])

assert_type(savgol_filter(_i16_nd, 10, 3), onp.ArrayND[np.float64])
assert_type(savgol_filter(_f32_nd, 10, 3), onp.ArrayND[np.float32])
assert_type(savgol_filter(_f64_nd, 10, 3), onp.ArrayND[np.float64])
