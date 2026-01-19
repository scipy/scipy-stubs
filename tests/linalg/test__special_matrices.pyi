from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import hankel, toeplitz

###

_py_b_1d: list[bool]
_py_b_2d: list[list[bool]]
_py_b_3d: list[list[list[bool]]]

_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_i_3d: list[list[list[int]]]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]

_py_c_1d: list[complex]
_py_c_2d: list[list[complex]]
_py_c_3d: list[list[list[complex]]]

_i8_1d: onp.Array1D[np.int8]
_i8_2d: onp.Array2D[np.int8]
_i8_3d: onp.Array3D[np.int8]
_i8_nd: onp.ArrayND[np.int8]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_3d: onp.Array3D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###

# toeplitz

assert_type(toeplitz(_py_b_1d), onp.Array2D[np.bool_])
assert_type(toeplitz(_py_b_2d), onp.Array3D[np.bool_])
assert_type(toeplitz(_py_b_3d), onp.ArrayND[np.bool_])

assert_type(toeplitz(_py_i_1d), onp.Array2D[np.int_])
assert_type(toeplitz(_py_i_2d), onp.Array3D[np.int_])
assert_type(toeplitz(_py_i_3d), onp.ArrayND[np.int_])

assert_type(toeplitz(_py_f_1d), onp.Array2D[np.float64])
assert_type(toeplitz(_py_f_2d), onp.Array3D[np.float64])
assert_type(toeplitz(_py_f_3d), onp.ArrayND[np.float64])

assert_type(toeplitz(_py_c_1d), onp.Array2D[np.complex128])
assert_type(toeplitz(_py_c_2d), onp.Array3D[np.complex128])
assert_type(toeplitz(_py_c_3d), onp.ArrayND[np.complex128])

assert_type(toeplitz(_i8_1d), onp.Array2D[np.int8])
assert_type(toeplitz(_i8_2d), onp.Array3D[np.int8])
assert_type(toeplitz(_i8_3d), onp.ArrayND[np.int8])
assert_type(toeplitz(_i8_nd), onp.ArrayND[np.int8])

assert_type(toeplitz(_f32_1d), onp.Array2D[np.float32])
assert_type(toeplitz(_f32_2d), onp.Array3D[np.float32])
assert_type(toeplitz(_f32_3d), onp.ArrayND[np.float32])
assert_type(toeplitz(_f32_nd), onp.ArrayND[np.float32])

assert_type(toeplitz(_f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(toeplitz(_f64_2d, _f64_2d), onp.Array3D[np.float64])
assert_type(toeplitz(_f64_3d, _f64_3d), onp.ArrayND[np.float64])
assert_type(toeplitz(_f64_nd, _f64_nd), onp.ArrayND[np.float64])

# hankel

assert_type(hankel(_py_i_1d), onp.Array2D[np.int_])
assert_type(hankel(_py_f_1d), onp.Array2D[np.float64])
assert_type(hankel(_py_c_1d), onp.Array2D[np.complex128])
assert_type(hankel(_f64_1d), onp.Array2D[np.float64])
assert_type(hankel(_f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(hankel(_f64_1d, _f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_f64_nd, _f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_py_i_2d), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(hankel(_py_i_3d), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
