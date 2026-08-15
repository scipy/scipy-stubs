# type-tests for `linalg/_decomp_polar.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import polar

###
# Input arrays

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

_b_2d: onp.Array2D[np.bool_]
_i8_2d: onp.Array2D[np.int8]
_i16_2d: onp.Array2D[np.int16]
_i32_2d: onp.Array2D[np.int32]
_i64_2d: onp.Array2D[np.int64]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f80_2d: onp.Array2D[np.float128]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]
_c160_2d: onp.Array2D[np.complex256]

_f32_3d: onp.Array3D[np.float32]
_f64_3d: onp.Array3D[np.float64]
_c128_3d: onp.Array3D[np.complex128]

###
# polar

assert_type(polar(_py_i_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(polar(_py_f_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(polar(_py_c_2d), tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128]])
assert_type(polar(_b_2d), tuple[onp.Array2D[np.float32], onp.Array2D[np.float32]])
assert_type(polar(_i8_2d), tuple[onp.Array2D[np.float32], onp.Array2D[np.float32]])
assert_type(polar(_i16_2d), tuple[onp.Array2D[np.float32], onp.Array2D[np.float32]])
assert_type(polar(_f16_2d), tuple[onp.Array2D[np.float32], onp.Array2D[np.float32]])
assert_type(polar(_f32_2d), tuple[onp.Array2D[np.float32], onp.Array2D[np.float32]])
assert_type(polar(_c64_2d), tuple[onp.Array2D[np.complex64], onp.Array2D[np.complex64]])
assert_type(polar(_i32_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(polar(_i64_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(polar(_f64_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(polar(_f80_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(polar(_c128_2d), tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128]])
assert_type(polar(_c160_2d), tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128]])

assert_type(polar(_f32_3d), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(polar(_f64_3d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(polar(_c128_3d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
