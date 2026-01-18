from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import variation

###

_py_float_1d: list[float]
_py_float_2d: list[list[float]]
_py_float_3d: list[list[list[float]]]

_bool_1d: onp.Array1D[np.bool_]
_i8_1d: onp.Array1D[np.int8]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f80_1d: onp.Array1D[np.float128]

_bool_2d: onp.Array2D[np.bool_]
_i8_2d: onp.Array2D[np.int8]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f80_2d: onp.Array2D[np.float128]

_bool_3d: onp.Array3D[np.bool_]
_i8_3d: onp.Array3D[np.int8]
_f16_3d: onp.Array3D[np.float16]
_f32_3d: onp.Array3D[np.float32]
_f64_3d: onp.Array3D[np.float64]
_f80_3d: onp.Array3D[np.float128]

_bool_nd: onp.ArrayND[np.bool_]
_i8_nd: onp.ArrayND[np.int8]
_f16_nd: onp.ArrayND[np.float16]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]
_f80_nd: onp.ArrayND[np.float128]

###

# 1d

assert_type(variation(_py_float_1d), np.float64)
assert_type(variation(_bool_1d), np.float64)
assert_type(variation(_i8_1d), np.float64)
assert_type(variation(_f16_1d), np.float16)
assert_type(variation(_f32_1d), np.float32)
assert_type(variation(_f64_1d), np.float64)
assert_type(variation(_f80_1d), np.float128)

assert_type(variation(_py_float_1d, axis=None), np.float64)
assert_type(variation(_bool_1d, axis=None), np.float64)
assert_type(variation(_i8_1d, axis=None), np.float64)
assert_type(variation(_f16_1d, axis=None), np.float16)
assert_type(variation(_f32_1d, axis=None), np.float32)
assert_type(variation(_f64_1d, axis=None), np.float64)
assert_type(variation(_f80_1d, axis=None), np.float128)

assert_type(variation(_py_float_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(variation(_bool_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(variation(_i8_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(variation(_f16_1d, keepdims=True), onp.Array1D[np.float16])
assert_type(variation(_f32_1d, keepdims=True), onp.Array1D[np.float32])
assert_type(variation(_f64_1d, keepdims=True), onp.Array1D[np.float64])
assert_type(variation(_f80_1d, keepdims=True), onp.Array1D[np.float128])

# 2d

assert_type(variation(_py_float_2d), onp.Array1D[np.float64])
assert_type(variation(_bool_2d), onp.Array1D[np.float64])
assert_type(variation(_i8_2d), onp.Array1D[np.float64])
assert_type(variation(_f16_2d), onp.Array1D[np.float16])
assert_type(variation(_f32_2d), onp.Array1D[np.float32])
assert_type(variation(_f64_2d), onp.Array1D[np.float64])
assert_type(variation(_f80_2d), onp.Array1D[np.float128])

assert_type(variation(_py_float_2d, axis=None), np.float64)
assert_type(variation(_bool_2d, axis=None), np.float64)
assert_type(variation(_i8_2d, axis=None), np.float64)
assert_type(variation(_f16_2d, axis=None), np.float16)
assert_type(variation(_f32_2d, axis=None), np.float32)
assert_type(variation(_f64_2d, axis=None), np.float64)
assert_type(variation(_f80_2d, axis=None), np.float128)

assert_type(variation(_py_float_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(variation(_bool_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(variation(_i8_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(variation(_f16_2d, keepdims=True), onp.Array2D[np.float16])
assert_type(variation(_f32_2d, keepdims=True), onp.Array2D[np.float32])
assert_type(variation(_f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(variation(_f80_2d, keepdims=True), onp.Array2D[np.float128])

# 3d

assert_type(variation(_py_float_3d), onp.ArrayND[np.float64])
# NOTE: We cannot overload on ">=2d" arrays until we drop support numpy<2.1 (invariant shape type param).
assert_type(variation(_bool_3d), onp.ArrayND[np.float64] | Any)
assert_type(variation(_i8_3d), onp.ArrayND[np.float64] | Any)
assert_type(variation(_f16_3d), onp.ArrayND[np.float16] | Any)
assert_type(variation(_f32_3d), onp.ArrayND[np.float32] | Any)
assert_type(variation(_f64_3d), onp.ArrayND[np.float64] | Any)
assert_type(variation(_f80_3d), onp.ArrayND[np.float128] | Any)

assert_type(variation(_py_float_3d, axis=None), np.float64)
assert_type(variation(_bool_3d, axis=None), np.float64)
assert_type(variation(_i8_3d, axis=None), np.float64)
assert_type(variation(_f16_3d, axis=None), np.float16)
assert_type(variation(_f32_3d, axis=None), np.float32)
assert_type(variation(_f64_3d, axis=None), np.float64)
assert_type(variation(_f80_3d, axis=None), np.float128)

assert_type(variation(_py_float_3d, keepdims=True), onp.ArrayND[np.float64])
assert_type(variation(_bool_3d, keepdims=True), onp.Array3D[np.float64])
assert_type(variation(_i8_3d, keepdims=True), onp.Array3D[np.float64])
assert_type(variation(_f16_3d, keepdims=True), onp.Array3D[np.float16])
assert_type(variation(_f32_3d, keepdims=True), onp.Array3D[np.float32])
assert_type(variation(_f64_3d, keepdims=True), onp.Array3D[np.float64])
assert_type(variation(_f80_3d, keepdims=True), onp.Array3D[np.float128])

# ?d

# NOTE: Pyrefly doesn't seem to be able to intersect the return types in case of multiple matching overloads,
# and in this case both return types are even identical (ArrayND[*] | Any).
assert_type(variation(_bool_nd), onp.ArrayND[np.float64] | Any)  # pyrefly:ignore[assert-type]
assert_type(variation(_i8_nd), onp.ArrayND[np.float64] | Any)  # pyrefly:ignore[assert-type]
assert_type(variation(_f16_nd), onp.ArrayND[np.float16] | Any)  # pyrefly:ignore[assert-type]
assert_type(variation(_f32_nd), onp.ArrayND[np.float32] | Any)  # pyrefly:ignore[assert-type]
assert_type(variation(_f64_nd), onp.ArrayND[np.float64] | Any)  # pyrefly:ignore[assert-type]
assert_type(variation(_f80_nd), onp.ArrayND[np.float128] | Any)  # pyrefly:ignore[assert-type]

assert_type(variation(_bool_nd, axis=None), np.float64)
assert_type(variation(_i8_nd, axis=None), np.float64)
assert_type(variation(_f16_nd, axis=None), np.float16)
assert_type(variation(_f32_nd, axis=None), np.float32)
assert_type(variation(_f64_nd, axis=None), np.float64)
assert_type(variation(_f80_nd, axis=None), np.float128)

assert_type(variation(_bool_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(variation(_i8_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(variation(_f16_nd, keepdims=True), onp.ArrayND[np.float16])
assert_type(variation(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(variation(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(variation(_f80_nd, keepdims=True), onp.ArrayND[np.float128])
