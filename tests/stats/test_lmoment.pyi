from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import lmoment

###

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_nd: list[list[list[list[float]]]]

_i64_1d: onp.Array1D[np.int64]
_i64_2d: onp.Array2D[np.int64]
_i64_nd: onp.ArrayND[np.int64]

_f16_1d: onp.Array1D[np.float16]
_f16_2d: onp.Array2D[np.float16]
_f16_nd: onp.ArrayND[np.float16]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###

# 1d, 0d

assert_type(lmoment(_py_f_1d, 4), np.float64)
assert_type(lmoment(_i64_1d, 4), np.float64)
assert_type(lmoment(_f64_1d, 4), np.float64)
assert_type(lmoment(_f32_1d, 4), np.float32)
assert_type(lmoment(_f16_1d, 4), np.float32)
assert_type(lmoment(_py_f_1d, 4, axis=None), np.float64)
assert_type(lmoment(_i64_1d, 4, axis=None), np.float64)
assert_type(lmoment(_f64_1d, 4, axis=None), np.float64)
assert_type(lmoment(_f32_1d, 4, axis=None), np.float32)
assert_type(lmoment(_f16_1d, 4, axis=None), np.float32)
assert_type(lmoment(_py_f_1d, 4, keepdims=True), onp.Array1D[np.float64])
assert_type(lmoment(_i64_1d, 4, keepdims=True), onp.Array1D[np.float64])
assert_type(lmoment(_f64_1d, 4, keepdims=True), onp.Array1D[np.float64])
assert_type(lmoment(_f32_1d, 4, keepdims=True), onp.Array1D[np.float32])
assert_type(lmoment(_f16_1d, 4, keepdims=True), onp.Array1D[np.float32])

# 1d, 1d

assert_type(lmoment(_py_f_1d), onp.Array1D[np.float64])
assert_type(lmoment(_i64_1d), onp.Array1D[np.float64])
assert_type(lmoment(_f64_1d), onp.Array1D[np.float64])
assert_type(lmoment(_f32_1d), onp.Array1D[np.float32])
assert_type(lmoment(_f16_1d), onp.Array1D[np.float32])
assert_type(lmoment(_py_f_1d, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_i64_1d, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_f64_1d, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_f32_1d, axis=None), onp.Array1D[np.float32])
assert_type(lmoment(_f16_1d, axis=None), onp.Array1D[np.float32])
assert_type(lmoment(_py_f_1d, keepdims=True), onp.Array2D[np.float64])
assert_type(lmoment(_i64_1d, keepdims=True), onp.Array2D[np.float64])
assert_type(lmoment(_f64_1d, keepdims=True), onp.Array2D[np.float64])
assert_type(lmoment(_f32_1d, keepdims=True), onp.Array2D[np.float32])
assert_type(lmoment(_f16_1d, keepdims=True), onp.Array2D[np.float32])

# 2d, 0d

assert_type(lmoment(_py_f_2d, 4), onp.Array1D[np.float64])
assert_type(lmoment(_i64_2d, 4), onp.Array1D[np.float64])
assert_type(lmoment(_f64_2d, 4), onp.Array1D[np.float64])
assert_type(lmoment(_f32_2d, 4), onp.Array1D[np.float32])
assert_type(lmoment(_f16_2d, 4), onp.Array1D[np.float32])
assert_type(lmoment(_py_f_2d, 4, axis=None), np.float64)
assert_type(lmoment(_i64_2d, 4, axis=None), np.float64)
assert_type(lmoment(_f64_2d, 4, axis=None), np.float64)
assert_type(lmoment(_f32_2d, 4, axis=None), np.float32)
assert_type(lmoment(_f16_2d, 4, axis=None), np.float32)
assert_type(lmoment(_py_f_2d, 4, keepdims=True), onp.Array2D[np.float64])
assert_type(lmoment(_i64_2d, 4, keepdims=True), onp.Array2D[np.float64])
assert_type(lmoment(_f64_2d, 4, keepdims=True), onp.Array2D[np.float64])
assert_type(lmoment(_f32_2d, 4, keepdims=True), onp.Array2D[np.float32])
assert_type(lmoment(_f16_2d, 4, keepdims=True), onp.Array2D[np.float32])

# 2d, 1d

assert_type(lmoment(_py_f_2d), onp.Array2D[np.float64])
assert_type(lmoment(_i64_2d), onp.Array2D[np.float64])
assert_type(lmoment(_f64_2d), onp.Array2D[np.float64])
assert_type(lmoment(_f32_2d), onp.Array2D[np.float32])
assert_type(lmoment(_f16_2d), onp.Array2D[np.float32])
assert_type(lmoment(_py_f_2d, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_i64_2d, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_f64_2d, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_f32_2d, axis=None), onp.Array1D[np.float32])
assert_type(lmoment(_f16_2d, axis=None), onp.Array1D[np.float32])
assert_type(lmoment(_py_f_2d, keepdims=True), onp.Array3D[np.float64])
assert_type(lmoment(_i64_2d, keepdims=True), onp.Array3D[np.float64])
assert_type(lmoment(_f64_2d, keepdims=True), onp.Array3D[np.float64])
assert_type(lmoment(_f32_2d, keepdims=True), onp.Array3D[np.float32])
assert_type(lmoment(_f16_2d, keepdims=True), onp.Array3D[np.float32])

# nd, 0d

assert_type(lmoment(_py_f_nd, 4), onp.ArrayND[np.float64] | np.float64)
# NOTE: Pyrefly doesn't seem to be able to intersect the return types in case of multiple matching overloads,
# and in this case both return types are even identical (viz. `ArrayND[*] | *`).
assert_type(lmoment(_i64_nd, 4), onp.ArrayND[np.float64] | np.float64)  # pyrefly:ignore[assert-type]
assert_type(lmoment(_f64_nd, 4), onp.ArrayND[np.float64] | np.float64)  # pyrefly:ignore[assert-type]
assert_type(lmoment(_f32_nd, 4), onp.ArrayND[np.float32] | np.float32)  # pyrefly:ignore[assert-type]
assert_type(lmoment(_f16_nd, 4), onp.ArrayND[np.float32] | np.float32)  # pyrefly:ignore[assert-type]
assert_type(lmoment(_py_f_nd, 4, axis=None), np.float64)
assert_type(lmoment(_i64_nd, 4, axis=None), np.float64)
assert_type(lmoment(_f64_nd, 4, axis=None), np.float64)
assert_type(lmoment(_f32_nd, 4, axis=None), np.float32)
assert_type(lmoment(_f16_nd, 4, axis=None), np.float32)
assert_type(lmoment(_py_f_nd, 4, keepdims=True), onp.ArrayND[np.float64])
assert_type(lmoment(_i64_nd, 4, keepdims=True), onp.ArrayND[np.float64])
assert_type(lmoment(_f64_nd, 4, keepdims=True), onp.ArrayND[np.float64])
assert_type(lmoment(_f32_nd, 4, keepdims=True), onp.ArrayND[np.float32])
assert_type(lmoment(_f16_nd, 4, keepdims=True), onp.ArrayND[np.float32])

# nd, 1d

assert_type(lmoment(_py_f_nd), onp.ArrayND[np.float64])
assert_type(lmoment(_i64_nd), onp.ArrayND[np.float64])
assert_type(lmoment(_f64_nd), onp.ArrayND[np.float64])
assert_type(lmoment(_f32_nd), onp.ArrayND[np.float32])
assert_type(lmoment(_f16_nd), onp.ArrayND[np.float32])
assert_type(lmoment(_py_f_nd, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_i64_nd, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_f64_nd, axis=None), onp.Array1D[np.float64])
assert_type(lmoment(_f32_nd, axis=None), onp.Array1D[np.float32])
assert_type(lmoment(_f16_nd, axis=None), onp.Array1D[np.float32])
assert_type(lmoment(_py_f_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(lmoment(_i64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(lmoment(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(lmoment(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(lmoment(_f16_nd, keepdims=True), onp.ArrayND[np.float32])
