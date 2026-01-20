# type-tests for `skew` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import kurtosis, skew

_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_i_3d: list[list[list[int]]]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]

_i16_1d: onp.Array1D[np.int16]
_i16_2d: onp.Array2D[np.int16]
_i16_3d: onp.Array3D[np.int16]
_i16_nd: onp.ArrayND[np.int16]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_3d: onp.Array3D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###
# skew

# 1d
assert_type(skew(_py_i_1d), np.float64)
assert_type(skew(_py_f_1d), np.float64)
assert_type(skew(_i16_1d), np.float64)
assert_type(skew(_f32_1d), np.float32)
assert_type(skew(_f64_1d), np.float64)
assert_type(skew(_py_i_1d, axis=None), np.float64)
assert_type(skew(_py_f_1d, axis=None), np.float64)
assert_type(skew(_i16_1d, axis=None), np.float64)
assert_type(skew(_f32_1d, axis=None), np.float32)
assert_type(skew(_f64_1d, axis=None), np.float64)
assert_type(skew(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_i16_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_f32_1d, keepdims=True), onp.ArrayND[np.float32])
assert_type(skew(_f64_1d, keepdims=True), onp.ArrayND[np.float64])

# 2d
assert_type(skew(_py_i_2d), onp.Array1D[np.float64])
assert_type(skew(_py_f_2d), onp.Array1D[np.float64])
assert_type(skew(_i16_2d), onp.Array1D[np.float64])
assert_type(skew(_f32_2d), onp.Array1D[np.float32])
assert_type(skew(_f64_2d), onp.Array1D[np.float64])
assert_type(skew(_py_i_2d, axis=None), np.float64)
assert_type(skew(_py_f_2d, axis=None), np.float64)
assert_type(skew(_i16_2d, axis=None), np.float64)
assert_type(skew(_f32_2d, axis=None), np.float32)
assert_type(skew(_f64_2d, axis=None), np.float64)
assert_type(skew(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_i16_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_f32_2d, keepdims=True), onp.ArrayND[np.float32])
assert_type(skew(_f64_2d, keepdims=True), onp.ArrayND[np.float64])

# 3d
assert_type(skew(_py_i_3d), onp.Array2D[np.float64])
assert_type(skew(_py_f_3d), onp.Array2D[np.float64])
assert_type(skew(_i16_3d), onp.Array2D[np.float64])
assert_type(skew(_f32_3d), onp.Array2D[np.float32])
assert_type(skew(_f64_3d), onp.Array2D[np.float64])
assert_type(skew(_py_i_3d, axis=None), np.float64)
assert_type(skew(_py_f_3d, axis=None), np.float64)
assert_type(skew(_i16_3d, axis=None), np.float64)
assert_type(skew(_f32_3d, axis=None), np.float32)
assert_type(skew(_f64_3d, axis=None), np.float64)
assert_type(skew(_py_i_3d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_py_f_3d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_i16_3d, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_f32_3d, keepdims=True), onp.ArrayND[np.float32])
assert_type(skew(_f64_3d, keepdims=True), onp.ArrayND[np.float64])

# nd
assert_type(skew(_i16_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(skew(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(skew(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(skew(_i16_nd, axis=None), np.float64)
assert_type(skew(_f32_nd, axis=None), np.float32)
assert_type(skew(_f64_nd, axis=None), np.float64)
assert_type(skew(_i16_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(skew(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(skew(_f64_nd, keepdims=True), onp.ArrayND[np.float64])

###
# kurtosis

# 1d
assert_type(kurtosis(_py_i_1d), np.float64)
assert_type(kurtosis(_py_f_1d), np.float64)
assert_type(kurtosis(_i16_1d), np.float64)
assert_type(kurtosis(_f32_1d), np.float32)
assert_type(kurtosis(_f64_1d), np.float64)
assert_type(kurtosis(_py_i_1d, axis=None), np.float64)
assert_type(kurtosis(_py_f_1d, axis=None), np.float64)
assert_type(kurtosis(_i16_1d, axis=None), np.float64)
assert_type(kurtosis(_f32_1d, axis=None), np.float32)
assert_type(kurtosis(_f64_1d, axis=None), np.float64)
assert_type(kurtosis(_py_i_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_py_f_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_i16_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_f32_1d, keepdims=True), onp.ArrayND[np.float32])
assert_type(kurtosis(_f64_1d, keepdims=True), onp.ArrayND[np.float64])

# 2d
assert_type(kurtosis(_py_i_2d), onp.Array1D[np.float64])
assert_type(kurtosis(_py_f_2d), onp.Array1D[np.float64])
assert_type(kurtosis(_i16_2d), onp.Array1D[np.float64])
assert_type(kurtosis(_f32_2d), onp.Array1D[np.float32])
assert_type(kurtosis(_f64_2d), onp.Array1D[np.float64])
assert_type(kurtosis(_py_i_2d, axis=None), np.float64)
assert_type(kurtosis(_py_f_2d, axis=None), np.float64)
assert_type(kurtosis(_i16_2d, axis=None), np.float64)
assert_type(kurtosis(_f32_2d, axis=None), np.float32)
assert_type(kurtosis(_f64_2d, axis=None), np.float64)
assert_type(kurtosis(_py_i_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_i16_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_f32_2d, keepdims=True), onp.ArrayND[np.float32])
assert_type(kurtosis(_f64_2d, keepdims=True), onp.ArrayND[np.float64])

# 3d
assert_type(kurtosis(_py_i_3d), onp.Array2D[np.float64])
assert_type(kurtosis(_py_f_3d), onp.Array2D[np.float64])
assert_type(kurtosis(_i16_3d), onp.Array2D[np.float64])
assert_type(kurtosis(_f32_3d), onp.Array2D[np.float32])
assert_type(kurtosis(_f64_3d), onp.Array2D[np.float64])
assert_type(kurtosis(_py_i_3d, axis=None), np.float64)
assert_type(kurtosis(_py_f_3d, axis=None), np.float64)
assert_type(kurtosis(_i16_3d, axis=None), np.float64)
assert_type(kurtosis(_f32_3d, axis=None), np.float32)
assert_type(kurtosis(_f64_3d, axis=None), np.float64)
assert_type(kurtosis(_py_i_3d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_py_f_3d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_i16_3d, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_f32_3d, keepdims=True), onp.ArrayND[np.float32])
assert_type(kurtosis(_f64_3d, keepdims=True), onp.ArrayND[np.float64])

# nd
assert_type(kurtosis(_i16_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(kurtosis(_f32_nd), np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(kurtosis(_f64_nd), np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(kurtosis(_i16_nd, axis=None), np.float64)
assert_type(kurtosis(_f32_nd, axis=None), np.float32)
assert_type(kurtosis(_f64_nd, axis=None), np.float64)
assert_type(kurtosis(_i16_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(kurtosis(_f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(kurtosis(_f64_nd, keepdims=True), onp.ArrayND[np.float64])
