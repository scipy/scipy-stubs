# type-tests for `stats/_quantile.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.stats import estimated_cdf, quantile

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_3d: onp.Array3D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]

###
# quantile

assert_type(quantile(_py_f_1d, 0.5), np.float64)
assert_type(quantile(_f64_1d, 0.5), np.float64)
assert_type(quantile(_f64_nd, 0.5, axis=None), np.float64)
assert_type(quantile(_f64_2d, 0.5, keepdims=True), onp.ArrayND[np.float64])
assert_type(quantile(_f64_1d, 0.5, method="hazen"), np.float64)

###
# estimated_cdf

assert_type(estimated_cdf(_py_f_1d, 0.5), np.float64)
assert_type(estimated_cdf(_f64_1d, 0.5), np.float64)
assert_type(estimated_cdf(_f32_1d, 0.5), np.float32)
assert_type(estimated_cdf(_py_f_2d, 0.5), onp.Array1D[np.float64])
assert_type(estimated_cdf(_f64_2d, 0.5), onp.Array1D[np.float64])
assert_type(estimated_cdf(_f32_2d, 0.5), onp.Array1D[np.float32])
assert_type(estimated_cdf(_py_f_3d, 0.5), onp.Array2D[np.float64])
assert_type(estimated_cdf(_f64_3d, 0.5), onp.Array2D[np.float64])
assert_type(estimated_cdf(_f32_3d, 0.5), onp.Array2D[np.float32])
assert_subtype[onp.ArrayND[np.float64] | Any](estimated_cdf(_f64_nd, 0.5))
assert_subtype[onp.ArrayND[np.float32] | Any](estimated_cdf(_f32_nd, 0.5))

assert_type(estimated_cdf(_py_f_1d, 0.5, keepdims=True), onp.Array1D[np.float64])
assert_type(estimated_cdf(_f64_1d, 0.5, keepdims=True), onp.Array1D[np.float64])
assert_type(estimated_cdf(_f32_1d, 0.5, keepdims=True), onp.Array1D[np.float32])
assert_type(estimated_cdf(_py_f_2d, 0.5, keepdims=True), onp.Array2D[np.float64])
assert_type(estimated_cdf(_f64_2d, 0.5, keepdims=True), onp.Array2D[np.float64])
assert_type(estimated_cdf(_f32_2d, 0.5, keepdims=True), onp.Array2D[np.float32])
assert_type(estimated_cdf(_py_f_3d, 0.5, keepdims=True), onp.Array3D[np.float64])
assert_type(estimated_cdf(_f64_3d, 0.5, keepdims=True), onp.Array3D[np.float64])
assert_type(estimated_cdf(_f32_3d, 0.5, keepdims=True), onp.Array3D[np.float32])
assert_type(estimated_cdf(_f64_nd, 0.5, keepdims=True), onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(estimated_cdf(_f32_nd, 0.5, keepdims=True), onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]

assert_type(estimated_cdf(_py_f_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f64_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_1d, _py_f_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_py_f_2d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f64_2d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_2d, _py_f_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_py_f_3d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f64_3d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_3d, _py_f_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_f64_nd, _py_f_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_nd, _py_f_1d), onp.ArrayND[np.float32])

assert_type(estimated_cdf(_py_f_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_py_f_2d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f64_2d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_2d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_py_f_3d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f64_3d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_3d, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f64_nd, _f64_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_nd, _f64_1d), onp.ArrayND[np.float64])

assert_type(estimated_cdf(_py_f_1d, _f32_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_f64_1d, _f32_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_1d, _f32_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_py_f_2d, _f32_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_f64_2d, _f32_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_2d, _f32_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_py_f_3d, _f32_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_f64_3d, _f32_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_3d, _f32_1d), onp.ArrayND[np.float32])
assert_type(estimated_cdf(_f64_nd, _f32_1d), onp.ArrayND[np.float64])
assert_type(estimated_cdf(_f32_nd, _f32_1d), onp.ArrayND[np.float32])
