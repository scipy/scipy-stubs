# type-tests for `mannwhitneyu` from `stats/_mannwhitneyu.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import mannwhitneyu

###

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

# statistic

# 1d
assert_type(mannwhitneyu(_py_i_1d, _py_i_1d).statistic, np.float64)
assert_type(mannwhitneyu(_py_f_1d, _py_f_1d).statistic, np.float64)
assert_type(mannwhitneyu(_i16_1d, _i16_1d).statistic, np.float64)
assert_type(mannwhitneyu(_f32_1d, _f32_1d).statistic, np.float32)
assert_type(mannwhitneyu(_f64_1d, _f64_1d).statistic, np.float64)
assert_type(mannwhitneyu(_py_i_1d, _py_i_1d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_py_f_1d, _py_f_1d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_i16_1d, _i16_1d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_f32_1d, _f32_1d, axis=None).statistic, np.float32)
assert_type(mannwhitneyu(_f64_1d, _f64_1d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_py_i_1d, _py_i_1d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_py_f_1d, _py_f_1d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_i16_1d, _i16_1d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f32_1d, _f32_1d, keepdims=True).statistic, onp.ArrayND[np.float32])
assert_type(mannwhitneyu(_f64_1d, _f64_1d, keepdims=True).statistic, onp.ArrayND[np.float64])

# 2d
assert_type(mannwhitneyu(_py_i_2d, _py_i_2d).statistic, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_py_f_2d, _py_f_2d).statistic, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_i16_2d, _i16_2d).statistic, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_f32_2d, _f32_2d).statistic, onp.Array1D[np.float32])
assert_type(mannwhitneyu(_f64_2d, _f64_2d).statistic, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_py_i_2d, _py_i_2d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_py_f_2d, _py_f_2d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_i16_2d, _i16_2d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_f32_2d, _f32_2d, axis=None).statistic, np.float32)
assert_type(mannwhitneyu(_f64_2d, _f64_2d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_py_i_2d, _py_i_2d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_py_f_2d, _py_f_2d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_i16_2d, _i16_2d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f32_2d, _f32_2d, keepdims=True).statistic, onp.ArrayND[np.float32])
assert_type(mannwhitneyu(_f64_2d, _f64_2d, keepdims=True).statistic, onp.ArrayND[np.float64])

# 3d
assert_type(mannwhitneyu(_py_i_3d, _py_i_3d).statistic, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_py_f_3d, _py_f_3d).statistic, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_i16_3d, _i16_3d).statistic, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_f32_3d, _f32_3d).statistic, onp.Array2D[np.float32])
assert_type(mannwhitneyu(_f64_3d, _f64_3d).statistic, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_py_i_3d, _py_i_3d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_py_f_3d, _py_f_3d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_i16_3d, _i16_3d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_f32_3d, _f32_3d, axis=None).statistic, np.float32)
assert_type(mannwhitneyu(_f64_3d, _f64_3d, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_py_i_3d, _py_i_3d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_py_f_3d, _py_f_3d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_i16_3d, _i16_3d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f32_3d, _f32_3d, keepdims=True).statistic, onp.ArrayND[np.float32])
assert_type(mannwhitneyu(_f64_3d, _f64_3d, keepdims=True).statistic, onp.ArrayND[np.float64])

# nd
assert_type(mannwhitneyu(_i16_nd, _i16_nd).statistic, np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(mannwhitneyu(_f32_nd, _f32_nd).statistic, np.float32 | onp.ArrayND[np.float32])  # pyrefly:ignore[assert-type]
assert_type(mannwhitneyu(_f64_nd, _f64_nd).statistic, np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(mannwhitneyu(_i16_nd, _i16_nd, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_f32_nd, _f32_nd, axis=None).statistic, np.float32)
assert_type(mannwhitneyu(_f64_nd, _f64_nd, axis=None).statistic, np.float64)
assert_type(mannwhitneyu(_i16_nd, _i16_nd, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f32_nd, _f32_nd, keepdims=True).statistic, onp.ArrayND[np.float32])
assert_type(mannwhitneyu(_f64_nd, _f64_nd, keepdims=True).statistic, onp.ArrayND[np.float64])

# pvalue

# 1d
assert_type(mannwhitneyu(_py_i_1d, _py_i_1d).pvalue, np.float64)
assert_type(mannwhitneyu(_py_f_1d, _py_f_1d).pvalue, np.float64)
assert_type(mannwhitneyu(_i16_1d, _i16_1d).pvalue, np.float64)
assert_type(mannwhitneyu(_f32_1d, _f32_1d).pvalue, np.float64)
assert_type(mannwhitneyu(_f64_1d, _f64_1d).pvalue, np.float64)
assert_type(mannwhitneyu(_py_i_1d, _py_i_1d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_py_f_1d, _py_f_1d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_i16_1d, _i16_1d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_f32_1d, _f32_1d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_f64_1d, _f64_1d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_py_i_1d, _py_i_1d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_py_f_1d, _py_f_1d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_i16_1d, _i16_1d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f32_1d, _f32_1d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f64_1d, _f64_1d, keepdims=True).pvalue, onp.ArrayND[np.float64])

# 2d
assert_type(mannwhitneyu(_py_i_2d, _py_i_2d).pvalue, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_py_f_2d, _py_f_2d).pvalue, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_i16_2d, _i16_2d).pvalue, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_f32_2d, _f32_2d).pvalue, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_f64_2d, _f64_2d).pvalue, onp.Array1D[np.float64])
assert_type(mannwhitneyu(_py_i_2d, _py_i_2d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_py_f_2d, _py_f_2d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_i16_2d, _i16_2d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_f32_2d, _f32_2d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_f64_2d, _f64_2d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_py_i_2d, _py_i_2d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_py_f_2d, _py_f_2d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_i16_2d, _i16_2d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f32_2d, _f32_2d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f64_2d, _f64_2d, keepdims=True).pvalue, onp.ArrayND[np.float64])

# 3d
assert_type(mannwhitneyu(_py_i_3d, _py_i_3d).pvalue, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_py_f_3d, _py_f_3d).pvalue, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_i16_3d, _i16_3d).pvalue, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_f32_3d, _f32_3d).pvalue, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_f64_3d, _f64_3d).pvalue, onp.Array2D[np.float64])
assert_type(mannwhitneyu(_py_i_3d, _py_i_3d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_py_f_3d, _py_f_3d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_i16_3d, _i16_3d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_f32_3d, _f32_3d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_f64_3d, _f64_3d, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_py_i_3d, _py_i_3d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_py_f_3d, _py_f_3d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_i16_3d, _i16_3d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f32_3d, _f32_3d, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f64_3d, _f64_3d, keepdims=True).pvalue, onp.ArrayND[np.float64])

# nd
assert_type(mannwhitneyu(_i16_nd, _i16_nd).pvalue, np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(mannwhitneyu(_f32_nd, _f32_nd).pvalue, np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(mannwhitneyu(_f64_nd, _f64_nd).pvalue, np.float64 | onp.ArrayND[np.float64])  # pyrefly:ignore[assert-type]
assert_type(mannwhitneyu(_i16_nd, _i16_nd, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_f32_nd, _f32_nd, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_f64_nd, _f64_nd, axis=None).pvalue, np.float64)
assert_type(mannwhitneyu(_i16_nd, _i16_nd, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f32_nd, _f32_nd, keepdims=True).pvalue, onp.ArrayND[np.float64])
assert_type(mannwhitneyu(_f64_nd, _f64_nd, keepdims=True).pvalue, onp.ArrayND[np.float64])
