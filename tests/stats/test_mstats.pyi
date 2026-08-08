# type-tests for `stats/mstats.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats.mstats import normaltest, spearmanr, tmax, tmean, tmin

###

_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_c_1d: list[complex]
_py_c_2d: list[list[complex]]

_i64_1d: onp.Array1D[np.int64]
_i64_2d: onp.Array2D[np.int64]
_i64_nd: onp.ArrayND[np.int64]

_f16_1d: onp.Array1D[np.float16]
_f16_2d: onp.Array2D[np.float16]
_f16_nd: onp.ArrayND[np.float16]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_3d: onp.Array3D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_c64_1d: onp.Array1D[np.complex64]
_c64_2d: onp.Array2D[np.complex64]
_c64_nd: onp.ArrayND[np.complex64]

_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]
_c128_nd: onp.ArrayND[np.complex128]

_m_f32_nd: onp.MArray[np.float32]
_m_f64_nd: onp.MArray[np.float64]

###
# tmean

assert_type(tmean(_py_i_1d), np.float64)
assert_type(tmean(_py_f_1d), np.float64)
assert_type(tmean(_py_c_1d), np.complex128)
assert_type(tmean(_py_i_2d), np.float64)
assert_type(tmean(_py_f_2d), np.float64)
assert_type(tmean(_py_c_2d), np.complex128)
assert_type(tmean(_i64_1d), np.float64)
assert_type(tmean(_i64_2d), np.float64)
assert_type(tmean(_i64_nd), np.float64)
assert_type(tmean(_f16_1d), np.float16)
assert_type(tmean(_f16_nd), np.float16)
assert_type(tmean(_f32_1d), np.float64)
assert_type(tmean(_f32_2d), np.float64)
assert_type(tmean(_f32_nd), np.float64)
assert_type(tmean(_f64_1d), np.float64)
assert_type(tmean(_f64_nd), np.float64)
assert_type(tmean(_c64_1d), np.complex128)
assert_type(tmean(_c64_nd), np.complex128)
assert_type(tmean(_c128_1d), np.complex128)
assert_type(tmean(_c128_nd), np.complex128)
assert_type(tmean(_m_f32_nd), np.float64)
assert_type(tmean(_m_f64_nd), np.float64)

assert_type(tmean(_py_i_1d, axis=0), np.float64)
assert_type(tmean(_py_f_1d, axis=0), np.float64)
assert_type(tmean(_py_c_1d, axis=0), np.complex128)
assert_type(tmean(_i64_1d, axis=0), np.float64)
assert_type(tmean(_f16_1d, axis=0), np.float16)
assert_type(tmean(_f32_1d, axis=0), np.float64)
assert_type(tmean(_f64_1d, axis=0), np.float64)
assert_type(tmean(_c64_1d, axis=0), np.complex128)
assert_type(tmean(_c128_1d, axis=0), np.complex128)

assert_type(tmean(_py_i_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_py_f_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_py_c_2d, axis=0), onp.MArray1D[np.complex128])
assert_type(tmean(_i64_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_f16_2d, axis=0), onp.MArray1D[np.float16])
assert_type(tmean(_f32_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_f64_2d, axis=0), onp.MArray1D[np.float64])
assert_type(tmean(_c64_2d, axis=0), onp.MArray1D[np.complex128])
assert_type(tmean(_c128_2d, axis=0), onp.MArray1D[np.complex128])

assert_type(tmean(_i64_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_f16_nd, axis=0), onp.MArray[np.float16] | Any)
assert_type(tmean(_f32_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_f64_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_c64_nd, axis=0), onp.MArray[np.complex128] | Any)
assert_type(tmean(_c128_nd, axis=0), onp.MArray[np.complex128] | Any)
assert_type(tmean(_m_f32_nd, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_m_f64_nd, axis=0), onp.MArray[np.float64] | Any)

assert_type(tmean(_f32_3d, axis=0), onp.MArray[np.float64] | Any)
assert_type(tmean(_f64_nd, (0.0, 1.0), (True, True), 0), onp.MArray[np.float64] | Any)

###
# tmin

assert_type(tmin(_py_i_1d), np.int_ | onp.MArray[np.int_])
assert_type(tmin(_f32_1d), np.float32 | onp.MArray[np.float32])

###
# tmax

assert_type(tmax(_py_i_1d), np.int_ | onp.MArray[np.int_])
assert_type(tmax(_f32_1d), np.float32 | onp.MArray[np.float32])

###
# spearmanr

assert_type(spearmanr(_py_i_1d, _py_i_1d).statistic, np.float64)
assert_type(spearmanr(_f64_nd, _f64_nd).statistic, np.float64)
assert_type(spearmanr(_f32_3d, _f32_3d, axis=None).statistic, np.float64)
assert_type(spearmanr(_py_f_1d, _py_f_1d, axis=0).statistic, np.float64)
assert_type(spearmanr(_i64_1d, _i64_1d, axis=1).statistic, np.float64)
assert_type(spearmanr(_py_i_2d, _py_i_2d, axis=0).statistic, onp.Array2D[np.float64])
assert_type(spearmanr(_f64_2d, _f64_2d, axis=1).statistic, onp.Array2D[np.float64])
assert_type(spearmanr(_m_f64_nd, _m_f64_nd, axis=0).statistic, onp.Array2D[np.float64] | Any)
assert_type(spearmanr(_f32_3d, _f32_3d, axis=0).statistic, onp.Array2D[np.float64] | Any)
assert_type(spearmanr(_f64_2d, axis=1).statistic, onp.Array2D[np.float64] | Any)

###
# normaltest

assert_type(normaltest(_f64_nd, axis=None).statistic, np.float64)
assert_type(normaltest(_py_i_1d).statistic, np.float64)
assert_type(normaltest(_f64_1d).pvalue, np.float64)
assert_type(normaltest(_i64_2d).statistic, onp.MArray1D[np.float64])
assert_type(normaltest(_f32_2d, axis=1).pvalue, onp.Array1D[np.float64])
assert_type(normaltest(_f32_3d).statistic, onp.MArray2D[np.float64])
assert_type(normaltest(_f32_nd).statistic, onp.MArray[np.float64] | Any)
assert_type(normaltest(_m_f64_nd, axis=0).statistic, onp.MArray[np.float64] | Any)
assert_type(normaltest(_f64_nd, axis=0).pvalue, onp.ArrayND[np.float64] | Any)
