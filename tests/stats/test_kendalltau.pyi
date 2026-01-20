# type-tests for `kendalltau` from `stats/_stats_py.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import kendalltau

###

_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_i_3d: list[list[list[int]]]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]

_i8_1d: onp.Array1D[np.int8]
_i8_2d: onp.Array2D[np.int8]
_i8_3d: onp.Array3D[np.int8]
_i8_nd: onp.ArrayND[np.int8]

_f16_1d: onp.Array1D[np.float16]
_f16_2d: onp.Array2D[np.float16]
_f16_3d: onp.Array3D[np.float16]
_f16_nd: onp.ArrayND[np.float16]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###

assert_type(kendalltau(_py_i_1d, _py_i_1d).statistic, np.float64)
assert_type(kendalltau(_py_f_1d, _py_f_1d).statistic, np.float64)
assert_type(kendalltau(_i8_1d, _i8_1d).statistic, np.float64)
assert_type(kendalltau(_f16_1d, _f16_1d).statistic, np.float64)
assert_type(kendalltau(_f64_1d, _f64_1d).statistic, np.float64)

assert_type(kendalltau(_py_i_2d, _py_i_2d).statistic, np.float64)
assert_type(kendalltau(_py_f_2d, _py_f_2d).statistic, np.float64)
assert_type(kendalltau(_i8_2d, _i8_2d).statistic, np.float64)
assert_type(kendalltau(_f16_2d, _f16_2d).statistic, np.float64)
assert_type(kendalltau(_f64_2d, _f64_2d).statistic, np.float64)

assert_type(kendalltau(_i8_nd, _i8_nd).statistic, np.float64)
assert_type(kendalltau(_f16_nd, _f16_nd).statistic, np.float64)
assert_type(kendalltau(_f64_nd, _f64_nd).statistic, np.float64)

assert_type(kendalltau(_py_i_1d, _py_i_1d, axis=0).statistic, np.float64)
assert_type(kendalltau(_py_f_1d, _py_f_1d, axis=0).statistic, np.float64)
assert_type(kendalltau(_i8_1d, _i8_1d, axis=0).statistic, np.float64)
assert_type(kendalltau(_f16_1d, _f16_1d, axis=0).statistic, np.float64)
assert_type(kendalltau(_f64_1d, _f64_1d, axis=0).statistic, np.float64)

assert_type(kendalltau(_py_i_2d, _py_i_2d, axis=0).statistic, onp.Array1D[np.float64])
assert_type(kendalltau(_py_f_2d, _py_f_2d, axis=0).statistic, onp.Array1D[np.float64])
assert_type(kendalltau(_i8_2d, _i8_2d, axis=0).statistic, onp.Array1D[np.float64])
assert_type(kendalltau(_f16_2d, _f16_2d, axis=0).statistic, onp.Array1D[np.float64])
assert_type(kendalltau(_f64_2d, _f64_2d, axis=0).statistic, onp.Array1D[np.float64])

assert_type(kendalltau(_py_i_3d, _py_i_3d, axis=0).statistic, onp.Array2D[np.float64])
assert_type(kendalltau(_py_f_3d, _py_f_3d, axis=0).statistic, onp.Array2D[np.float64])
assert_type(kendalltau(_i8_3d, _i8_3d, axis=0).statistic, onp.Array2D[np.float64])
assert_type(kendalltau(_f16_3d, _f16_3d, axis=0).statistic, onp.Array2D[np.float64])
assert_type(kendalltau(_f64_3d, _f64_3d, axis=0).statistic, onp.Array2D[np.float64])

# NOTE: Pyrefly doesn't seem to be able to intersect the return types in case of multiple matching overloads,
# and in this case both return types are even identical (ArrayND[float64] | Any).
assert_type(kendalltau(_i8_nd, _i8_nd, axis=0).statistic, onp.ArrayND[np.float64] | Any)  # pyrefly:ignore[assert-type]
assert_type(kendalltau(_f16_nd, _f16_nd, axis=0).statistic, onp.ArrayND[np.float64] | Any)  # pyrefly:ignore[assert-type]
assert_type(kendalltau(_f64_nd, _f64_nd, axis=0).statistic, onp.ArrayND[np.float64] | Any)  # pyrefly:ignore[assert-type]
