# type-tests for `rankdata` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import rankdata

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

assert_type(rankdata(_py_i_1d), onp.Array1D[np.float64])
assert_type(rankdata(_py_f_1d), onp.Array1D[np.float64])
assert_type(rankdata(_i8_1d), onp.Array1D[np.float64])
assert_type(rankdata(_f16_1d), onp.Array1D[np.float64])
assert_type(rankdata(_f64_1d), onp.Array1D[np.float64])

assert_type(rankdata(_py_i_2d), onp.Array1D[np.float64])
assert_type(rankdata(_py_f_2d), onp.Array1D[np.float64])
assert_type(rankdata(_i8_2d), onp.Array1D[np.float64])
assert_type(rankdata(_f16_2d), onp.Array1D[np.float64])
assert_type(rankdata(_f64_2d), onp.Array1D[np.float64])

assert_type(rankdata(_py_i_3d), onp.Array1D[np.float64])
assert_type(rankdata(_py_f_3d), onp.Array1D[np.float64])
assert_type(rankdata(_i8_3d), onp.Array1D[np.float64])
assert_type(rankdata(_f16_3d), onp.Array1D[np.float64])
assert_type(rankdata(_f64_3d), onp.Array1D[np.float64])

assert_type(rankdata(_i8_nd), onp.Array1D[np.float64])
assert_type(rankdata(_f16_nd), onp.Array1D[np.float64])
assert_type(rankdata(_f64_nd), onp.Array1D[np.float64])

assert_type(rankdata(_py_i_1d, axis=0), onp.Array1D[np.float64])
assert_type(rankdata(_py_f_1d, axis=0), onp.Array1D[np.float64])
assert_type(rankdata(_i8_1d, axis=0), onp.Array1D[np.float64])
assert_type(rankdata(_f16_1d, axis=0), onp.Array1D[np.float64])
assert_type(rankdata(_f64_1d, axis=0), onp.Array1D[np.float64])

assert_type(rankdata(_py_i_2d, axis=0), onp.Array2D[np.float64])
assert_type(rankdata(_py_f_2d, axis=0), onp.Array2D[np.float64])
assert_type(rankdata(_i8_2d, axis=0), onp.Array2D[np.float64])
assert_type(rankdata(_f16_2d, axis=0), onp.Array2D[np.float64])
assert_type(rankdata(_f64_2d, axis=0), onp.Array2D[np.float64])

assert_type(rankdata(_py_i_3d, axis=0), onp.Array3D[np.float64])
assert_type(rankdata(_py_f_3d, axis=0), onp.Array3D[np.float64])
assert_type(rankdata(_i8_3d, axis=0), onp.Array3D[np.float64])
assert_type(rankdata(_f16_3d, axis=0), onp.Array3D[np.float64])
assert_type(rankdata(_f64_3d, axis=0), onp.Array3D[np.float64])

assert_type(rankdata(_i8_nd, axis=0), onp.ArrayND[np.float64])
assert_type(rankdata(_f16_nd, axis=0), onp.ArrayND[np.float64])
assert_type(rankdata(_f64_nd, axis=0), onp.ArrayND[np.float64])
