# type-tests for `spearmanr` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import spearmanr

###

_py_i_1d: list[int]
_py_i_2d: list[list[int]]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]

_i8_1d: onp.Array1D[np.int8]
_i8_2d: onp.Array2D[np.int8]
_i8_nd: onp.ArrayND[np.int8]

_f16_1d: onp.Array1D[np.float16]
_f16_2d: onp.Array2D[np.float16]
_f16_nd: onp.ArrayND[np.float16]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###

assert_type(spearmanr(_py_i_1d, _py_i_1d).statistic, np.float64)
assert_type(spearmanr(_py_f_1d, _py_f_1d).statistic, np.float64)
assert_type(spearmanr(_i8_1d, _i8_1d).statistic, np.float64)
assert_type(spearmanr(_f16_1d, _f16_1d).statistic, np.float64)
assert_type(spearmanr(_f64_1d, _f64_1d).statistic, np.float64)

assert_type(spearmanr(_py_i_2d, _py_i_2d).statistic, onp.Array2D[np.float64])
assert_type(spearmanr(_py_f_2d, _py_f_2d).statistic, onp.Array2D[np.float64])
assert_type(spearmanr(_i8_2d, _i8_2d).statistic, onp.Array2D[np.float64])
assert_type(spearmanr(_f16_2d, _f16_2d).statistic, onp.Array2D[np.float64])
assert_type(spearmanr(_f64_2d, _f64_2d).statistic, onp.Array2D[np.float64])

assert_type(spearmanr(_py_i_2d).statistic, np.float64 | onp.Array2D[np.float64])
assert_type(spearmanr(_py_f_2d).statistic, np.float64 | onp.Array2D[np.float64])
assert_type(spearmanr(_i8_2d).statistic, np.float64 | onp.Array2D[np.float64])
assert_type(spearmanr(_f16_2d).statistic, np.float64 | onp.Array2D[np.float64])
assert_type(spearmanr(_f64_2d).statistic, np.float64 | onp.Array2D[np.float64])

# NOTE: Pyrefly doesn't seem to be able to intersect the return types in case of multiple matching overloads,
# and in this case both return types are even identical (float64 | Array2D[float64]).
assert_type(spearmanr(_i8_nd, _i8_nd).statistic, np.float64 | onp.Array2D[np.float64])  # pyrefly:ignore[assert-type]
assert_type(spearmanr(_f16_nd, _f16_nd).statistic, np.float64 | onp.Array2D[np.float64])  # pyrefly:ignore[assert-type]
assert_type(spearmanr(_f64_nd, _f64_nd).statistic, np.float64 | onp.Array2D[np.float64])  # pyrefly:ignore[assert-type]

assert_type(spearmanr(_i8_nd).statistic, np.float64 | onp.Array2D[np.float64])
assert_type(spearmanr(_f16_nd).statistic, np.float64 | onp.Array2D[np.float64])
assert_type(spearmanr(_f64_nd).statistic, np.float64 | onp.Array2D[np.float64])
