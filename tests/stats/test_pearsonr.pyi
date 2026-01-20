# type-tests for `pearsonr` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.stats import pearsonr

###

_py_i_1d: list[int]
_py_i_2d: list[list[int]]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]

_i8_1d: onp.Array1D[np.int8]
_i8_2d: onp.Array2D[np.int8]

_f16_1d: onp.Array1D[np.float16]
_f16_2d: onp.Array2D[np.float16]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

###

assert_type(pearsonr(_py_i_1d, _py_i_1d).statistic, np.float64)
assert_type(pearsonr(_py_i_1d, _py_f_1d).statistic, np.float64)
assert_type(pearsonr(_py_i_1d, _i8_1d).statistic, np.float64)
assert_type(pearsonr(_py_i_1d, _f16_1d).statistic, np.float64)
assert_type(pearsonr(_py_i_1d, _f64_1d).statistic, np.float64)
assert_type(pearsonr(_py_f_1d, _py_i_1d).statistic, np.float64)
assert_type(pearsonr(_py_f_1d, _py_f_1d).statistic, np.float64)
assert_type(pearsonr(_py_f_1d, _i8_1d).statistic, np.float64)
assert_type(pearsonr(_py_f_1d, _f16_1d).statistic, np.float64)
assert_type(pearsonr(_py_f_1d, _f64_1d).statistic, np.float64)
assert_type(pearsonr(_i8_1d, _py_i_1d).statistic, np.float64)
assert_type(pearsonr(_i8_1d, _py_f_1d).statistic, np.float64)
assert_type(pearsonr(_i8_1d, _i8_1d).statistic, np.float64)
assert_type(pearsonr(_i8_1d, _f16_1d).statistic, np.float64)
assert_type(pearsonr(_i8_1d, _f64_1d).statistic, np.float64)
assert_type(pearsonr(_f16_1d, _py_i_1d).statistic, np.float64)
assert_type(pearsonr(_f16_1d, _py_f_1d).statistic, np.float64)
assert_type(pearsonr(_f16_1d, _i8_1d).statistic, np.float64)
assert_type(pearsonr(_f16_1d, _f16_1d).statistic, npc.floating)
assert_type(pearsonr(_f16_1d, _f64_1d).statistic, np.float64)
assert_type(pearsonr(_f64_1d, _py_i_1d).statistic, np.float64)
assert_type(pearsonr(_f64_1d, _py_f_1d).statistic, np.float64)
assert_type(pearsonr(_f64_1d, _i8_1d).statistic, np.float64)
assert_type(pearsonr(_f64_1d, _f16_1d).statistic, np.float64)
assert_type(pearsonr(_f64_1d, _f64_1d).statistic, np.float64)

assert_type(pearsonr(_py_i_2d, _py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_i_2d, _py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_i_2d, _i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_i_2d, _f16_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_i_2d, _f64_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_f_2d, _py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_f_2d, _py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_f_2d, _i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_f_2d, _f16_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_py_f_2d, _f64_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_i8_2d, _py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_i8_2d, _py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_i8_2d, _i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_i8_2d, _f16_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_i8_2d, _f64_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f16_2d, _py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f16_2d, _py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f16_2d, _i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f16_2d, _f16_2d, axis=None).statistic, npc.floating)
assert_type(pearsonr(_f16_2d, _f64_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f64_2d, _py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f64_2d, _py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f64_2d, _i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f64_2d, _f16_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(_f64_2d, _f64_2d, axis=None).statistic, np.float64)

assert_type(pearsonr(_py_i_2d, _py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_i_2d, _py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_i_2d, _i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_i_2d, _f16_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_i_2d, _f64_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_f_2d, _py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_f_2d, _py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_f_2d, _i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_f_2d, _f16_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_py_f_2d, _f64_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_i8_2d, _py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_i8_2d, _py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_i8_2d, _i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_i8_2d, _f16_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_i8_2d, _f64_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f16_2d, _py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f16_2d, _py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f16_2d, _i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f16_2d, _f16_2d).statistic, onp.ArrayND[npc.floating])
assert_type(pearsonr(_f16_2d, _f64_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f64_2d, _py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f64_2d, _py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f64_2d, _i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f64_2d, _f16_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(_f64_2d, _f64_2d).statistic, onp.ArrayND[np.float64])

assert_type(pearsonr(_f16_1d, _f16_1d).pvalue, np.float64)
assert_type(pearsonr(_f16_2d, _f16_2d, axis=None).pvalue, np.float64)
assert_type(pearsonr(_f16_2d, _f16_2d).pvalue, onp.ArrayND[np.float64])
