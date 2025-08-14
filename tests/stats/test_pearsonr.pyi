# type-tests for `pearsonr` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.stats import pearsonr

py_i_1d: list[int]
py_i_2d: list[list[int]]

py_f_1d: list[float]
py_f_2d: list[list[float]]

i8_1d: onp.Array1D[np.int8]
i8_2d: onp.Array2D[np.int8]

f16_1d: onp.Array1D[np.float16]
f16_2d: onp.Array2D[np.float16]

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]

###

assert_type(pearsonr(py_i_1d, py_i_1d).statistic, np.float64)
assert_type(pearsonr(py_i_1d, py_f_1d).statistic, np.float64)
assert_type(pearsonr(py_i_1d, i8_1d).statistic, np.float64)
assert_type(pearsonr(py_i_1d, f16_1d).statistic, np.float64)
assert_type(pearsonr(py_i_1d, f64_1d).statistic, np.float64)
assert_type(pearsonr(py_f_1d, py_i_1d).statistic, np.float64)
assert_type(pearsonr(py_f_1d, py_f_1d).statistic, np.float64)
assert_type(pearsonr(py_f_1d, i8_1d).statistic, np.float64)
assert_type(pearsonr(py_f_1d, f16_1d).statistic, np.float64)
assert_type(pearsonr(py_f_1d, f64_1d).statistic, np.float64)
assert_type(pearsonr(i8_1d, py_i_1d).statistic, np.float64)
assert_type(pearsonr(i8_1d, py_f_1d).statistic, np.float64)
assert_type(pearsonr(i8_1d, i8_1d).statistic, np.float64)
assert_type(pearsonr(i8_1d, f16_1d).statistic, np.float64)
assert_type(pearsonr(i8_1d, f64_1d).statistic, np.float64)
assert_type(pearsonr(f16_1d, py_i_1d).statistic, np.float64)
assert_type(pearsonr(f16_1d, py_f_1d).statistic, np.float64)
assert_type(pearsonr(f16_1d, i8_1d).statistic, np.float64)
assert_type(pearsonr(f16_1d, f16_1d).statistic, npc.floating)
assert_type(pearsonr(f16_1d, f64_1d).statistic, np.float64)
assert_type(pearsonr(f64_1d, py_i_1d).statistic, np.float64)
assert_type(pearsonr(f64_1d, py_f_1d).statistic, np.float64)
assert_type(pearsonr(f64_1d, i8_1d).statistic, np.float64)
assert_type(pearsonr(f64_1d, f16_1d).statistic, np.float64)
assert_type(pearsonr(f64_1d, f64_1d).statistic, np.float64)

assert_type(pearsonr(py_i_2d, py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_i_2d, py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_i_2d, i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_i_2d, f16_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_i_2d, f64_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_f_2d, py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_f_2d, py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_f_2d, i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_f_2d, f16_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(py_f_2d, f64_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(i8_2d, py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(i8_2d, py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(i8_2d, i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(i8_2d, f16_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(i8_2d, f64_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f16_2d, py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f16_2d, py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f16_2d, i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f16_2d, f16_2d, axis=None).statistic, npc.floating)
assert_type(pearsonr(f16_2d, f64_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f64_2d, py_i_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f64_2d, py_f_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f64_2d, i8_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f64_2d, f16_2d, axis=None).statistic, np.float64)
assert_type(pearsonr(f64_2d, f64_2d, axis=None).statistic, np.float64)

assert_type(pearsonr(py_i_2d, py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_i_2d, py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_i_2d, i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_i_2d, f16_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_i_2d, f64_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_f_2d, py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_f_2d, py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_f_2d, i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_f_2d, f16_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(py_f_2d, f64_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(i8_2d, py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(i8_2d, py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(i8_2d, i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(i8_2d, f16_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(i8_2d, f64_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f16_2d, py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f16_2d, py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f16_2d, i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f16_2d, f16_2d).statistic, onp.ArrayND[npc.floating])
assert_type(pearsonr(f16_2d, f64_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f64_2d, py_i_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f64_2d, py_f_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f64_2d, i8_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f64_2d, f16_2d).statistic, onp.ArrayND[np.float64])
assert_type(pearsonr(f64_2d, f64_2d).statistic, onp.ArrayND[np.float64])

assert_type(pearsonr(f16_1d, f16_1d).pvalue, np.float64)
assert_type(pearsonr(f16_2d, f16_2d, axis=None).pvalue, np.float64)
assert_type(pearsonr(f16_2d, f16_2d).pvalue, onp.ArrayND[np.float64])
