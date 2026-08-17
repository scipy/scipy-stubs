# type-tests for `stats/_bws_test.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.stats import bws_test
from scipy.stats._resampling import PermutationTestResult

###

_i64_1d: onp.Array1D[np.int64]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_py_f_1d: list[float]

###
# bws_test

assert_type(bws_test(_py_f_1d, _py_f_1d), PermutationTestResult[np.float64, onp.Array1D[np.float64]])
assert_type(bws_test(_f64_1d, _f64_1d), PermutationTestResult[np.float64, onp.Array1D[np.float64]])
assert_type(bws_test(_i64_1d, _i64_1d), PermutationTestResult[np.float64, onp.Array1D[np.float64]])
assert_type(bws_test(_f64_1d, _f64_1d, alternative="less"), PermutationTestResult[np.float64, onp.Array1D[np.float64]])
assert_type(bws_test(_f32_1d, _f32_1d), PermutationTestResult[np.float32, onp.Array1D[np.float32]])
# workaround for pyrefly (1.2.0), which infers `Unknown` here
assert_subtype[PermutationTestResult[onp.ArrayND[np.float64] | np.float64, onp.ArrayND[np.float64]]](bws_test(_f64_nd, _f64_nd))

assert_type(bws_test(_f64_2d, _f64_2d, axis=1), PermutationTestResult[onp.Array1D[np.float64], onp.Array2D[np.float64]])
assert_type(bws_test(_f32_2d, _f32_2d, axis=1), PermutationTestResult[onp.Array1D[np.float32], onp.Array2D[np.float32]])

assert_type(bws_test(_f64_1d, _f64_1d, axis=None), PermutationTestResult[np.float64, onp.Array1D[np.float64]])
assert_type(bws_test(_f64_2d, _f64_2d, axis=None), PermutationTestResult[np.float64, onp.Array1D[np.float64]])
assert_type(bws_test(_f32_2d, _f32_2d, axis=None), PermutationTestResult[np.float32, onp.Array1D[np.float32]])
assert_type(bws_test(_f64_nd, _f64_nd, axis=None), PermutationTestResult[np.float64, onp.Array1D[np.float64]])
