# type-tests for `stats/_bws_test.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import bws_test
from scipy.stats._resampling import PermutationTestResult

###

_f64_1d: onp.Array1D[np.float64]
_py_f_1d: list[float]

###
# bws_test

assert_type(bws_test(_py_f_1d, _py_f_1d), PermutationTestResult)
assert_type(bws_test(_f64_1d, _f64_1d), PermutationTestResult)
assert_type(bws_test(_f64_1d, _f64_1d, alternative="less"), PermutationTestResult)
