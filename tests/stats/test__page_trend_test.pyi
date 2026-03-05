# type-tests for `stats/_page_trend_test.pyi`

from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import page_trend_test
from scipy.stats._page_trend_test import PageTrendTestResult

###

_f64_2d: onp.Array2D[np.float64]
_py_f_2d: list[list[float]]

###
# page_trend_test

_pt = page_trend_test(_py_f_2d)
assert_type(_pt, PageTrendTestResult)
assert_type(_pt.statistic, np.float64)
assert_type(_pt.pvalue, np.float64)
assert_type(_pt.method, Literal["asymptotic", "exact"])

assert_type(page_trend_test(_f64_2d), PageTrendTestResult)
assert_type(page_trend_test(_f64_2d, ranked=True), PageTrendTestResult)
assert_type(page_trend_test(_f64_2d, method="exact"), PageTrendTestResult)
