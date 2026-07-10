# type-tests for `stats/_binomtest.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import binomtest

###
# binomtest

_r_0d = binomtest(5, 10)
assert_type(_r_0d.k, np.float64)
assert_type(_r_0d.n, np.float64)
assert_type(_r_0d.statistic, np.float64)
assert_type(_r_0d.pvalue, np.float64)
assert_type(_r_0d.proportion_ci().low, np.float64)
assert_type(_r_0d.proportion_ci().high, np.float64)

_r_1d = binomtest([5], [10, 20])
assert_type(_r_1d.k, onp.ArrayND[np.float64])
assert_type(_r_1d.n, onp.ArrayND[np.float64])
assert_type(_r_1d.statistic, onp.ArrayND[np.float64])
assert_type(_r_1d.pvalue, onp.ArrayND[np.float64])
assert_type(_r_1d.proportion_ci().low, onp.ArrayND[np.float64])
assert_type(_r_1d.proportion_ci().high, onp.ArrayND[np.float64])
