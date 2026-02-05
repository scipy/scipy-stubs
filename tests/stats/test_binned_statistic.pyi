# type-tests for `binned_statistic_2d` and `binned_statistic_dd` from `stats/_binned_statistic.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import binned_statistic_2d, binned_statistic_dd

###

_f64_1d: onp.Array1D[np.float64]

###

assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d).binnumber, onp.Array1D[np.intp])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d, expand_binnumbers=False).binnumber, onp.Array1D[np.intp])
assert_type(binned_statistic_2d(_f64_1d, _f64_1d, _f64_1d, expand_binnumbers=True).binnumber, onp.Array2D[np.intp])

assert_type(binned_statistic_dd(_f64_1d, _f64_1d).binnumber, onp.Array1D[np.intp])
assert_type(binned_statistic_dd(_f64_1d, _f64_1d, expand_binnumbers=False).binnumber, onp.Array1D[np.intp])
assert_type(binned_statistic_dd(_f64_1d, _f64_1d, expand_binnumbers=True).binnumber, onp.Array2D[np.intp])
