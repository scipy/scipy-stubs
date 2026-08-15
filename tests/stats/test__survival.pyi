# type-tests for `stats/_survival.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import ecdf, logrank
from scipy.stats._common import ConfidenceInterval
from scipy.stats._survival import ECDFResult, EmpiricalDistributionFunction, LogRankResult

###

_i64_1d: onp.Array1D[np.int64]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_f80_1d: onp.Array1D[np.float128]

_py_f_1d: list[float]

###
# ecdf

_ecdf_r = ecdf(_py_f_1d)
assert_type(_ecdf_r, ECDFResult)
assert_type(_ecdf_r.cdf, EmpiricalDistributionFunction)
assert_type(_ecdf_r.sf, EmpiricalDistributionFunction)

_edf = _ecdf_r.cdf
assert_type(_edf.quantiles, onp.Array1D[np.float64])
assert_type(_edf.probabilities, onp.Array1D[np.float64])
assert_type(_edf.evaluate(_f64_nd), onp.ArrayND[np.float64])
assert_type(_edf.confidence_interval(), ConfidenceInterval[EmpiricalDistributionFunction])
assert_type(_edf.confidence_interval(0.99, method="log-log"), ConfidenceInterval[EmpiricalDistributionFunction])

assert_type(ecdf(_f64_1d), ECDFResult)
assert_type(ecdf(_f32_1d), ECDFResult)
assert_type(ecdf(_i64_1d), ECDFResult)

_ecdf_f80 = ecdf(_f80_1d)
assert_type(_ecdf_f80, ECDFResult[np.longdouble])
assert_type(_ecdf_f80.cdf, EmpiricalDistributionFunction[np.longdouble])
assert_type(_ecdf_f80.sf, EmpiricalDistributionFunction[np.longdouble])
assert_type(_ecdf_f80.cdf.quantiles, onp.Array1D[np.longdouble])
assert_type(_ecdf_f80.cdf.probabilities, onp.Array1D[np.float64])
assert_type(_ecdf_f80.cdf.evaluate(_f64_nd), onp.ArrayND[np.float64])
assert_type(_ecdf_f80.cdf.confidence_interval(), ConfidenceInterval[EmpiricalDistributionFunction[np.longdouble]])

###
# logrank

_lr = logrank(_py_f_1d, _py_f_1d)
assert_type(_lr, LogRankResult)
assert_type(_lr.statistic, np.float64)
assert_type(_lr.pvalue, np.float64)

assert_type(logrank(_f64_1d, _f64_1d, alternative="less"), LogRankResult)
