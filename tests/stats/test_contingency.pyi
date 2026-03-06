# type-tests for `stats/contingency.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.stats._common import ConfidenceInterval
from scipy.stats._crosstab import CrosstabResult
from scipy.stats._odds_ratio import OddsRatioResult
from scipy.stats._relative_risk import RelativeRiskResult
from scipy.stats.contingency import (
    Chi2ContingencyResult,
    association,
    chi2_contingency,
    crosstab,
    expected_freq,
    margins,
    odds_ratio,
    relative_risk,
)

###

_f64_2d: onp.Array2D[np.float64]
_i64_2d: onp.Array2D[np.int64]

_py_f_2d: list[list[float]]
_py_i_2d: list[list[int]]
_py_U_1d: list[str]
_py_i_1d: list[int]

###
# margins

assert_type(margins(_f64_2d), list[onp.ArrayND[npc.number | np.timedelta64]])

###
# expected_freq

assert_type(expected_freq(_py_f_2d), np.float64 | onp.ArrayND[np.float64])

###
# chi2_contingency

_chi2 = chi2_contingency(_py_i_2d)
assert_type(_chi2, Chi2ContingencyResult)
assert_type(_chi2.statistic, np.float64)
assert_type(_chi2.pvalue, np.float64)
assert_type(_chi2.dof, int)
assert_type(_chi2.expected_freq, onp.ArrayND[np.float64])

assert_type(chi2_contingency(_f64_2d, correction=False), Chi2ContingencyResult)

###
# association

assert_type(association(_py_i_2d), float)
assert_type(association(_f64_2d, method="tschuprow"), float)
assert_type(association(_f64_2d, method="pearson", correction=True), float)

###
# crosstab

_ct_s = crosstab(_py_U_1d, _py_U_1d)
assert_type(_ct_s, CrosstabResult[np.str_])

_ct_i = crosstab(_py_i_1d, _py_i_1d)
assert_type(_ct_i, CrosstabResult[np.int_] | CrosstabResult[np.bool_])

###
# odds_ratio

_or = odds_ratio(_py_i_2d)
assert_type(_or, OddsRatioResult)
assert_type(_or.statistic, float)
assert_type(_or.confidence_interval(), ConfidenceInterval)

assert_type(odds_ratio(_i64_2d, kind="sample"), OddsRatioResult)

###
# relative_risk

_rr = relative_risk(10, 100, 5, 100)
assert_type(_rr, RelativeRiskResult)
assert_type(_rr.relative_risk, float)
assert_type(_rr.exposed_cases, int)
assert_type(_rr.exposed_total, int)
assert_type(_rr.control_cases, int)
assert_type(_rr.control_total, int)
assert_type(_rr.confidence_interval(), ConfidenceInterval)
assert_type(_rr.confidence_interval(confidence_level=0.99), ConfidenceInterval)
