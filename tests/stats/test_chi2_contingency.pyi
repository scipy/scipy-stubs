# type-tests for `scipy.stats.chi2_contingency` top-level export

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import chi2_contingency
from scipy.stats.contingency import Chi2ContingencyResult

###

_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###

assert_type(chi2_contingency(_f32_2d, correction=False), Chi2ContingencyResult[tuple[int, int]])
assert_type(chi2_contingency(_f64_2d, correction=False), Chi2ContingencyResult[tuple[int, int]])
assert_type(chi2_contingency(_f64_3d, correction=False), Chi2ContingencyResult[tuple[int, int, int]])
assert_type(chi2_contingency(_f64_nd, correction=False), Chi2ContingencyResult)
assert_type(chi2_contingency(_f64_2d), Chi2ContingencyResult[tuple[int, int]])
assert_type(chi2_contingency(_f64_2d, lambda_="log-likelihood"), Chi2ContingencyResult[tuple[int, int]])

###

res = chi2_contingency(_f64_2d)
assert_type(res.statistic, np.float64)
assert_type(res.pvalue, np.float64)
assert_type(res.dof, int)
assert_type(res.expected_freq, onp.Array2D[np.float64])
