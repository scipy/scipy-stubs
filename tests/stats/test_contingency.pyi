# type-tests for `stats/contingency.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats.contingency import (
    Chi2ContingencyResult,
    association,
    chi2_contingency,
    expected_freq,
    margins,
    odds_ratio,
    relative_risk,
)

###

_bool_2d: onp.Array2D[np.bool_]
_i64_2d: onp.Array2D[np.int64]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_py_f_2d: list[list[float]]
_py_i_2d: list[list[int]]
_py_i_1d: list[int]

###

# margins
assert_type(margins(_i64_2d), list[onp.Array2D[np.int64]])
assert_type(margins(_f32_2d), list[onp.Array2D[np.float32]])
assert_type(margins(_f64_2d), list[onp.Array2D[np.float64]])
assert_type(margins(_f64_3d), list[onp.Array3D[np.float64]])
assert_type(margins(_f64_nd), list[onp.ArrayND[np.float64]])
assert_type(margins(_bool_2d), list[onp.ArrayND[np.int_]])

# expected_freq
assert_type(expected_freq(_py_f_2d), onp.ArrayND[np.float64])
assert_type(expected_freq(_f32_2d), onp.Array2D[np.float64])  # always casts to float64
assert_type(expected_freq(_f64_2d), onp.Array2D[np.float64])
assert_type(expected_freq(_f64_3d), onp.Array3D[np.float64])
assert_type(expected_freq(_f64_nd), onp.ArrayND[np.float64])

# chi2_contingency
assert_type(chi2_contingency(_f32_2d, correction=False), Chi2ContingencyResult[tuple[int, int]])
assert_type(chi2_contingency(_f64_2d, correction=False), Chi2ContingencyResult[tuple[int, int]])
assert_type(chi2_contingency(_f64_3d, correction=False), Chi2ContingencyResult[tuple[int, int, int]])
assert_type(chi2_contingency(_f64_nd, correction=False), Chi2ContingencyResult)

# association
assert_type(association(_py_i_2d), float)
assert_type(association(_i64_2d), float)

# odds_ratio
_or = odds_ratio(_py_i_2d)
assert_type(_or.statistic, float)

# relative_risk
_rr = relative_risk(10, 100, 5, 100)
assert_type(_rr.relative_risk, float)
assert_type(_rr.exposed_cases, int)
assert_type(_rr.exposed_total, int)
assert_type(_rr.control_cases, int)
assert_type(_rr.control_total, int)
