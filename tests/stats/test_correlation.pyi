# type-tests for `stats/_correlation.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import chatterjeexi, spearmanrho

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

###

# chatterjeexi
assert_type(chatterjeexi(_f64_1d, _f64_1d).statistic, np.float64)
assert_type(chatterjeexi(_f64_2d, _f64_2d).statistic, onp.Array1D[np.float64])
assert_type(chatterjeexi(_f64_2d, _f64_2d, keepdims=True).statistic, onp.ArrayND[np.float64])

# spearmanrho
assert_type(spearmanrho(_f64_1d, _f64_1d).statistic, np.float64)
assert_type(spearmanrho(_f64_2d, _f64_2d).statistic, onp.Array1D[np.float64])
assert_type(spearmanrho(_f64_2d, _f64_2d, keepdims=True).statistic, onp.ArrayND[np.float64])
