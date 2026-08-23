# type-tests for `stats/_correlation.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import chatterjeexi, siegelslopes, spearmanrho, theilslopes

###

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

###

# chatterjeexi
assert_type(chatterjeexi(_f64_1d, _f64_1d).statistic, np.float64)
assert_type(chatterjeexi(_f32_1d, _f32_1d).statistic, np.float32)
assert_type(chatterjeexi(_f64_2d, _f64_2d).statistic, onp.Array1D[np.float64])
assert_type(chatterjeexi(_f32_2d, _f32_2d).statistic, onp.Array1D[np.float32])
assert_type(chatterjeexi(_f64_2d, _f64_2d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(chatterjeexi(_f32_2d, _f32_2d, keepdims=True).statistic, onp.ArrayND[np.float32])
assert_type(chatterjeexi(_f32_1d, _f64_1d).statistic, np.float64 | Any)

# spearmanrho
assert_type(spearmanrho(_f64_1d, _f64_1d).statistic, np.float64)
assert_type(spearmanrho(_f32_1d, _f32_1d).statistic, np.float32)
assert_type(spearmanrho(_f64_2d, _f64_2d).statistic, onp.Array1D[np.float64])
assert_type(spearmanrho(_f32_2d, _f32_2d).statistic, onp.Array1D[np.float32])
assert_type(spearmanrho(_f64_2d, _f64_2d, keepdims=True).statistic, onp.ArrayND[np.float64])
assert_type(spearmanrho(_f32_2d, _f32_2d, keepdims=True).statistic, onp.ArrayND[np.float32])
assert_type(spearmanrho(_f32_1d, _f64_1d).statistic, np.float64 | Any)

# theilslopes
assert_type(theilslopes(_f64_1d).slope, np.float64)
assert_type(theilslopes(_f32_1d).slope, np.float32)
assert_type(theilslopes(_f32_1d, _f32_1d).slope, np.float32)
assert_type(theilslopes(_f32_2d, axis=0).slope, onp.Array1D[np.float32])
assert_type(theilslopes(_f32_2d, keepdims=True).slope, onp.ArrayND[np.float32])
assert_type(theilslopes(_f32_1d, _f64_1d).slope, np.float64 | Any)

# siegelslopes
assert_type(siegelslopes(_f64_1d).slope, np.float64)
assert_type(siegelslopes(_f32_1d).slope, np.float32)
assert_type(siegelslopes(_f32_1d, _f32_1d).slope, np.float32)
assert_type(siegelslopes(_f32_2d, axis=0).slope, onp.Array1D[np.float32])
assert_type(siegelslopes(_f32_2d, keepdims=True).slope, onp.ArrayND[np.float32])
assert_type(siegelslopes(_f32_1d, _f64_1d).slope, np.float64 | Any)
