# type-tests for `stats/_multicomp.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import dunnett
from scipy.stats._common import ConfidenceInterval
from scipy.stats._multicomp import DunnettResult

###

_f1d: onp.Array1D[np.float64]
_f80_1d: onp.Array1D[np.longdouble]
_py_f_1d: list[float]

###
# dunnett

_d = dunnett(_py_f_1d, _py_f_1d, control=_py_f_1d)
assert_type(_d, DunnettResult[np.float64])
assert_type(_d.statistic, onp.Array1D[np.float64])
assert_type(_d.pvalue, onp.Array1D[np.float64])
assert_type(_d.confidence_interval(), ConfidenceInterval[np.float64])

assert_type(dunnett(_f1d, control=_f1d), DunnettResult[np.float64])
assert_type(dunnett(_f1d, _f1d, _f1d, control=_f1d, alternative="less"), DunnettResult[np.float64])
assert_type(dunnett(_f80_1d, control=_f80_1d), DunnettResult[np.longdouble])
assert_type(dunnett(_f1d, control=_f80_1d), DunnettResult[np.float64])
assert_type(dunnett(_f80_1d, control=_f1d), DunnettResult[np.longdouble])
assert_type(dunnett(_f1d, _f80_1d, control=_f1d), DunnettResult[np.longdouble])
