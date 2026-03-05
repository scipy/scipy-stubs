# type-tests for `stats/_mgc.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import multiscale_graphcorr
from scipy.stats._mgc import MGCResult

###

_f64_2d: onp.Array2D[np.float64]

###
# multiscale_graphcorr

_mgc = multiscale_graphcorr(_f64_2d, _f64_2d)
assert_type(_mgc, MGCResult)
assert_type(_mgc.statistic, np.float64)
assert_type(_mgc.pvalue, np.float64)

assert_type(multiscale_graphcorr(_f64_2d, _f64_2d, reps=100, workers=2), MGCResult)
