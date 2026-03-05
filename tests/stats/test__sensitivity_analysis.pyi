# type-tests for `stats/_sensitivity_analysis.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import BootstrapSobolResult, SobolResult

###
# sobol_indices

def _func(x: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]: ...

_si = sobol_indices(func=_func, n=1024, dists=[])
assert_type(_si, SobolResult)
assert_type(_si.first_order, onp.Array2D[np.float64])
assert_type(_si.total_order, onp.Array2D[np.float64])

_bs = _si.bootstrap(confidence_level=0.95, n_resamples=100)
assert_type(_bs, BootstrapSobolResult)
assert_type(_bs.first_order, BootstrapResult)
assert_type(_bs.total_order, BootstrapResult)
