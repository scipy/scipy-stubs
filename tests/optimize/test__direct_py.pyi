from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import direct

###
# direct

def _func(x: onp.Array1D[np.float64]) -> float: ...

_res = direct(_func, bounds=([0.0, 0.0], [1.0, 1.0]))
assert_type(_res.x, onp.Array1D[np.float64])
assert_type(_res.fun, float | np.float64)
assert_type(_res.status, int)
assert_type(_res.success, bool)
assert_type(_res.message, str)
assert_type(_res.nfev, int)
assert_type(_res.nit, int)
