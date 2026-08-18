from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import dual_annealing

###

def _obj(x: onp.Array1D[np.float64]) -> float: ...

_f64_2d: onp.Array2D[np.float64]

###

_res = dual_annealing(_obj, bounds=_f64_2d)
assert_type(_res.x, onp.Array1D[np.float64])
assert_type(_res.fun, np.float64)
assert_type(_res.status, int)
assert_type(_res.success, bool)
assert_type(_res.message, list[str])
assert_type(_res.nit, int)
assert_type(_res.nfev, int)
assert_type(_res.njev, int)
assert_type(_res.nhev, int)
