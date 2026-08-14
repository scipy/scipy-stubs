from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import Bounds, shgo
from scipy.optimize._shgo import OptimizeResult

###

def _obj(x: onp.Array1D[np.float64]) -> float: ...

_f64_2d: onp.Array2D[np.float64]

###

assert_type(shgo(_obj, bounds=[(-5.0, 5.0), (-2.0, 2.0)]), OptimizeResult)
assert_type(shgo(_obj, bounds=((-5.0, 5.0), (-2.0, 2.0))), OptimizeResult)
assert_type(shgo(_obj, bounds=[(-5, 5), (-2, 2)]), OptimizeResult)
assert_type(shgo(_obj, bounds=[(None, None), (None, None)]), OptimizeResult)
assert_type(shgo(_obj, bounds=[(0.0, None), (None, 1.0)]), OptimizeResult)
assert_type(shgo(_obj, bounds=_f64_2d), OptimizeResult)
assert_type(shgo(_obj, bounds=Bounds([-5.0], [5.0])), OptimizeResult)
