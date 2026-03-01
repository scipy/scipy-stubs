from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import dual_annealing
from scipy.optimize._optimize import OptimizeResult

def obj(x: onp.Array1D[np.float64]) -> float: ...

bounds_2d: onp.Array2D[np.float64]

assert_type(dual_annealing(obj, bounds=bounds_2d), OptimizeResult)
