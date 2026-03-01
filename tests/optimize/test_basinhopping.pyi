from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import basinhopping
from scipy.optimize._basinhopping import OptimizeResult

def obj(x: onp.Array1D[np.float64]) -> float: ...

x0: onp.Array1D[np.float64]

assert_type(basinhopping(obj, x0), OptimizeResult)
