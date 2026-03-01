from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import differential_evolution
from scipy.optimize._differentialevolution import OptimizeResult

def obj(x: onp.Array1D[np.float64]) -> float: ...

assert_type(differential_evolution(obj, bounds=([-5.0], [5.0])), OptimizeResult)
