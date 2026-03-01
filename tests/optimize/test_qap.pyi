from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import quadratic_assignment
from scipy.optimize._qap import OptimizeResult

a2d: onp.Array2D[np.float64]

assert_type(quadratic_assignment(a2d, a2d), OptimizeResult)
assert_type(quadratic_assignment(a2d, a2d, method="faq"), OptimizeResult)
assert_type(quadratic_assignment(a2d, a2d, method="2opt"), OptimizeResult)
