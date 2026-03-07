from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.optimize import fmin_slsqp

def _func(x: onp.Array1D[np.float64]) -> float: ...

###
# fmin_slsqp

assert_type(fmin_slsqp(_func, [0.5, 0.5]), onp.Array1D[np.float64])

assert_type(fmin_slsqp(_func, [0.5, 0.5], full_output=True)[0], onp.Array1D[np.float64])
assert_type(fmin_slsqp(_func, [0.5, 0.5], full_output=True)[1], float | np.float64)
assert_type(fmin_slsqp(_func, [0.5, 0.5], full_output=True)[2], int)
assert_type(fmin_slsqp(_func, [0.5, 0.5], full_output=True)[3], Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
assert_subtype[str](fmin_slsqp(_func, [0.5, 0.5], full_output=True)[4])
