from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import fmin_tnc

###
# fmin_tnc

def _func(x: onp.Array1D[np.float64]) -> float: ...
def _fgrad(x: onp.Array1D[np.float64]) -> tuple[float, onp.Array1D[np.float64]]: ...
def _grad(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...

_x0: list[float]

assert_type(fmin_tnc(_func, _x0)[0], onp.Array1D[np.float64])
assert_type(fmin_tnc(_func, _x0)[1], int)
assert_type(fmin_tnc(_func, _x0)[2], Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7])
assert_type(fmin_tnc(_fgrad, _x0)[0], onp.Array1D[np.float64])
assert_type(fmin_tnc(_func, _x0, _grad)[0], onp.Array1D[np.float64])
