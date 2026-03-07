from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import lsq_linear

###
# lsq_linear

_A: list[list[float]]
_b: list[float]

_res = lsq_linear(_A, _b)
assert_type(_res.x, onp.Array1D[np.float64])
assert_type(_res.fun, onp.Array1D[np.float64])
assert_type(_res.const, float | np.float64)
assert_type(_res.optimality, float | np.float64)
assert_type(_res.active_mask, onp.Array1D[np.intp])
assert_type(_res.nit, int)
assert_type(_res.status, Literal[-1, 0, 1, 2, 3])
assert_type(_res.message, str)
assert_type(_res.success, bool)

assert_type(lsq_linear(_A, _b, method="bvls").x, onp.Array1D[np.float64])
