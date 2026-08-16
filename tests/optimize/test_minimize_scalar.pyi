from typing import assert_type

import numpy as np
import numpy.polynomial as npp
import optype.numpy as onp

from scipy.optimize import minimize_scalar

def f(x: float, /) -> float: ...

bracket: None
bounds: tuple[float, float]
arr_1d: onp.Array1D[np.float64]

res = minimize_scalar(f)
assert_type(res.success, bool)
assert_type(res.nit, int)
assert_type(res.nfev, int)

# https://github.com/scipy/scipy-stubs/issues/949
res = minimize_scalar(f, bounds=bounds)
res = minimize_scalar(f, bracket, bounds)

res = minimize_scalar(f, bracket=[0, 2])
res = minimize_scalar(f, bracket=[0, 1, 2])
res = minimize_scalar(f, bracket=arr_1d)
res = minimize_scalar(f, bounds=[0, 2], method="bounded")
res = minimize_scalar(f, bounds=arr_1d, method="bounded")

# https://github.com/scipy/scipy-stubs/issues/465
p = npp.Polynomial([3, -2, 1, 1, 0.2])
res_poly = minimize_scalar(p)
assert_type(res.success, bool)
