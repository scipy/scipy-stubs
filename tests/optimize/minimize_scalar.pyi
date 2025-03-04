from typing_extensions import assert_type

import numpy.polynomial as npp
from scipy.optimize import minimize_scalar

def f(x: float) -> float: ...

res = minimize_scalar(f)
assert_type(res.success, bool)
assert_type(res.nit, int)
assert_type(res.nfev, int)

# https://github.com/scipy/scipy-stubs/issues/465
p = npp.Polynomial([3, -2, 1, 1, 0.2])
res_poly = minimize_scalar(p)
assert_type(res.success, bool)
