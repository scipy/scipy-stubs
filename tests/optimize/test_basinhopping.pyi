from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import basinhopping

###

def _fn_f64_1d(x: onp.Array1D[np.float64]) -> float: ...
def _fn_f64_1d_jac(x: onp.Array1D[np.float64]) -> tuple[float, onp.Array1D[np.float64]]: ...

_f64_1d: onp.Array1D[np.float64]

###

assert_type(basinhopping(_fn_f64_1d, _f64_1d).success, bool)
assert_type(basinhopping(_fn_f64_1d, _f64_1d).nit, int)
assert_type(basinhopping(_fn_f64_1d, _f64_1d).message, list[str])
assert_type(basinhopping(_fn_f64_1d, _f64_1d).x, onp.Array1D[np.float64])
assert_type(basinhopping(_fn_f64_1d, _f64_1d).fun, float | np.float64)
assert_type(basinhopping(_fn_f64_1d, _f64_1d).minimization_failures, int)
assert_type(basinhopping(_fn_f64_1d, _f64_1d).lowest_optimization_result.status, int)

# https://github.com/scipy/scipy-stubs/issues/1730
assert_type(basinhopping(_fn_f64_1d, _f64_1d, minimizer_kwargs={"args": (1, 2)}).x, onp.Array1D[np.float64])
assert_type(basinhopping(_fn_f64_1d, _f64_1d, minimizer_kwargs={"bounds": [(0, 1)]}).x, onp.Array1D[np.float64])

# https://github.com/scipy/scipy-stubs/issues/1733
assert_type(basinhopping(_fn_f64_1d, _f64_1d, minimizer_kwargs={"jac": False}).x, onp.Array1D[np.float64])
assert_type(basinhopping(_fn_f64_1d_jac, _f64_1d, minimizer_kwargs={"jac": True}).x, onp.Array1D[np.float64])
