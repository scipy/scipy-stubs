from collections.abc import Callable
from typing import Any, Literal, assert_type, overload

import numpy as np
import optype.numpy as onp

from scipy.optimize import Bounds, OptimizeResult, minimize

###

def _f_f32(x: onp.Array1D[np.float64]) -> np.float32: ...
def _f_f64(x: onp.Array1D[np.float64]) -> np.float64: ...
def _f_f64_f64(CCT: onp.Array1D[np.float64], uv_: onp.ArrayND[np.float64]) -> np.float64: ...
def _f_float(x: onp.Array1D[np.float64]) -> float: ...
def _f_jac_f32(x: onp.Array1D[np.float64]) -> tuple[np.float32, onp.Array1D[np.float64]]: ...
def _callback_x(x: onp.ArrayND[np.float64]) -> bool: ...
def _callback_x_state(x: onp.ArrayND[np.float64], state: OptimizeResult) -> bool: ...
def _g_1d(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...
def _dg_2d(x: onp.Array1D[np.float64]) -> onp.Array2D[np.float64]: ...
def _custmin(
    fun: Callable[..., object], x0: onp.ToFloat1D, args: tuple[object, ...] = (), **options: object
) -> OptimizeResult: ...

#
@overload
def _f_jac_over(x: onp.Array1D[np.float64], extra: Literal[False] = False) -> tuple[float, onp.Array1D[np.float64]]: ...
@overload
def _f_jac_over(x: onp.Array1D[np.float64], extra: Literal[True]) -> tuple[float, onp.Array1D[np.float64], np.float64]: ...

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_f64_1d_list: list[onp.Array1D[np.float64]]

###

minimize(lambda x: x**2, 0.0)  # type: ignore[misc]  # pyrefly: ignore[implicit-any-lambda]

minimize(_f_f64_f64, x0=6400, args=(_f64_nd,), method="Nelder-Mead", options={"fatol": 1e-10})

assert_type(minimize(_f_f32, _f64_1d).fun, np.float32)
assert_type(minimize(_f_f64, _f64_1d).fun, np.float64)
assert_type(minimize(_f_float, _f64_1d).fun, float)
assert_type(minimize(_f_f32, _f64_1d, method="BFGS").fun, np.float32)
assert_type(minimize(_f_jac_f32, _f64_1d, jac=True).fun, np.float32)
assert_type(minimize(_f_jac_f32, _f64_1d, (), "SLSQP", True).fun, np.float32)
assert_type(minimize(_f_jac_over, _f64_1d, method="L-BFGS-B", jac=True).fun, float)
assert_type(minimize(_f_jac_over, _f64_1d, (), "L-BFGS-B", True).fun, float)
assert_type(minimize(_f_float, _f64_1d, method="L-BFGS-B", options={"workers": 2}).fun, float)

assert_type(minimize(_f_f32, _f64_1d, method="Nelder-Mead").fun, np.float64)
assert_type(minimize(_f_f32, _f64_1d, method="cobyqa").fun, np.float64)
assert_type(minimize(_f_f32, _f64_1d, (), "COBYLA").fun, np.float64)

assert_type(minimize(_f_f32, _f64_1d, method="COBYLA", options={"catol": 1e-6}).fun, np.float64)
assert_type(minimize(_f_f32, _f64_1d, method="COBYLA").nfev, np.intp)
assert_type(minimize(_f_f32, _f64_1d, method="COBYLA").maxcv, float)
assert_type(minimize(_f_f32, _f64_1d, (), "COBYLA").nfev, np.intp)
assert_type(minimize(_f_f32, _f64_1d, method="cobyqa").nfev, int)

assert_type(minimize(_f_float, _f64_1d, bounds=[(0, 1), (0, 1)]).fun, float)
assert_type(minimize(_f_float, _f64_1d, bounds=[[0, 1], [0, 1]]).fun, float)
assert_type(minimize(_f_float, _f64_1d, bounds=_f64_2d).fun, float)
assert_type(minimize(_f_float, _f64_1d, bounds=[(0, None), (None, 1)]).fun, float)
assert_type(minimize(_f_float, _f64_1d, bounds=Bounds(0, 1)).fun, float)

assert_type(minimize(_f_float, _f64_1d, method="SLSQP", constraints={"type": "ineq", "fun": _g_1d, "jac": _dg_2d}).fun, float)

assert_type(minimize(_f_float, _f64_1d, method="trust-constr").jac, onp.Array1D[np.float64] | Any)
assert_type(minimize(_f_float, _f64_1d, method="BFGS").hess_inv, onp.Array2D[np.float64] | Any)

assert_type(minimize(_f_float, _f64_1d, callback=_callback_x).fun, float)
assert_type(minimize(_f_float, _f64_1d, callback=lambda p: _f64_1d_list.append(p.copy())).fun, float)
assert_type(minimize(_f_float, _f64_1d, method="trust-constr", callback=_callback_x_state).fun, float)
assert_type(minimize(_f_float, _f64_1d, method="CG", options={"maxiter": None}).fun, float)

assert_type(minimize(_f_float, _f64_1d, method=_custmin).fun, float)
