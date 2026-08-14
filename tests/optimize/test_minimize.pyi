from typing import Literal, assert_type, overload

import numpy as np
import optype.numpy as onp

from scipy.optimize import minimize

###

def _f_f32(x: onp.Array1D[np.float64]) -> np.float32: ...
def _f_f64(x: onp.Array1D[np.float64]) -> np.float64: ...
def _f_f64_f64(CCT: onp.Array1D[np.float64], uv_: onp.ArrayND[np.float64]) -> np.float64: ...
def _f_float(x: onp.Array1D[np.float64]) -> float: ...
def _f_jac_f32(x: onp.Array1D[np.float64]) -> tuple[np.float32, onp.Array1D[np.float64]]: ...

#
@overload
def _f_jac_over(x: onp.Array1D[np.float64], extra: Literal[False] = False) -> tuple[float, onp.Array1D[np.float64]]: ...
@overload
def _f_jac_over(x: onp.Array1D[np.float64], extra: Literal[True]) -> tuple[float, onp.Array1D[np.float64], np.float64]: ...

_f64_1d: onp.Array1D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

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

assert_type(minimize(_f_f32, _f64_1d, method="Nelder-Mead").fun, np.float64)
assert_type(minimize(_f_f32, _f64_1d, method="cobyqa").fun, np.float64)
assert_type(minimize(_f_f32, _f64_1d, (), "COBYLA").fun, np.float64)
