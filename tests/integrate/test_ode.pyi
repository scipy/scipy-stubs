from typing import assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.integrate import complex_ode, ode

_f64_1d: onp.Array1D[np.float64]

def _f_f64(t: float, y: onp.Array1D[np.float64]) -> list[float]: ...

###
# ode (based on the docstring example)

y0: list[complex]
t0: int
t: float

def f(t: float, y: npt.NDArray[np.complex128], arg1: float) -> list[np.complex128]: ...
def jac(t: float, y: npt.NDArray[np.complex128], arg1: float) -> list[list[np.complex128 | int]]: ...

r1 = ode(f, jac).set_integrator("zvode", method="bdf", atol=_f64_1d)
assert_type(r1, ode[np.complex128, float])
r1.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
assert_type(r1.t, float)
assert_type(r1.integrate(t), onp.Array1D[np.complex128])
assert_type(ode(_f_f64).set_integrator("lsoda", rtol=_f64_1d, atol=_f64_1d), ode[np.float64, *tuple[()]])

# complex_ode

r2 = complex_ode(f, jac).set_integrator("vode", method="bdf", rtol=_f64_1d)
assert_type(r2, complex_ode[float])
r2.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
assert_type(r2.t, float)
assert_type(r2.integrate(t), onp.Array1D[np.complex128])
