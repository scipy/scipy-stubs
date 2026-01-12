from typing import assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.integrate import ode

# based on the example from the `ode` docstring

y0: list[complex]
t0: int

def f(t: float, y: npt.NDArray[np.complex128], arg1: float) -> list[np.complex128]: ...
def jac(t: float, y: npt.NDArray[np.complex128], arg1: float) -> list[list[np.complex128 | int]]: ...

r = ode(f, jac).set_integrator("zvode", method="bdf")
assert_type(r, ode[np.complex128, float])

_ = r.set_initial_value(y0, t0).set_f_params(2.0).set_jac_params(2.0)
assert_type(r.t, float)

t: float
assert_type(r.integrate(t), onp.Array1D[np.complex128])
