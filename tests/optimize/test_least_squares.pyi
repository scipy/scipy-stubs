from typing import assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.optimize import least_squares

###

def _f_f64_0d(x: npt.NDArray[np.float64]) -> float: ...
def _f_f64_nd(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

_f64_nd: npt.NDArray[np.float64]

###
# Regression tests for a mypy bug, see:
# https://github.com/scipy/scipy-stubs/issues/939
# https://github.com/python/mypy/issues/20079

assert_type(least_squares(_f_f64_nd, x0=_f64_nd).x, onp.Array1D[np.float64])
assert_type(least_squares(_f_f64_0d, x0=_f64_nd).x, onp.Array1D[np.float64])

###
# Regression tests for https://github.com/scipy/scipy-stubs/issues/1852

assert_type(least_squares(_f_f64_0d, 0.0).active_mask, onp.Array1D[np.int_ | np.float64])
assert_type(least_squares(_f_f64_0d, 0.0, method="trf").active_mask, onp.Array1D[np.int_ | np.float64])
assert_type(least_squares(_f_f64_0d, 0.0, bounds=(0, 1)).active_mask, onp.Array1D[np.int_ | np.float64])
assert_type(least_squares(_f_f64_0d, 0.0, method="dogbox").active_mask, onp.Array1D[np.int_])
assert_type(least_squares(_f_f64_0d, 0.0, method="lm").active_mask, onp.Array1D[np.int_])

###
# callable `loss`

def _rho(z: onp.Array1D[np.float64]) -> onp.Array2D[np.float64]: ...

assert_type(least_squares(_f_f64_0d, 0.0, loss=_rho).x, onp.Array1D[np.float64])
assert_type(least_squares(_f_f64_0d, 0.0, method="dogbox", loss=_rho).x, onp.Array1D[np.float64])
assert_type(least_squares(_f_f64_0d, 0.0, "2-point", (0, 1), "dogbox", loss=_rho).x, onp.Array1D[np.float64])

###
# scalar `diff_step`

assert_type(least_squares(_f_f64_0d, 0.0, diff_step=0.1).x, onp.Array1D[np.float64])
assert_type(least_squares(_f_f64_0d, 0.0, method="dogbox", diff_step=0.1).x, onp.Array1D[np.float64])
assert_type(least_squares(_f_f64_0d, 0.0, "2-point", (0, 1), "dogbox", diff_step=0.1).x, onp.Array1D[np.float64])
