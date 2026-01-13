from typing import Literal, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.integrate import cubature

###

def f(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

r = cubature(f, [0], [1])
assert_type(r.estimate, onp.ArrayND[np.float64])
assert_type(r.error, onp.ArrayND[np.float64])
assert_type(r.status, Literal["converged", "not_converged"])
assert_type(r.regions[0].estimate, onp.ArrayND[np.float64])
assert_type(r.regions[0].error, onp.ArrayND[np.float64])
assert_type(r.regions[0].a, onp.Array1D[np.float64])
assert_type(r.regions[0].b, onp.Array1D[np.float64])
assert_type(r.subdivisions, int)
assert_type(r.atol, float)
assert_type(r.rtol, float)
