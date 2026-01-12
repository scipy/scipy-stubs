from typing import assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.integrate import solve_bvp
from scipy.interpolate import PPoly

###
# first example from the `solve_bvp` docstring

x: onp.Array1D[np.float64] = ...
y: onp.Array2D[np.float64] = ...

def fun0(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def bc0(ya: npt.NDArray[np.float64], yb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

res0 = solve_bvp(fun0, bc0, x, y)
assert_type(res0.sol, PPoly)
assert_type(res0.x, onp.Array1D[np.float64])
assert_type(res0.y, onp.Array2D[np.float64])
assert_type(res0.yp, onp.Array2D[np.float64])

# second example from the `solve_bvp` docstring

def fun1(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def bc1(ya: npt.NDArray[np.float64], yb: npt.NDArray[np.float64], p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

res1 = solve_bvp(fun1, bc1, x, y, p=[6])
assert_type(res1.sol, PPoly)
assert_type(res1.x, onp.Array1D[np.float64])
assert_type(res1.y, onp.Array2D[np.float64])
assert_type(res1.yp, onp.Array2D[np.float64])
