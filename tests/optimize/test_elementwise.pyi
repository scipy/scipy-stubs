from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize.elementwise import bracket_minimum, bracket_root, find_minimum, find_root

type _VecB = onp.Array1D[np.bool]
type _VecI32 = onp.Array1D[np.int32]
type _VecF64 = onp.Array1D[np.float64]
type _MatF64 = onp.Array2D[np.float64]

def f_1d(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...
def f_2d(x: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]: ...
def g_1d(x: onp.Array1D[np.float64], a: float) -> onp.Array1D[np.float64]: ...

# find_root
assert_type(find_root(f_1d, (-1.0, 1.0)).success, _VecB)
assert_type(find_root(f_1d, (-1.0, 1.0)).status, _VecI32)
assert_type(find_root(f_1d, (-1.0, 1.0)).x, _VecF64)
assert_type(find_root(f_1d, (-1.0, 1.0)).f_x, _VecF64)
assert_type(find_root(f_1d, (-1.0, 1.0)).nfev, _VecI32)
assert_type(find_root(f_1d, (-1.0, 1.0)).nit, _VecI32)
assert_type(find_root(f_1d, (-1.0, 1.0)).bracket, tuple[_VecF64, _VecF64])
assert_type(find_root(f_1d, (-1.0, 1.0)).f_bracket, tuple[_VecF64, _VecF64])
assert_type(find_root(f_2d, (-1.0, 1.0)).x, _MatF64)
assert_type(find_root(g_1d, (-1.0, 1.0), args=(0.5,)).x, _VecF64)

# find_minimum
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)).success, _VecB)
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)).status, _VecI32)
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)).x, _VecF64)
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)).f_x, _VecF64)
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)).nfev, _VecI32)
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)).nit, _VecI32)
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)).bracket, tuple[_VecF64, _VecF64, _VecF64])
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)).f_bracket, tuple[_VecF64, _VecF64, _VecF64])
assert_type(find_minimum(f_2d, (-1.0, 0.0, 1.0)).x, _MatF64)
assert_type(find_minimum(g_1d, (-1.0, 0.0, 1.0), args=(0.5,)).x, _VecF64)

# bracket_root
assert_type(bracket_root(f_1d, -1.0).success, _VecB)
assert_type(bracket_root(f_1d, -1.0).status, _VecI32)
assert_type(bracket_root(f_1d, -1.0).nfev, _VecI32)
assert_type(bracket_root(f_1d, -1.0).nit, _VecI32)
assert_type(bracket_root(f_1d, -1.0).bracket, tuple[_VecF64, _VecF64])
assert_type(bracket_root(f_1d, -1.0).f_bracket, tuple[_VecF64, _VecF64])
assert_type(bracket_root(f_2d, -1.0).bracket, tuple[_MatF64, _MatF64])
assert_type(bracket_root(g_1d, -1.0, args=(0.5,)).bracket, tuple[_VecF64, _VecF64])

# bracket_minimum
assert_type(bracket_minimum(f_1d, 0.0).success, _VecB)
assert_type(bracket_minimum(f_1d, 0.0).status, _VecI32)
assert_type(bracket_minimum(f_1d, 0.0).nfev, _VecI32)
assert_type(bracket_minimum(f_1d, 0.0).nit, _VecI32)
assert_type(bracket_minimum(f_1d, 0.0).bracket, tuple[_VecF64, _VecF64, _VecF64])
assert_type(bracket_minimum(f_1d, 0.0).f_bracket, tuple[_VecF64, _VecF64, _VecF64])
assert_type(bracket_minimum(f_2d, 0.0).bracket, tuple[_MatF64, _MatF64, _MatF64])
assert_type(bracket_minimum(g_1d, 0.0, args=(0.5,)).bracket, tuple[_VecF64, _VecF64, _VecF64])
