from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize.elementwise import bracket_minimum, bracket_root, find_minimum, find_root

type _BOr1D = np.bool | onp.Array1D[np.bool]
type _I32Or1D = np.int32 | onp.Array1D[np.int32]
type _F64Or1D = np.float64 | onp.Array1D[np.float64] | Any
type _F64Or2D = np.float64 | onp.Array2D[np.float64] | Any

def _f_1d(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...
def _f_2d(x: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]: ...
def _g_1d(x: onp.Array1D[np.float64], a: float) -> onp.Array1D[np.float64]: ...

_f64_1d: onp.Array1D[np.float64]

# find_root
assert_type(find_root(_f_1d, (-1.0, 1.0)).success, _BOr1D)
assert_type(find_root(_f_1d, (-1.0, 1.0)).status, _I32Or1D)
assert_type(find_root(_f_1d, (-1.0, 1.0)).x, _F64Or1D)
assert_type(find_root(_f_1d, (-1.0, 1.0)).f_x, _F64Or1D)
assert_type(find_root(_f_1d, (-1.0, 1.0)).nfev, _I32Or1D)
assert_type(find_root(_f_1d, (-1.0, 1.0)).nit, _I32Or1D)
assert_type(find_root(_f_1d, (-1.0, 1.0)).bracket, tuple[_F64Or1D, _F64Or1D])
assert_type(find_root(_f_1d, (-1.0, 1.0)).f_bracket, tuple[_F64Or1D, _F64Or1D])
assert_type(find_root(_f_2d, (-1.0, 1.0)).x, _F64Or2D)
assert_type(find_root(_g_1d, (-1.0, 1.0), args=(0.5,)).x, _F64Or1D)
assert_type(find_root(_f_1d, bracket_root(_f_1d, -1.0).bracket).x, _F64Or1D)
assert_type(find_root(_f_1d, (-1.0, _f64_1d)).x, _F64Or1D)

# find_minimum
assert_type(find_minimum(_f_1d, (-1.0, 0.0, 1.0)).success, _BOr1D)
assert_type(find_minimum(_f_1d, (-1.0, 0.0, 1.0)).status, _I32Or1D)
assert_type(find_minimum(_f_1d, (-1.0, 0.0, 1.0)).x, _F64Or1D)
assert_type(find_minimum(_f_1d, (-1.0, 0.0, 1.0)).f_x, _F64Or1D)
assert_type(find_minimum(_f_1d, (-1.0, 0.0, 1.0)).nfev, _I32Or1D)
assert_type(find_minimum(_f_1d, (-1.0, 0.0, 1.0)).nit, _I32Or1D)
assert_type(find_minimum(_f_1d, (-1.0, 0.0, 1.0)).bracket, tuple[_F64Or1D, _F64Or1D, _F64Or1D])
assert_type(find_minimum(_f_1d, (-1.0, 0.0, 1.0)).f_bracket, tuple[_F64Or1D, _F64Or1D, _F64Or1D])
assert_type(find_minimum(_f_2d, (-1.0, 0.0, 1.0)).x, _F64Or2D)
assert_type(find_minimum(_g_1d, (-1.0, 0.0, 1.0), args=(0.5,)).x, _F64Or1D)
assert_type(find_minimum(_f_1d, bracket_minimum(_f_1d, 0.0).bracket).x, _F64Or1D)

# bracket_root
assert_type(bracket_root(_f_1d, -1.0).success, _BOr1D)
assert_type(bracket_root(_f_1d, -1.0).status, _I32Or1D)
assert_type(bracket_root(_f_1d, -1.0).nfev, _I32Or1D)
assert_type(bracket_root(_f_1d, -1.0).nit, _I32Or1D)
assert_type(bracket_root(_f_1d, -1.0).bracket, tuple[_F64Or1D, _F64Or1D])
assert_type(bracket_root(_f_1d, -1.0).f_bracket, tuple[_F64Or1D, _F64Or1D])
assert_type(bracket_root(_f_2d, -1.0).bracket, tuple[_F64Or2D, _F64Or2D])
assert_type(bracket_root(_g_1d, -1.0, args=(0.5,)).bracket, tuple[_F64Or1D, _F64Or1D])

# bracket_minimum
assert_type(bracket_minimum(_f_1d, 0.0).success, _BOr1D)
assert_type(bracket_minimum(_f_1d, 0.0).status, _I32Or1D)
assert_type(bracket_minimum(_f_1d, 0.0).nfev, _I32Or1D)
assert_type(bracket_minimum(_f_1d, 0.0).nit, _I32Or1D)
assert_type(bracket_minimum(_f_1d, 0.0).bracket, tuple[_F64Or1D, _F64Or1D, _F64Or1D])
assert_type(bracket_minimum(_f_1d, 0.0).f_bracket, tuple[_F64Or1D, _F64Or1D, _F64Or1D])
assert_type(bracket_minimum(_f_2d, 0.0).bracket, tuple[_F64Or2D, _F64Or2D, _F64Or2D])
assert_type(bracket_minimum(_g_1d, 0.0, args=(0.5,)).bracket, tuple[_F64Or1D, _F64Or1D, _F64Or1D])
