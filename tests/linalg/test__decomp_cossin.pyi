# type-tests for `linalg/_decomp_cossin.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import cossin

_Float1D: TypeAlias = onp.Array1D[npc.floating]
_Float2D: TypeAlias = onp.Array2D[npc.floating]
_Inexact2D: TypeAlias = onp.Array2D[npc.inexact]

###

f64_2d: onp.Array2D[np.float64]
f64_nd: onp.ArrayND[np.float64]
py_c_2d: list[list[complex]]
c128_nd: onp.ArrayND[np.complex128]

###
# cossin

assert_type(cossin(f64_2d), tuple[_Float2D, _Float2D, _Float2D])
assert_type(cossin(f64_nd), tuple[_Float2D, _Float2D, _Float2D])
assert_type(cossin(f64_2d, separate=True), tuple[tuple[_Float2D, _Float2D], _Float1D, tuple[_Float2D, _Float2D]])
assert_type(cossin(f64_nd, separate=True), tuple[tuple[_Float2D, _Float2D], _Float1D, tuple[_Float2D, _Float2D]])

assert_type(cossin(py_c_2d), tuple[_Inexact2D, _Inexact2D, _Inexact2D])
assert_type(cossin(c128_nd), tuple[_Float2D, _Float2D, _Float2D])
assert_type(cossin(py_c_2d, separate=True), tuple[tuple[_Inexact2D, _Inexact2D], _Float1D, tuple[_Inexact2D, _Inexact2D]])
assert_type(cossin(c128_nd, separate=True), tuple[tuple[_Float2D, _Float2D], _Float1D, tuple[_Float2D, _Float2D]])
