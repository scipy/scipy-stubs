# type-tests for `interpolate/_fitpack_py.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import BSpline, splantider, splder, splev, splint, sproot
from scipy.interpolate._fitpack_py import insert, spalde

###

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]

_Float: TypeAlias = float | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]
_Float2D: TypeAlias = onp.Array2D[np.float64]
_FloatND: TypeAlias = onp.ArrayND[np.float64]
_TCK1D: TypeAlias = tuple[_Float1D, _Float1D, int]
_TCK2D: TypeAlias = tuple[_Float1D, list[_Float1D], int]

tck_1d: _TCK1D
tck_2d: _TCK2D
bspl: BSpline[np.float64]

###
# splev

assert_type(splev(f64_1d, bspl), _FloatND)
assert_type(splev(f64_1d, tck_1d), _FloatND)
assert_type(splev(f64_1d, tck_2d), list[_FloatND])

###
# splint

assert_type(splint(0.0, 1.0, bspl), _Float | _Float1D)
assert_type(splint(0.0, 1.0, bspl, full_output=True), tuple[_Float | _Float1D, _Float1D])
assert_type(splint(0.0, 1.0, tck_1d), _Float)
assert_type(splint(0.0, 1.0, tck_1d, full_output=True), tuple[_Float, _Float1D])
assert_type(splint(0.0, 1.0, tck_2d), list[_Float])
assert_type(splint(0.0, 1.0, tck_2d, full_output=True), tuple[list[_Float], _Float1D])

###
# sproot

assert_type(sproot(bspl), _Float1D | _Float2D)
assert_type(sproot(tck_1d), _Float1D)
assert_type(sproot(tck_2d), list[_Float1D])

###
# spalde

assert_type(spalde(0.5, tck_1d), _Float1D)
assert_type(spalde(0.5, tck_2d), list[_Float1D])
assert_type(spalde(f64_1d, tck_1d), list[_Float1D])
assert_type(spalde(f64_1d, tck_2d), list[list[_Float1D]])
assert_type(spalde(f64_2d, tck_1d), list[list[_Float1D]])
assert_type(spalde(f64_2d, tck_2d), list[list[list[_Float1D]]])

###
# insert

assert_type(insert(0.5, bspl), BSpline[np.float64])
assert_type(insert(0.5, tck_1d), _TCK1D)
assert_type(insert(0.5, tck_2d), _TCK2D)

###
# splder / splantider

assert_type(splder(bspl), BSpline[np.float64])
assert_type(splder(tck_1d), _TCK1D)
assert_type(splder(tck_2d), _TCK2D)

assert_type(splantider(bspl), BSpline[np.float64])
assert_type(splantider(tck_1d), _TCK1D)
assert_type(splantider(tck_2d), _TCK2D)
