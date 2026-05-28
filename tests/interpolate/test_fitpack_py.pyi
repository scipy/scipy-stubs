# type-tests for `interpolate/_fitpack_py.pyi`

from typing import LiteralString, assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import (
    BSpline,
    bisplev,
    bisplrep,
    insert,
    spalde,
    splantider,
    splder,
    splev,
    splint,
    splprep,
    splrep,
    sproot,
)

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

type _Float1D = onp.Array1D[np.float64]
type _Float2D = onp.Array2D[np.float64]
type _FloatND = onp.ArrayND[np.float64]
type _TCK1D = tuple[_Float1D, _Float1D, int]
type _TCK2D = tuple[_Float1D, list[_Float1D], int]
type _OutTCK2 = list[_Float1D | _Float2D | int]
type _OutTCKU1 = tuple[list[_Float1D | list[_Float1D] | int], _Float1D]

tck_1d: _TCK1D
tck_2d: _TCK2D
bspl: BSpline[np.float64]

###
# splev

assert_type(splev(_f64_1d, bspl), _FloatND)
assert_type(splev(_f64_1d, tck_1d), _FloatND)
assert_type(splev(_f64_1d, tck_2d), list[_FloatND])

###
# splint

assert_type(splint(0.0, 1.0, bspl), float | _Float1D)
assert_type(splint(0.0, 1.0, bspl, full_output=True), tuple[float, _Float1D] | tuple[_Float1D, _Float1D])
assert_type(splint(0.0, 1.0, tck_1d), float)
assert_type(splint(0.0, 1.0, tck_1d, full_output=True), tuple[float, _Float1D])
assert_type(splint(0.0, 1.0, tck_2d), list[float])
assert_type(splint(0.0, 1.0, tck_2d, full_output=True), tuple[list[float], _Float1D])

###
# sproot

assert_type(sproot(bspl), _Float1D | _Float2D)
assert_type(sproot(tck_1d), _Float1D)
assert_type(sproot(tck_2d), list[_Float1D])

###
# spalde

assert_type(spalde(0.5, tck_1d), _Float1D)
assert_type(spalde(0.5, tck_2d), list[_Float1D])
assert_type(spalde(_f64_1d, tck_1d), list[_Float1D])
assert_type(spalde(_f64_1d, tck_2d), list[list[_Float1D]])
assert_type(spalde(_f64_2d, tck_1d), list[list[_Float1D]])
assert_type(spalde(_f64_2d, tck_2d), list[list[list[_Float1D]]])

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

###
# splrep

assert_type(splrep(_f64_1d, _f64_1d), _TCK1D)
assert_type(splrep(_f64_1d, _f64_1d, full_output=True), tuple[_TCK1D, float, int, LiteralString])

###
# splprep

assert_type(splprep(_f64_2d), _OutTCKU1)
assert_type(splprep(_f64_2d, full_output=True), tuple[_OutTCKU1, float, int, LiteralString])

###
# bisplrep

assert_type(bisplrep(_f64_1d, _f64_1d, _f64_1d), _OutTCK2)
assert_type(bisplrep(_f64_1d, _f64_1d, _f64_1d, full_output=True), tuple[_OutTCK2, float, int, LiteralString])

###
# bisplev

assert_type(bisplev(_f64_1d, _f64_1d, tck_1d), _Float2D)
