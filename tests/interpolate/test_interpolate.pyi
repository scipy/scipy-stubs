# type-tests for `interpolate/_interpolate.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import BPoly, NdPPoly, interp1d, interp2d, lagrange

###

_f64_1d: onp.Array1D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###
# BPoly

_bpoly: BPoly[np.float64]
assert_type(_bpoly(0.5), onp.ArrayND[np.float64])
assert_type(_bpoly.derivative(), BPoly[np.float64])
assert_type(_bpoly.antiderivative(), BPoly[np.float64])
assert_type(_bpoly.integrate(0.0, 1.0), onp.ArrayND[np.float64])
assert_type(BPoly.from_derivatives(_f64_1d, _f64_nd), BPoly[np.float64])

###
# NdPPoly

_ndppoly: NdPPoly[np.float64]
assert_type(_ndppoly(0.5), onp.ArrayND[np.float64])
assert_type(_ndppoly.derivative((1,)), NdPPoly[np.float64])
assert_type(_ndppoly.antiderivative((1,)), NdPPoly[np.float64])
assert_type(_ndppoly.integrate(((0.0, 1.0),)), onp.ArrayND[np.float64])

###
# interp1d

interp1d_f: interp1d
assert_type(interp1d(_f64_1d, _f64_1d), interp1d)
assert_type(interp1d_f(0.5), onp.ArrayND[np.float64 | np.complex128])

###
# interp2d (deprecated, __init__ takes Never)

_interp2d_type: type[interp2d]  # pyright: ignore[reportDeprecated]

###
# lagrange

assert_type(lagrange(_f64_1d, _f64_1d), np.poly1d)
