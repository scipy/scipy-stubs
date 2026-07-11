# type-tests for `interpolate/_interpolate.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import BPoly, NdPPoly, PPoly, interp1d, interp2d, lagrange

###

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_c128_2d: onp.Array2D[np.complex128]

###
# PPoly

_ppoly_0d = PPoly(_f64_2d, _f64_1d)
assert_type(_ppoly_0d, PPoly[np.float64, tuple[()]])
assert_type(_ppoly_0d(0.5), onp.Array[tuple[()], np.float64])
assert_type(_ppoly_0d(_f64_1d), onp.ArrayND[np.float64])
assert_type(_ppoly_0d.integrate(0.0, 1.0), onp.Array[tuple[()], np.float64])
assert_type(_ppoly_0d.solve(), onp.Array1D[np.float64])
assert_type(_ppoly_0d.roots(), onp.Array1D[np.float64])

_ppoly_1d = PPoly(_f64_3d, _f64_1d)
assert_type(_ppoly_1d, PPoly[np.float64, tuple[int]])
assert_type(_ppoly_1d(0.5), onp.Array1D[np.float64])
assert_type(_ppoly_1d.integrate(0.0, 1.0), onp.Array1D[np.float64])
assert_type(_ppoly_1d.solve(), onp.Array1D[np.object_])
assert_type(_ppoly_1d.roots(), onp.Array1D[np.object_])

_ppoly_nd = PPoly(_f64_nd, _f64_1d)
assert_type(_ppoly_nd, PPoly[np.float64])
assert_type(_ppoly_nd.solve(), onp.Array1D[np.float64])

_ppoly_c = PPoly(_c128_2d, _f64_1d)
assert_type(_ppoly_c, PPoly[np.complex128, tuple[()]])
assert_type(_ppoly_c(0.5), onp.Array[tuple[()], np.complex128])
assert_type(_ppoly_c.integrate(0.0, 1.0), onp.Array[tuple[()], np.complex128])
# solve and roots raise a `ValueError` for complex-valued coefficients
# pyrefly: ignore [no-matching-overload]
_ppoly_c.solve()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
# pyrefly: ignore [no-matching-overload]
_ppoly_c.roots()  # type: ignore[misc]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

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

assert_type(lagrange(_f64_1d, _f64_1d), np.poly1d)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
