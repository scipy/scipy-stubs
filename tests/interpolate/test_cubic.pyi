# type-tests for `interpolate/_cubic.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import Akima1DInterpolator, CubicHermiteSpline, CubicSpline, PchipInterpolator, pchip_interpolate

###

x_1d: onp.Array1D[np.float64]
y_1d: onp.Array1D[np.float64]
y_1d_c: onp.Array1D[np.complex128]
y_2d: onp.Array2D[np.float64]
y_nd: onp.ArrayND[np.float64]

###
# CubicHermiteSpline

chs_f = CubicHermiteSpline(x_1d, y_1d, y_1d)
assert_type(chs_f, CubicHermiteSpline[np.float64, tuple[()]])

chs_c = CubicHermiteSpline(x_1d, y_1d_c, y_1d_c)
assert_type(chs_c, CubicHermiteSpline[np.complex128, tuple[()]])

chs_2d = CubicHermiteSpline(x_1d, y_2d, y_2d)
assert_type(chs_2d, CubicHermiteSpline[np.float64, tuple[int]])

chs_nd = CubicHermiteSpline(x_1d, y_nd, y_nd)
assert_type(chs_nd, CubicHermiteSpline[np.float64])

###
# PchipInterpolator

pchip = PchipInterpolator(x_1d, y_1d)
assert_type(pchip, PchipInterpolator[tuple[()]])

pchip_2d = PchipInterpolator(x_1d, y_2d)
assert_type(pchip_2d, PchipInterpolator[tuple[int]])

###
# Akima1DInterpolator

akima = Akima1DInterpolator(x_1d, y_1d)
assert_type(akima, Akima1DInterpolator[tuple[()]])

akima_makima = Akima1DInterpolator(x_1d, y_1d, method="makima")
assert_type(akima_makima, Akima1DInterpolator[tuple[()]])

###
# CubicSpline

cs_f = CubicSpline(x_1d, y_1d)
assert_type(cs_f, CubicSpline[np.float64, tuple[()]])
assert_type(cs_f(0.5), onp.Array[tuple[()], np.float64])
assert_type(cs_f.integrate(0.0, 1.0), onp.Array[tuple[()], np.float64])
assert_type(cs_f.solve(), onp.Array1D[np.float64])
assert_type(cs_f.roots(), onp.Array1D[np.float64])

cs_c = CubicSpline(x_1d, y_1d_c)
assert_type(cs_c, CubicSpline[np.complex128, tuple[()]])

cs_nat = CubicSpline(x_1d, y_1d, bc_type="natural")
assert_type(cs_nat, CubicSpline[np.float64, tuple[()]])

cs_2d = CubicSpline(x_1d, y_2d)
assert_type(cs_2d, CubicSpline[np.float64, tuple[int]])
assert_type(cs_2d(0.5), onp.Array1D[np.float64])
assert_type(cs_2d.solve(), onp.Array1D[np.object_])
assert_type(cs_2d.roots(), onp.Array1D[np.object_])

cs_nd = CubicSpline(x_1d, y_nd)
assert_type(cs_nd, CubicSpline[np.float64])
assert_type(cs_nd.solve(), onp.Array1D[np.float64])

###
# pchip_interpolate

assert_type(pchip_interpolate(x_1d, y_1d, 0.5), np.float64)
assert_type(pchip_interpolate(x_1d, y_1d, x_1d), onp.ArrayND[np.float64])
