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

###
# CubicHermiteSpline

chs_f = CubicHermiteSpline(x_1d, y_1d, y_1d)
assert_type(chs_f, CubicHermiteSpline[np.float64])

chs_c = CubicHermiteSpline(x_1d, y_1d_c, y_1d_c)
assert_type(chs_c, CubicHermiteSpline[np.complex128])

###
# PchipInterpolator

pchip = PchipInterpolator(x_1d, y_1d)
assert_type(pchip, PchipInterpolator)

###
# Akima1DInterpolator

akima = Akima1DInterpolator(x_1d, y_1d)
assert_type(akima, Akima1DInterpolator)

akima_makima = Akima1DInterpolator(x_1d, y_1d, method="makima")
assert_type(akima_makima, Akima1DInterpolator)

###
# CubicSpline

cs_f = CubicSpline(x_1d, y_1d)
assert_type(cs_f, CubicSpline[np.float64])

cs_c = CubicSpline(x_1d, y_1d_c)
assert_type(cs_c, CubicSpline[np.complex128])

cs_nat = CubicSpline(x_1d, y_1d, bc_type="natural")
assert_type(cs_nat, CubicSpline[np.float64])

###
# pchip_interpolate

assert_type(pchip_interpolate(x_1d, y_1d, 0.5), np.float64)
assert_type(pchip_interpolate(x_1d, y_1d, x_1d), onp.ArrayND[np.float64])
