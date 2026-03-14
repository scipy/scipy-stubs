from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import (
    BivariateSpline,
    InterpolatedUnivariateSpline,
    LSQBivariateSpline,
    LSQSphereBivariateSpline,
    LSQUnivariateSpline,
    RectBivariateSpline,
    RectSphereBivariateSpline,
    SmoothBivariateSpline,
    SmoothSphereBivariateSpline,
    UnivariateSpline,
)

x: onp.Array1D[np.float64]
y: onp.Array1D[np.float64]
x2d: onp.Array2D[np.float64]

# UnivariateSpline
us = UnivariateSpline(x, y)
assert_type(us, UnivariateSpline)
assert_type(us(x), onp.Array1D[np.float64])
assert_type(us.get_knots(), onp.Array1D[np.float64])
assert_type(us.get_coeffs(), onp.Array1D[np.float64])
assert_type(us.get_residual(), float)
assert_type(us.roots(), onp.Array1D[np.float64])
assert_type(us.derivatives(0.5), onp.Array1D[np.float64])
assert_type(us.derivative(), UnivariateSpline)
assert_type(us.antiderivative(), UnivariateSpline)
assert_type(us.integral(0.0, 1.0), float)

# InterpolatedUnivariateSpline
ius = InterpolatedUnivariateSpline(x, y)
assert_type(ius, InterpolatedUnivariateSpline)
assert_type(ius(x), onp.Array1D[np.float64])

# LSQUnivariateSpline
lus = LSQUnivariateSpline(x, y, x[2:-2])
assert_type(lus, LSQUnivariateSpline)
assert_type(lus(x), onp.Array1D[np.float64])

# SmoothBivariateSpline
sbs = SmoothBivariateSpline(x, y, y)
assert_type(sbs, SmoothBivariateSpline)
assert_type(sbs(x, y), onp.Array1D[np.float64])
assert_type(sbs.ev(x, y), onp.Array2D[np.float64])
assert_type(sbs.get_knots(), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(sbs.get_coeffs(), onp.Array1D[np.float64])
assert_type(sbs.get_residual(), float)
assert_type(sbs.integral(0.0, 1.0, 0.0, 1.0), float)

# LSQBivariateSpline
lbs = LSQBivariateSpline(x, y, y, x[2:-2], x[2:-2])
assert_type(lbs, LSQBivariateSpline)
assert_type(lbs(x, y), onp.Array1D[np.float64])
assert_type(lbs.ev(x, y), onp.Array2D[np.float64])

# RectBivariateSpline
rbs = RectBivariateSpline(x, y, x2d)
assert_type(rbs, RectBivariateSpline)
assert_type(rbs(x, y), onp.Array1D[np.float64])
assert_type(rbs.ev(x, y), onp.Array2D[np.float64])

# BivariateSpline

bvs: BivariateSpline
assert_type(bvs(x, y), onp.Array1D[np.float64])
assert_type(bvs.ev(x, y), onp.Array2D[np.float64])
assert_type(bvs.integral(0.0, 1.0, 0.0, 1.0), float)
assert_type(bvs.get_knots(), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bvs.get_coeffs(), onp.Array1D[np.float64])
assert_type(bvs.get_residual(), float)

# SmoothSphereBivariateSpline

ssbs = SmoothSphereBivariateSpline(x, y, y)
assert_type(ssbs, SmoothSphereBivariateSpline)
assert_type(ssbs(x, y), onp.Array2D[np.float64])
assert_type(ssbs.ev(x, y), onp.Array2D[np.float64])

# LSQSphereBivariateSpline

lsbs = LSQSphereBivariateSpline(x, y, y, x, x)
assert_type(lsbs, LSQSphereBivariateSpline)
assert_type(lsbs(x, y), onp.Array2D[np.float64])
assert_type(lsbs.ev(x, y), onp.Array2D[np.float64])

# RectSphereBivariateSpline

rsbs = RectSphereBivariateSpline(x, y, x2d)
assert_type(rsbs, RectSphereBivariateSpline)
assert_type(rsbs(x, y), onp.Array2D[np.float64])
assert_type(rsbs.ev(x, y), onp.Array2D[np.float64])
