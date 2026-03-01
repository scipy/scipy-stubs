# type-tests for `interpolate/_ndgriddata.pyi` and `interpolate/_interpnd.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator, griddata

###

pts_2d: onp.Array2D[np.float64]
vals_f: onp.Array1D[np.float64]
vals_c: onp.Array1D[np.complex128]
xi: onp.Array2D[np.float64]

###
# NearestNDInterpolator

nn_f = NearestNDInterpolator(pts_2d, vals_f)
assert_type(nn_f, NearestNDInterpolator[np.float64])

nn_c = NearestNDInterpolator(pts_2d, vals_c)
assert_type(nn_c, NearestNDInterpolator[np.complex128])

###
# LinearNDInterpolator

ln_f = LinearNDInterpolator(pts_2d, vals_f)
assert_type(ln_f, LinearNDInterpolator[np.float64])

ln_c = LinearNDInterpolator(pts_2d, vals_c)
assert_type(ln_c, LinearNDInterpolator[np.complex128])

###
# CloughTocher2DInterpolator

ct_f = CloughTocher2DInterpolator(pts_2d, vals_f)
assert_type(ct_f, CloughTocher2DInterpolator[np.float64])

ct_c = CloughTocher2DInterpolator(pts_2d, vals_c)
assert_type(ct_c, CloughTocher2DInterpolator[np.complex128])

###
# griddata

assert_type(griddata(pts_2d, vals_f, xi), onp.Array[onp.AtLeast1D[Any], np.float64])
assert_type(griddata(pts_2d, vals_c, xi), onp.Array[onp.AtLeast1D[Any], np.complex128])
