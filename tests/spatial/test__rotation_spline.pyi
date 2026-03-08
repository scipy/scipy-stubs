# type-tests for `spatial/transform/_rotation_spline.pyi`

from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation, RotationSpline

###

_f64_1d: onp.Array1D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

_R_1d: Rotation[tuple[int]]
_R_nd: Rotation

###
# RotationSpline

# __init__

_spline = RotationSpline(_f64_1d, _R_1d)
_spline = RotationSpline(_f64_1d, _R_nd)
_spline = RotationSpline(_f64_nd, _R_1d)
_spline = RotationSpline(_f64_nd, _R_nd)

# class attributes

assert_type(RotationSpline.MAX_ITER, Literal[10])
assert_type(RotationSpline.TOL, float)

# instance attributes

assert_type(_spline.times, onp.Array1D[np.float64])
assert_type(_spline.rotations, Rotation[tuple[int]])
assert_type(_spline.interpolator, PPoly[np.float64])

# __call__

assert_type(_spline(1), Rotation[tuple[()]])
assert_type(_spline(1, 0), Rotation[tuple[()]])
assert_type(_spline(1, order=0), Rotation[tuple[()]])
assert_type(_spline(1, order=1), onp.Array1D[np.float64])
assert_type(_spline(1, order=2), onp.Array1D[np.float64])

assert_type(_spline(_f64_1d), Rotation[tuple[int]])
assert_type(_spline(_f64_1d, 0), Rotation[tuple[int]])
assert_type(_spline(_f64_1d, order=0), Rotation[tuple[int]])
assert_type(_spline(_f64_1d, order=1), onp.Array2D[np.float64])
assert_type(_spline(_f64_1d, order=2), onp.Array2D[np.float64])
