# type-tests for `spatial/transform/_rotation_spline.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation, RotationSpline

###

_times: onp.Array1D[np.float64]
_rots: Rotation

_spline = RotationSpline(_times, _rots)

assert_type(RotationSpline.MAX_ITER, int)
assert_type(RotationSpline.TOL, float)

assert_type(_spline.times, onp.Array1D[np.int32 | np.int64 | np.float32 | np.float64])
assert_type(_spline.rotations, Rotation)
assert_type(_spline.interpolator, PPoly)

assert_type(_spline(_times), Rotation | onp.ArrayND[np.float64])
assert_type(_spline(_times, order=0), Rotation | onp.ArrayND[np.float64])

assert_type(_spline(_times, order=1), onp.ArrayND[np.float64])
assert_type(_spline(_times, order=2), onp.ArrayND[np.float64])
