# type-tests for `signal/_czt.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import CZT, ZoomFFT, czt, czt_points, zoom_fft

###

_Complex: TypeAlias = np.complex128 | np.clongdouble

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###

# CZT

_czt_transform = CZT(16)
assert_type(_czt_transform.m, int)
assert_type(_czt_transform.n, int)
assert_type(_czt_transform.points(), onp.Array1D[np.complex128])
assert_type(_czt_transform(_f64_1d), onp.Array1D[_Complex])
assert_type(_czt_transform(_f64_2d), onp.Array2D[_Complex])
assert_type(_czt_transform(_f64_3d), onp.Array3D[_Complex])
assert_type(_czt_transform(_f64_nd), onp.ArrayND[_Complex])

# ZoomFFT

_zoom_transform = ZoomFFT(16, 0.5)
assert_type(_zoom_transform(_f64_1d), onp.Array1D[_Complex])
assert_type(_zoom_transform(_f64_2d), onp.Array2D[_Complex])
assert_type(_zoom_transform(_f64_3d), onp.Array3D[_Complex])
assert_type(_zoom_transform(_f64_nd), onp.ArrayND[_Complex])

# czt_points

assert_type(czt_points(16), onp.Array1D[np.complex128])

# czt

assert_type(czt(_f64_1d), onp.Array1D[_Complex])
assert_type(czt(_f64_2d), onp.Array2D[_Complex])
assert_type(czt(_f64_3d), onp.Array3D[_Complex])
assert_type(czt(_f64_nd), onp.ArrayND[_Complex])

# zoom_fft

assert_type(zoom_fft(_f64_1d, 0.5), onp.Array1D[_Complex])
assert_type(zoom_fft(_f64_2d, 0.5), onp.Array2D[_Complex])
assert_type(zoom_fft(_f64_3d, 0.5), onp.Array3D[_Complex])
assert_type(zoom_fft(_f64_nd, 0.5), onp.ArrayND[_Complex])
