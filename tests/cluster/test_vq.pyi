from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.cluster import vq

###

_i64_1d: onp.Array1D[np.int64]
_i64_2d: onp.Array2D[np.int64]
_f32_2d: onp.Array2D[np.float32]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_c128_2d: onp.Array2D[np.complex128]

_py_f_1d: list[float]

###

# whiten
assert_type(vq.whiten(_i64_2d), onp.Array2D[np.float64])
assert_type(vq.whiten(_f64_2d), onp.Array2D[np.float64])
assert_type(vq.whiten(_c128_2d), onp.Array2D[np.complex128])
assert_type(vq.whiten(_i64_1d), onp.Array1D[np.float64])
assert_type(vq.whiten(_f64_1d), onp.Array1D[np.float64])
assert_subtype[onp.ArrayND[np.float64]](vq.whiten(_f64_nd))

# vq
assert_type(vq.vq(_f32_2d, _f32_2d), tuple[onp.Array1D[np.int32], onp.Array1D[np.float32]])
assert_type(vq.vq(_f64_2d, _f64_2d), tuple[onp.Array1D[np.int32], onp.Array1D[np.float64]])

# py_vq
assert_type(vq.py_vq(_i64_2d, _i64_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(vq.py_vq(_f32_2d, _f32_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(vq.py_vq(_f64_2d, _f64_2d), tuple[onp.Array1D[np.intp], onp.Array1D[np.float64]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

# kmeans
assert_type(vq.kmeans(_f32_2d, 2), tuple[onp.Array2D[np.float32], np.float32])
assert_type(vq.kmeans(_f64_2d, 2), tuple[onp.Array2D[np.float64], np.float64])
assert_type(vq.kmeans(_f32_1d, 2), tuple[onp.Array1D[np.float32], np.float32])
assert_type(vq.kmeans(_f64_1d, 2), tuple[onp.Array1D[np.float64], np.float64])
assert_subtype[tuple[onp.ArrayND[np.float64 | Any], np.float64 | Any]](vq.kmeans(_f64_nd, 2))

# kmeans2
assert_type(vq.kmeans2(_f32_2d, 2), tuple[onp.Array2D[np.float32], onp.Array1D[np.int32]])
assert_type(vq.kmeans2(_f64_2d, 2), tuple[onp.Array2D[np.float64], onp.Array1D[np.int32]])
assert_type(vq.kmeans2(_f32_1d, 2), tuple[onp.Array1D[np.float32], onp.Array1D[np.int32]])
assert_type(vq.kmeans2(_f64_1d, 2), tuple[onp.Array1D[np.float64], onp.Array1D[np.int32]])
assert_type(vq.kmeans2(_py_f_1d, 2), tuple[onp.Array1D[np.float64], onp.Array1D[np.int32]])
assert_subtype[tuple[onp.ArrayND[np.float64 | Any], onp.Array1D[np.int32]]](vq.kmeans2(_f64_nd, 2))
