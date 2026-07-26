# type-tests for `linalg/_misc.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import bandwidth, norm

###

py_i: int
py_f: float
py_c: complex

f32: np.float32
f80: np.float128

f32_2d: onp.Array2D[np.float32]
f64_2d: onp.Array2D[np.float64]
c128_2d: onp.Array2D[np.complex128]

f64_3d: onp.Array3D[np.float64]
c128_3d: onp.Array3D[np.complex128]

b1_nd: onp.ArrayND[np.bool]
f16_nd: onp.ArrayND[np.float16]
f64_nd: onp.ArrayND[np.float64]
f32_nd: onp.ArrayND[np.float32]
f80_nd: onp.ArrayND[np.float128]
i32_nd: onp.ArrayND[np.int32]
c64_nd: onp.ArrayND[np.complex64]
c128_nd: onp.ArrayND[np.complex128]
c256_nd: onp.ArrayND[np.complex256]

py_f_1d: list[float]
py_f_2d: list[list[float]]

###
# bandwidth

assert_type(bandwidth(f64_2d), tuple[np.int64, np.int64])
assert_type(bandwidth(f64_3d), tuple[onp.Array1D[np.int64], onp.Array1D[np.int64]])
assert_type(bandwidth(c128_2d), tuple[np.int64, np.int64])
assert_type(bandwidth(c128_3d), tuple[onp.Array1D[np.int64], onp.Array1D[np.int64]])

###
# norm

assert_type(norm(py_i), np.float64)
assert_type(norm(py_f), np.float64)
assert_type(norm(py_c), np.float64)
assert_type(norm(f32), np.float32)
assert_type(norm(f80), np.float64 | Any)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

assert_type(norm(i32_nd), np.float64)
assert_type(norm(f64_nd), float)
assert_type(norm(c128_nd), float)
assert_type(norm(py_f_1d), float)
assert_type(norm(py_f_2d), float)
assert_type(norm(f32_nd), float | np.float32)
assert_type(norm(c64_nd), float | np.float32)
assert_type(norm(b1_nd), np.float64 | Any)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(norm(f16_nd), np.float64 | Any)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(norm(f80_nd), np.float64 | Any)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(norm(c256_nd), np.float64 | Any)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

assert_type(norm(f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(norm(f64_nd, None, 0, keepdims=True), onp.ArrayND[np.float64])
assert_type(norm(i32_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(norm(py_f_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(norm(f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(norm(c64_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(norm(f80_nd, keepdims=True), onp.ArrayND[np.float64 | Any])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(norm(c256_nd, keepdims=True), onp.ArrayND[np.float64 | Any])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

assert_type(norm(f64_2d, keepdims=True), onp.Array2D[np.float64])
assert_type(norm(f32_2d, keepdims=True), onp.Array2D[np.float32])
