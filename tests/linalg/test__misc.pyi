# type-tests for `linalg/_misc.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import norm

###

py_i: int
py_f: float
py_c: complex

f32: np.float32
f80: np.float128

f64_nd: onp.ArrayND[np.float64]
f32_nd: onp.ArrayND[np.float32]
f80_nd: onp.ArrayND[np.float128]
i32_nd: onp.ArrayND[np.int32]
c128_nd: onp.ArrayND[np.complex128]

py_f_1d: list[float]
py_f_2d: list[list[float]]

###
# norm

assert_type(norm(py_i), np.float64)
assert_type(norm(py_f), np.float64)
assert_type(norm(py_c), np.float64)
assert_type(norm(f32), np.float32)
assert_type(norm(f80), np.longdouble)

assert_type(norm(f64_nd), np.float64)
assert_type(norm(i32_nd), np.float64)
assert_type(norm(f32_nd), np.float64)
assert_type(norm(f80_nd), np.float64)
assert_type(norm(c128_nd), np.float64)
assert_type(norm(py_f_1d), np.float64)
assert_type(norm(py_f_2d), np.float64)

assert_type(norm(f64_nd, None, None, True), onp.ArrayND[np.float64])
assert_type(norm(f64_nd, None, 0, True), onp.ArrayND[np.float64])
assert_type(norm(f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(norm(f32_nd, None, None, True), onp.ArrayND[np.float32])
assert_type(norm(f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(norm(f80_nd, None, None, True), onp.ArrayND[np.longdouble])
assert_type(norm(f80_nd, keepdims=True), onp.ArrayND[np.longdouble])
