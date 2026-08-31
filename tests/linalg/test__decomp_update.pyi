# type-tests for `linalg/_decomp_update.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import qr_delete, qr_insert, qr_update

###

f32_nd: onp.ArrayND[np.float32]
f64_nd: onp.ArrayND[np.float64]
c64_nd: onp.ArrayND[np.complex64]
c128_nd: onp.ArrayND[np.complex128]

py_float_2d: list[list[float]]
py_complex_2d: list[list[complex]]

type _QRf32 = tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]]
type _QRf64 = tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
type _QRc64 = tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]]
type _QRc128 = tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]]

###
# qr_delete

assert_type(qr_delete(f32_nd, f32_nd, 0), _QRf32)
assert_type(qr_delete(f64_nd, f64_nd, 0), _QRf64)
assert_type(qr_delete(c64_nd, c64_nd, 0), _QRc64)
assert_type(qr_delete(c128_nd, c128_nd, 0), _QRc128)
assert_type(qr_delete(py_float_2d, py_float_2d, 0), _QRf64)
assert_type(qr_delete(py_complex_2d, py_complex_2d, 0), _QRc128)

###
# qr_insert

assert_type(qr_insert(f32_nd, f32_nd, f32_nd, 0), _QRf32)
assert_type(qr_insert(f64_nd, f64_nd, f64_nd, 0), _QRf64)
assert_type(qr_insert(c64_nd, c64_nd, c64_nd, 0), _QRc64)
assert_type(qr_insert(c128_nd, c128_nd, c128_nd, 0), _QRc128)
assert_type(qr_insert(py_float_2d, py_float_2d, py_float_2d, 0), _QRf64)
assert_type(qr_insert(py_complex_2d, py_complex_2d, py_complex_2d, 0), _QRc128)

###
# qr_update

assert_type(qr_update(f32_nd, f32_nd, f32_nd, f32_nd), _QRf32)
assert_type(qr_update(f64_nd, f64_nd, f64_nd, f64_nd), _QRf64)
assert_type(qr_update(c64_nd, c64_nd, c64_nd, c64_nd), _QRc64)
assert_type(qr_update(c128_nd, c128_nd, c128_nd, c128_nd), _QRc128)
assert_type(qr_update(py_float_2d, py_float_2d, py_float_2d, py_float_2d), _QRf64)
assert_type(qr_update(py_complex_2d, py_complex_2d, py_complex_2d, py_complex_2d), _QRc128)
