# type-tests for `linalg/_decomp_qr.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import qr, qr_multiply, rq

###

type _Int1D = onp.Array1D[np.int32 | np.int64]
type _IntND = onp.ArrayND[np.int32 | np.int64]
type _Float1D = onp.Array1D[npc.floating]
type _Float2D = onp.Array2D[npc.floating]
type _FloatND = onp.ArrayND[npc.floating]
type _Inexact1D = onp.Array1D[npc.inexact]
type _Inexact2D = onp.Array2D[npc.inexact]
type _InexactND = onp.ArrayND[npc.inexact]

###

_bool_2d: onp.Array2D[np.bool]
_i8_2d: onp.Array2D[np.int8]
_i32_2d: onp.Array2D[np.int32]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f32_3d: onp.Array3D[np.float32]
_f64_3d: onp.Array3D[np.float64]
_f80_2d: onp.Array2D[np.float128]
_c64_2d: onp.Array2D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]
_c128_3d: onp.Array3D[np.complex128]
_c160_2d: onp.Array2D[np.complex256]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

###
# qr

assert_type(qr(_i8_2d), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(qr(_f32_2d), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(qr(_i32_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(qr(_f64_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(qr(_f64_3d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(qr(_py_f_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(qr(_c64_2d), tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64]])
assert_type(qr(_c128_2d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(qr(_py_c_2d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

assert_type(qr(_f64_2d, mode="economic"), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(qr(_f64_2d, pivoting=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64], onp.ArrayND[np.int32]])
assert_type(qr(_f64_2d, False, mode="r"), tuple[onp.ArrayND[np.float64]])
assert_type(qr(_f64_2d, False, mode="r", pivoting=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.int32]])
assert_type(
    qr(_f64_2d, False, mode="raw"), tuple[tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]], onp.ArrayND[np.float64]]
)
assert_type(
    qr(_f64_2d, False, mode="raw", pivoting=True),
    tuple[tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]], onp.ArrayND[np.float64], onp.ArrayND[np.int32]],
)

assert_type(qr(_c64_2d, pivoting=True), tuple[onp.ArrayND[np.complex64], onp.ArrayND[np.complex64], onp.ArrayND[np.int32]])
assert_type(qr(_c128_2d, mode="economic"), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(qr(_c128_2d, pivoting=True), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128], onp.ArrayND[np.int32]])
assert_type(qr(_c128_2d, False, mode="r"), tuple[onp.ArrayND[np.complex128]])
assert_type(qr(_c128_2d, False, mode="r", pivoting=True), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.int32]])
assert_type(
    qr(_c128_2d, False, mode="raw"),
    tuple[tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]], onp.ArrayND[np.complex128]],
)
assert_type(
    qr(_c128_2d, False, mode="raw", pivoting=True),
    tuple[tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]], onp.ArrayND[np.complex128], onp.ArrayND[np.int32]],
)

assert_type(qr(_bool_2d), tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(qr(_f16_2d), tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(qr(_f80_2d), tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(qr(_c160_2d), tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(qr(_f80_2d, False, mode="r", pivoting=True), tuple[onp.ArrayND[np.float64 | Any], onp.ArrayND[np.int32]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]

###
# qr_multiply

assert_type(qr_multiply(_f64_2d, _f64_1d), tuple[_Float1D, _Inexact2D])
assert_type(qr_multiply(_f64_2d, _f64_2d), tuple[_Float2D, _Inexact2D])
assert_type(qr_multiply(_f64_3d, _f64_3d), tuple[_FloatND, _InexactND])
assert_type(qr_multiply(_f64_2d, _f64_1d, "right", True), tuple[_Float1D | _Float2D, _Float2D, _Int1D])
assert_type(qr_multiply(_f64_3d, _f64_3d, pivoting=True), tuple[_FloatND, _FloatND, _IntND])

assert_type(qr_multiply(_c128_2d, _c128_1d), tuple[_Inexact1D | _Inexact2D, _Inexact2D])
assert_type(qr_multiply(_c128_3d, _c128_3d), tuple[_InexactND, _InexactND])
assert_type(qr_multiply(_c128_2d, _c128_1d, "right", True), tuple[_Inexact1D | _Inexact2D, _Inexact2D, _Int1D])
assert_type(qr_multiply(_c128_3d, _c128_3d, pivoting=True), tuple[_InexactND, _InexactND, _IntND])

###
# rq

assert_type(rq(_i8_2d), tuple[onp.Array2D[np.float32], onp.Array2D[np.float32]])
assert_type(rq(_f32_2d), tuple[onp.Array2D[np.float32], onp.Array2D[np.float32]])
assert_type(rq(_i32_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(rq(_f64_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(rq(_py_f_2d), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(rq(_c64_2d), tuple[onp.Array2D[np.complex64], onp.Array2D[np.complex64]])
assert_type(rq(_c128_2d), tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128]])
assert_type(rq(_py_c_2d), tuple[onp.Array2D[np.complex128], onp.Array2D[np.complex128]])

assert_type(rq(_f32_3d), tuple[onp.ArrayND[np.float32], onp.ArrayND[np.float32]])
assert_type(rq(_f64_3d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(rq(_c128_3d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])

assert_type(rq(_f64_2d, mode="economic"), tuple[onp.Array2D[np.float64], onp.Array2D[np.float64]])
assert_type(rq(_f64_2d, mode="r"), onp.Array2D[np.float64])
assert_type(rq(_f64_3d, mode="r"), onp.ArrayND[np.float64])
assert_type(rq(_f32_2d, mode="r"), onp.Array2D[np.float32])
assert_type(rq(_c64_2d, mode="r"), onp.Array2D[np.complex64])
assert_type(rq(_c128_2d, mode="r"), onp.Array2D[np.complex128])
assert_type(rq(_c128_3d, mode="r"), onp.ArrayND[np.complex128])

assert_type(rq(_bool_2d), tuple[onp.Array2D[np.float64 | Any], onp.Array2D[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(rq(_f16_2d), tuple[onp.Array2D[np.float64 | Any], onp.Array2D[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(rq(_f80_2d), tuple[onp.Array2D[np.float64 | Any], onp.Array2D[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(rq(_c160_2d), tuple[onp.Array2D[np.float64 | Any], onp.Array2D[np.float64 | Any]])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
assert_type(rq(_f80_2d, mode="r"), onp.Array2D[np.float64 | Any])  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
