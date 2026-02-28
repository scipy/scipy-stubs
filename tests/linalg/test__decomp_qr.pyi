# type-tests for `linalg/_decomp_qr.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.linalg import qr, qr_multiply, rq

###

_Int1D: TypeAlias = onp.Array1D[np.int32 | np.int64]
_IntND: TypeAlias = onp.ArrayND[np.int32 | np.int64]
_Float1D: TypeAlias = onp.Array1D[npc.floating]
_Float2D: TypeAlias = onp.Array2D[npc.floating]
_FloatND: TypeAlias = onp.ArrayND[npc.floating]
_Inexact1D: TypeAlias = onp.Array1D[npc.inexact]
_Inexact2D: TypeAlias = onp.Array2D[npc.inexact]
_InexactND: TypeAlias = onp.ArrayND[npc.inexact]

###

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

###
# qr

assert_type(qr(f64_2d), tuple[_FloatND, _FloatND])
assert_type(qr(f64_2d, pivoting=True), tuple[_FloatND, _FloatND, _IntND])
assert_type(qr(f64_2d, False, None, "r"), tuple[_FloatND])
assert_type(qr(f64_2d, False, None, "r", True), tuple[_FloatND, _IntND])
assert_type(qr(f64_2d, False, None, "raw"), tuple[tuple[_FloatND, _FloatND], _FloatND])
assert_type(qr(f64_2d, False, None, "raw", True), tuple[tuple[_FloatND, _FloatND], _FloatND, _IntND])

assert_type(qr(c128_2d), tuple[_InexactND, _InexactND])
assert_type(qr(c128_2d, pivoting=True), tuple[_InexactND, _InexactND, _IntND])
assert_type(qr(c128_2d, False, None, "r"), tuple[_InexactND])
assert_type(qr(c128_2d, False, None, "r", True), tuple[_InexactND, _IntND])
assert_type(qr(c128_2d, False, None, "raw"), tuple[tuple[_InexactND, _InexactND], _InexactND])
assert_type(qr(c128_2d, False, None, "raw", True), tuple[tuple[_InexactND, _InexactND], _InexactND, _IntND])

###
# qr_multiply

assert_type(qr_multiply(f64_2d, f64_1d), tuple[_Float1D, _Inexact2D])
assert_type(qr_multiply(f64_2d, f64_2d), tuple[_Float2D, _Inexact2D])
assert_type(qr_multiply(f64_3d, f64_3d), tuple[_FloatND, _InexactND])
assert_type(qr_multiply(f64_2d, f64_1d, "right", True), tuple[_Float1D | _Float2D, _Float2D, _Int1D])
assert_type(qr_multiply(f64_3d, f64_3d, pivoting=True), tuple[_FloatND, _FloatND, _IntND])

assert_type(qr_multiply(c128_2d, c128_1d), tuple[_Inexact1D | _Inexact2D, _Inexact2D])
assert_type(qr_multiply(c128_3d, c128_3d), tuple[_InexactND, _InexactND])
assert_type(qr_multiply(c128_2d, c128_1d, "right", True), tuple[_Inexact1D | _Inexact2D, _Inexact2D, _Int1D])
assert_type(qr_multiply(c128_3d, c128_3d, pivoting=True), tuple[_InexactND, _InexactND, _IntND])

###
# rq

assert_type(rq(f64_2d), tuple[_Float2D, _Float2D])
assert_type(rq(f64_3d), tuple[_FloatND, _FloatND])
assert_type(rq(f64_2d, False, None, "r"), _Float2D)
assert_type(rq(f64_3d, False, None, "r"), _FloatND)
assert_type(rq(f64_2d, mode="r"), _Float2D)
assert_type(rq(f64_3d, mode="r"), _FloatND)

assert_type(rq(c128_2d), tuple[_Inexact2D, _Inexact2D])
assert_type(rq(c128_3d), tuple[_InexactND, _InexactND])
assert_type(rq(c128_2d, False, None, "r"), _Inexact2D)
assert_type(rq(c128_3d, False, None, "r"), _InexactND)
assert_type(rq(c128_2d, mode="r"), _Inexact2D)
assert_type(rq(c128_3d, mode="r"), _InexactND)
