# type-tests for `linalg/_decomp_schur.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.linalg import rsf2csf, schur

###

_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

_bool_2d: onp.Array2D[np.bool]
_i8_2d: onp.Array2D[np.int8]
_i16_2d: onp.Array2D[np.int16]
_i32_2d: onp.Array2D[np.int32]
_i64_2d: onp.Array2D[np.int64]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f80_2d: onp.Array2D[np.float128]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]
_c160_2d: onp.Array2D[np.complex256]

_f32_3d: onp.Array3D[np.float32]
_f64_3d: onp.Array3D[np.float64]
_c64_3d: onp.Array3D[np.complex64]
_c128_3d: onp.Array3D[np.complex128]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]
_c64_nd: onp.ArrayND[np.complex64]
_c128_nd: onp.ArrayND[np.complex128]
_f_nd: onp.ArrayND[np.float32 | np.float64]

def _sort(x: complex, /) -> bool: ...

type _Res2_2D[ScalarT: np.generic] = tuple[onp.Array2D[ScalarT], onp.Array2D[ScalarT]]
type _Res2_3D[ScalarT: np.generic] = tuple[onp.Array3D[ScalarT], onp.Array3D[ScalarT]]
type _Res2_ND[ScalarT: np.generic] = tuple[onp.ArrayND[ScalarT], onp.ArrayND[ScalarT]]
type _Res3_2D[ScalarT: np.generic] = tuple[onp.Array2D[ScalarT], onp.Array2D[ScalarT], int]
type _Res3_3D[ScalarT: np.generic] = tuple[onp.Array3D[ScalarT], onp.Array3D[ScalarT], onp.ArrayND[np.int64]]
type _Res3_ND[ScalarT: np.generic] = tuple[onp.ArrayND[ScalarT], onp.ArrayND[ScalarT], int | Any]

###
# schur

assert_type(schur(_bool_2d), _Res2_2D[np.float32])
assert_type(schur(_i8_2d), _Res2_2D[np.float64])
assert_type(schur(_i16_2d), _Res2_2D[np.float64])
assert_type(schur(_i32_2d), _Res2_2D[np.float64])
assert_type(schur(_i64_2d), _Res2_2D[np.float64])
assert_type(schur(_f16_2d), _Res2_2D[np.float32])
assert_type(schur(_f32_2d), _Res2_2D[np.float32])
assert_type(schur(_f64_2d), _Res2_2D[np.float64])
assert_type(schur(_f80_2d), _Res2_2D[np.float64])
assert_type(schur(_c64_2d), _Res2_2D[np.complex64])
assert_type(schur(_c128_2d), _Res2_2D[np.complex128])
assert_type(schur(_c160_2d), _Res2_2D[np.complex128])

assert_type(schur(_bool_2d, sort="lhp"), _Res3_2D[np.float32])
assert_type(schur(_i8_2d, sort="lhp"), _Res3_2D[np.float64])
assert_type(schur(_i16_2d, sort="lhp"), _Res3_2D[np.float64])
assert_type(schur(_i32_2d, sort="lhp"), _Res3_2D[np.float64])
assert_type(schur(_i64_2d, sort="lhp"), _Res3_2D[np.float64])
assert_type(schur(_f16_2d, sort="lhp"), _Res3_2D[np.float32])
assert_type(schur(_f32_2d, sort="lhp"), _Res3_2D[np.float32])
assert_type(schur(_f64_2d, sort="lhp"), _Res3_2D[np.float64])
assert_type(schur(_f80_2d, sort="lhp"), _Res3_2D[np.float64])
assert_type(schur(_c64_2d, sort="lhp"), _Res3_2D[np.complex64])
assert_type(schur(_c128_2d, sort="lhp"), _Res3_2D[np.complex128])
assert_type(schur(_c160_2d, sort="lhp"), _Res3_2D[np.complex128])

assert_type(schur(_bool_2d, output="c"), _Res2_2D[np.complex64])
assert_type(schur(_i8_2d, output="c"), _Res2_2D[np.complex128])
assert_type(schur(_i16_2d, output="c"), _Res2_2D[np.complex128])
assert_type(schur(_i32_2d, output="c"), _Res2_2D[np.complex128])
assert_type(schur(_i64_2d, output="c"), _Res2_2D[np.complex128])
assert_type(schur(_f16_2d, output="c"), _Res2_2D[np.complex64])
assert_type(schur(_f32_2d, output="c"), _Res2_2D[np.complex64])
assert_type(schur(_f64_2d, output="c"), _Res2_2D[np.complex128])
assert_type(schur(_f80_2d, output="c"), _Res2_2D[np.complex128])
assert_type(schur(_c64_2d, output="c"), _Res2_2D[np.complex64])
assert_type(schur(_c128_2d, output="c"), _Res2_2D[np.complex128])
assert_type(schur(_c160_2d, output="c"), _Res2_2D[np.complex128])

assert_type(schur(_bool_2d, output="c", sort="lhp"), _Res3_2D[np.complex64])
assert_type(schur(_i8_2d, output="c", sort="lhp"), _Res3_2D[np.complex128])
assert_type(schur(_i16_2d, output="c", sort="lhp"), _Res3_2D[np.complex128])
assert_type(schur(_i32_2d, output="c", sort="lhp"), _Res3_2D[np.complex128])
assert_type(schur(_i64_2d, output="c", sort="lhp"), _Res3_2D[np.complex128])
assert_type(schur(_f16_2d, output="c", sort="lhp"), _Res3_2D[np.complex64])
assert_type(schur(_f32_2d, output="c", sort="lhp"), _Res3_2D[np.complex64])
assert_type(schur(_f64_2d, output="c", sort="lhp"), _Res3_2D[np.complex128])
assert_type(schur(_f80_2d, output="c", sort="lhp"), _Res3_2D[np.complex128])
assert_type(schur(_c64_2d, output="c", sort="lhp"), _Res3_2D[np.complex64])
assert_type(schur(_c128_2d, output="c", sort="lhp"), _Res3_2D[np.complex128])
assert_type(schur(_c160_2d, output="c", sort="lhp"), _Res3_2D[np.complex128])

assert_type(schur(_f64_2d, output="c", sort=_sort), _Res3_2D[np.complex128])
assert_type(schur(_c128_2d, sort=_sort), _Res3_2D[np.complex128])

assert_type(schur(_f64_3d), _Res2_3D[np.float64])
assert_type(schur(_f64_nd), _Res2_ND[np.float64])
assert_type(schur(_f32_3d, sort="lhp"), _Res3_3D[np.float32])
assert_type(schur(_f64_3d, sort="lhp"), _Res3_3D[np.float64])
assert_type(schur(_c64_3d, sort="lhp"), _Res3_3D[np.complex64])
assert_type(schur(_c128_3d, sort="lhp"), _Res3_3D[np.complex128])
assert_type(schur(_f32_3d, output="c", sort="lhp"), _Res3_3D[np.complex64])
assert_type(schur(_f64_3d, output="c", sort="lhp"), _Res3_3D[np.complex128])
assert_type(schur(_py_f_2d, sort="lhp"), _Res3_ND[np.float64])
assert_type(schur(_py_f_2d, output="c", sort="lhp"), _Res3_ND[np.complex128])
assert_type(schur(_py_c_2d, sort="lhp"), _Res3_ND[np.complex128])
assert_subtype[_Res3_ND[np.float32]](schur(_f32_nd, sort="lhp"))
assert_subtype[_Res3_ND[np.float64]](schur(_f64_nd, sort="lhp"))
assert_subtype[_Res3_ND[np.complex64]](schur(_c64_nd, sort="lhp"))
assert_subtype[_Res3_ND[np.complex128]](schur(_c128_nd, sort="lhp"))
assert_subtype[_Res3_ND[np.complex64]](schur(_f32_nd, output="c", sort="lhp"))
assert_subtype[_Res3_ND[np.complex128]](schur(_f64_nd, output="c", sort="lhp"))
assert_type(schur(_f_nd), _Res2_ND[np.float64 | Any])
assert_type(schur(_f_nd, sort="lhp"), _Res3_ND[np.float64 | Any])

###
# rsf2csf

assert_type(rsf2csf(_i32_2d, _i32_2d), _Res2_ND[np.complex128])
assert_type(rsf2csf(_f32_2d, _f32_2d), _Res2_ND[np.complex64])
assert_type(rsf2csf(_f64_2d, _f64_2d), _Res2_ND[np.complex128])
assert_type(rsf2csf(_c64_2d, _c64_2d), _Res2_ND[np.complex64])
assert_type(rsf2csf(_c128_2d, _c128_2d), _Res2_ND[np.complex128])
