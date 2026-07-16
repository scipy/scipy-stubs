# type-tests for `linalg/_decomp_schur.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import rsf2csf, schur

###

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

_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

type _Res2_2D[ScalarT: np.generic] = tuple[onp.Array2D[ScalarT], onp.Array2D[ScalarT]]
type _Res2_3D[ScalarT: np.generic] = tuple[onp.Array3D[ScalarT], onp.Array3D[ScalarT]]
type _Res2_ND[ScalarT: np.generic] = tuple[onp.ArrayND[ScalarT], onp.ArrayND[ScalarT]]
type _Res3_2D[ScalarT: np.generic] = tuple[onp.Array2D[ScalarT], onp.Array2D[ScalarT], int]

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

assert_type(schur(_f64_3d), _Res2_3D[np.float64])
assert_type(schur(_f64_nd), _Res2_ND[np.float64])

###
# rsf2csf

assert_type(rsf2csf(_i32_2d, _i32_2d), _Res2_ND[np.complex128])
assert_type(rsf2csf(_f32_2d, _f32_2d), _Res2_ND[np.complex64])
assert_type(rsf2csf(_f64_2d, _f64_2d), _Res2_ND[np.complex128])
assert_type(rsf2csf(_c64_2d, _c64_2d), _Res2_ND[np.complex64])
assert_type(rsf2csf(_c128_2d, _c128_2d), _Res2_ND[np.complex128])
