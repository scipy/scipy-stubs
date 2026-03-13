# type-tests for `linalg/_decomp_schur.pyi`

from typing import TypeAlias, TypeVar, assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import rsf2csf, schur

###

_bool_2d: onp.Array2D[np.bool_]
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

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_Res2ND: TypeAlias = tuple[onp.ArrayND[_ScalarT], onp.ArrayND[_ScalarT]]
_Res3ND: TypeAlias = tuple[onp.ArrayND[_ScalarT], onp.ArrayND[_ScalarT], int]

###
# schur

assert_type(schur(_bool_2d), _Res2ND[np.float32])
assert_type(schur(_i8_2d), _Res2ND[np.float64])
assert_type(schur(_i16_2d), _Res2ND[np.float64])
assert_type(schur(_i32_2d), _Res2ND[np.float64])
assert_type(schur(_i64_2d), _Res2ND[np.float64])
assert_type(schur(_f16_2d), _Res2ND[np.float32])
assert_type(schur(_f32_2d), _Res2ND[np.float32])
assert_type(schur(_f64_2d), _Res2ND[np.float64])
assert_type(schur(_f80_2d), _Res2ND[np.float64])
assert_type(schur(_c64_2d), _Res2ND[np.complex64])
assert_type(schur(_c128_2d), _Res2ND[np.complex128])
assert_type(schur(_c160_2d), _Res2ND[np.complex128])

assert_type(schur(_bool_2d, sort="lhp"), _Res3ND[np.float32])
assert_type(schur(_i8_2d, sort="lhp"), _Res3ND[np.float64])
assert_type(schur(_i16_2d, sort="lhp"), _Res3ND[np.float64])
assert_type(schur(_i32_2d, sort="lhp"), _Res3ND[np.float64])
assert_type(schur(_i64_2d, sort="lhp"), _Res3ND[np.float64])
assert_type(schur(_f16_2d, sort="lhp"), _Res3ND[np.float32])
assert_type(schur(_f32_2d, sort="lhp"), _Res3ND[np.float32])
assert_type(schur(_f64_2d, sort="lhp"), _Res3ND[np.float64])
assert_type(schur(_f80_2d, sort="lhp"), _Res3ND[np.float64])
assert_type(schur(_c64_2d, sort="lhp"), _Res3ND[np.complex64])
assert_type(schur(_c128_2d, sort="lhp"), _Res3ND[np.complex128])
assert_type(schur(_c160_2d, sort="lhp"), _Res3ND[np.complex128])

assert_type(schur(_bool_2d, output="c"), _Res2ND[np.complex64])
assert_type(schur(_i8_2d, output="c"), _Res2ND[np.complex128])
assert_type(schur(_i16_2d, output="c"), _Res2ND[np.complex128])
assert_type(schur(_i32_2d, output="c"), _Res2ND[np.complex128])
assert_type(schur(_i64_2d, output="c"), _Res2ND[np.complex128])
assert_type(schur(_f16_2d, output="c"), _Res2ND[np.complex64])
assert_type(schur(_f32_2d, output="c"), _Res2ND[np.complex64])
assert_type(schur(_f64_2d, output="c"), _Res2ND[np.complex128])
assert_type(schur(_f80_2d, output="c"), _Res2ND[np.complex128])
assert_type(schur(_c64_2d, output="c"), _Res2ND[np.complex64])
assert_type(schur(_c128_2d, output="c"), _Res2ND[np.complex128])
assert_type(schur(_c160_2d, output="c"), _Res2ND[np.complex128])

assert_type(schur(_bool_2d, output="c", sort="lhp"), _Res3ND[np.complex64])
assert_type(schur(_i8_2d, output="c", sort="lhp"), _Res3ND[np.complex128])
assert_type(schur(_i16_2d, output="c", sort="lhp"), _Res3ND[np.complex128])
assert_type(schur(_i32_2d, output="c", sort="lhp"), _Res3ND[np.complex128])
assert_type(schur(_i64_2d, output="c", sort="lhp"), _Res3ND[np.complex128])
assert_type(schur(_f16_2d, output="c", sort="lhp"), _Res3ND[np.complex64])
assert_type(schur(_f32_2d, output="c", sort="lhp"), _Res3ND[np.complex64])
assert_type(schur(_f64_2d, output="c", sort="lhp"), _Res3ND[np.complex128])
assert_type(schur(_f80_2d, output="c", sort="lhp"), _Res3ND[np.complex128])
assert_type(schur(_c64_2d, output="c", sort="lhp"), _Res3ND[np.complex64])
assert_type(schur(_c128_2d, output="c", sort="lhp"), _Res3ND[np.complex128])
assert_type(schur(_c160_2d, output="c", sort="lhp"), _Res3ND[np.complex128])

###
# rsf2csf

assert_type(rsf2csf(_i32_2d, _i32_2d), _Res2ND[np.complex128])
assert_type(rsf2csf(_f32_2d, _f32_2d), _Res2ND[np.complex64])
assert_type(rsf2csf(_f64_2d, _f64_2d), _Res2ND[np.complex128])
assert_type(rsf2csf(_c64_2d, _c64_2d), _Res2ND[np.complex64])
assert_type(rsf2csf(_c128_2d, _c128_2d), _Res2ND[np.complex128])
