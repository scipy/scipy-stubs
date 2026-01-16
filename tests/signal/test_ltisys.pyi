from typing import Any, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import bode, freqresp, impulse, lsim, lti, step

###

_VecF32: TypeAlias = onp.Array1D[np.float32]
_VecF64: TypeAlias = onp.Array1D[np.float64]
_ArrF64: TypeAlias = onp.ArrayND[np.float64]
_VecC64: TypeAlias = onp.Array1D[np.complex64]
_VecC128: TypeAlias = onp.Array1D[np.complex128]
_ArrC128: TypeAlias = onp.ArrayND[np.complex128]
_VecC128ish: TypeAlias = onp.Array1D[np.complex128 | Any]
_ArrC128ish: TypeAlias = onp.ArrayND[np.complex128 | Any]

###

_f32_1d: _VecF32
_f32_2d: onp.Array2D[np.float32]
_f64_1d: _VecF64
_f64_2d: onp.Array2D[np.float64]
_c64_1d: _VecC64
_c64_2d: onp.Array2D[np.complex64]
_c128_1d: _VecC128
_c128_2d: onp.Array2D[np.complex128]

_lti_f32: lti[np.float32]
_lti_f64: lti[np.float64]
_lti_c64: lti[np.complex64]
_lti_c128: lti[np.complex128]

_tf_cont_f32: tuple[onp.ArrayND[np.float32], _VecF32]
_tf_cont_f64: tuple[_ArrF64, _VecF64]
_tf_cont_c64: tuple[onp.ArrayND[np.complex64], _VecC64]
_tf_cont_c128: tuple[_ArrC128, _VecC128]

_zpk_cont_f32: tuple[_VecF32, _VecF32, float]
_zpk_cont_f64: tuple[_VecF64, _VecF64, float]
_zpk_cont_c64: tuple[_VecC64, _VecC64, float]
_zpk_cont_c128: tuple[_VecC128, _VecC128, float]

_ss_cont_f32: tuple[onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array2D[np.float32]]
_ss_cont_f64: tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64]]
_ss_cont_c64: tuple[onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array2D[np.complex64]]
_ss_cont_c128: tuple[
    onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128]
]

###
# lsim (same as impulse and step)

# f32
assert_type(lsim(_lti_f32, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_tf_cont_f32, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_zpk_cont_f32, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_ss_cont_f32, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
# c64
assert_type(lsim(_lti_c64, None, _f64_1d), tuple[_VecF64, _VecC128, _ArrC128])
assert_type(lsim(_tf_cont_c64, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
assert_type(lsim(_zpk_cont_c64, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
assert_type(lsim(_ss_cont_c64, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
# f64
assert_type(lsim(_lti_f64, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_tf_cont_f64, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_zpk_cont_f64, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_ss_cont_f64, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
# c128
assert_type(lsim(_lti_c128, None, _f64_1d), tuple[_VecF64, _VecC128, _ArrC128])
assert_type(lsim(_tf_cont_c128, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
assert_type(lsim(_zpk_cont_c128, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
assert_type(lsim(_ss_cont_c128, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])

###
# impulse (same as lsim and step)

# f32
assert_type(impulse(_lti_f32), tuple[_VecF64, _VecF64])
assert_type(impulse(_tf_cont_f32), tuple[_VecF64, _VecF64])
assert_type(impulse(_zpk_cont_f32), tuple[_VecF64, _VecF64])
assert_type(impulse(_ss_cont_f32), tuple[_VecF64, _VecF64])
# c64
assert_type(impulse(_lti_c64), tuple[_VecF64, _VecC128])
assert_type(impulse(_tf_cont_c64), tuple[_VecF64, _VecC128ish])
assert_type(impulse(_zpk_cont_c64), tuple[_VecF64, _VecC128ish])
assert_type(impulse(_ss_cont_c64), tuple[_VecF64, _VecC128ish])
# f64
assert_type(impulse(_lti_f64), tuple[_VecF64, _VecF64])
assert_type(impulse(_tf_cont_f64), tuple[_VecF64, _VecF64])
assert_type(impulse(_zpk_cont_f64), tuple[_VecF64, _VecF64])
assert_type(impulse(_ss_cont_f64), tuple[_VecF64, _VecF64])
# c128
assert_type(impulse(_lti_c128), tuple[_VecF64, _VecC128])
assert_type(impulse(_tf_cont_c128), tuple[_VecF64, _VecC128ish])
assert_type(impulse(_zpk_cont_c128), tuple[_VecF64, _VecC128ish])
assert_type(impulse(_ss_cont_c128), tuple[_VecF64, _VecC128ish])

###
# step (same as lsim and impulse)

# f32
assert_type(step(_lti_f32), tuple[_VecF64, _VecF64])
assert_type(step(_tf_cont_f32), tuple[_VecF64, _VecF64])
assert_type(step(_zpk_cont_f32), tuple[_VecF64, _VecF64])
assert_type(step(_ss_cont_f32), tuple[_VecF64, _VecF64])
# c64
assert_type(step(_lti_c64), tuple[_VecF64, _VecC128])
assert_type(step(_tf_cont_c64), tuple[_VecF64, _VecC128ish])
assert_type(step(_zpk_cont_c64), tuple[_VecF64, _VecC128ish])
assert_type(step(_ss_cont_c64), tuple[_VecF64, _VecC128ish])
# f64
assert_type(step(_lti_f64), tuple[_VecF64, _VecF64])
assert_type(step(_tf_cont_f64), tuple[_VecF64, _VecF64])
assert_type(step(_zpk_cont_f64), tuple[_VecF64, _VecF64])
assert_type(step(_ss_cont_f64), tuple[_VecF64, _VecF64])
# c128
assert_type(step(_lti_c128), tuple[_VecF64, _VecC128])
assert_type(step(_tf_cont_c128), tuple[_VecF64, _VecC128ish])
assert_type(step(_zpk_cont_c128), tuple[_VecF64, _VecC128ish])
assert_type(step(_ss_cont_c128), tuple[_VecF64, _VecC128ish])

###
# bode

# f32
assert_type(bode(_lti_f32), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_tf_cont_f32), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_zpk_cont_f32), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_ss_cont_f32), tuple[_VecF32, _VecF32, _VecF32])
# c64
assert_type(bode(_lti_c64), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_tf_cont_c64), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_zpk_cont_c64), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_ss_cont_c64), tuple[_VecF32, _VecF32, _VecF32])
# f64
assert_type(bode(_lti_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_tf_cont_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_zpk_cont_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_ss_cont_f64), tuple[_VecF64, _VecF64, _VecF64])
# c128
assert_type(bode(_lti_c128), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_tf_cont_c128), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_zpk_cont_c128), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_ss_cont_c128), tuple[_VecF64, _VecF64, _VecF64])

###
# freqresp

# f32
assert_type(freqresp(_lti_f32), tuple[_VecF32, _VecC64])
assert_type(freqresp(_tf_cont_f32), tuple[_VecF32, _VecC64])
assert_type(freqresp(_zpk_cont_f32), tuple[_VecF32, _VecC64])
assert_type(freqresp(_ss_cont_f32), tuple[_VecF32, _VecC64])
# c64
assert_type(freqresp(_lti_c64), tuple[_VecF32, _VecC64])
assert_type(freqresp(_tf_cont_c64), tuple[_VecF32, _VecC64])
assert_type(freqresp(_zpk_cont_c64), tuple[_VecF32, _VecC64])
assert_type(freqresp(_ss_cont_c64), tuple[_VecF32, _VecC64])
# f64
assert_type(freqresp(_lti_f64), tuple[_VecF64, _VecC128])
assert_type(freqresp(_tf_cont_f64), tuple[_VecF64, _VecC128])
assert_type(freqresp(_zpk_cont_f64), tuple[_VecF64, _VecC128])
assert_type(freqresp(_ss_cont_f64), tuple[_VecF64, _VecC128])
# c128
assert_type(freqresp(_lti_c128), tuple[_VecF64, _VecC128])
assert_type(freqresp(_tf_cont_c128), tuple[_VecF64, _VecC128])
assert_type(freqresp(_zpk_cont_c128), tuple[_VecF64, _VecC128])
assert_type(freqresp(_ss_cont_c128), tuple[_VecF64, _VecC128])
