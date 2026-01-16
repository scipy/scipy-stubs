from typing import Any, TypeAlias, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import bode, dbode, dfreqresp, dimpulse, dlsim, dlti, dstep, freqresp, impulse, lsim, lti, step
from scipy.signal._ltisys import StateSpaceDiscrete, TransferFunctionDiscrete, ZerosPolesGainDiscrete

###

_VecF32: TypeAlias = onp.Array1D[np.float32]
_VecF64: TypeAlias = onp.Array1D[np.float64]
_MatF64: TypeAlias = onp.Array2D[np.float64]
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
_dlti_f32: dlti[np.float32]
_dlti_f64: dlti[np.float64]
_dlti_c64: dlti[np.complex64]
_dlti_c128: dlti[np.complex128]
_tf_disc_f32: TransferFunctionDiscrete[np.float32]
_tf_disc_f64: TransferFunctionDiscrete[np.float64]
_zpk_disc_f32: ZerosPolesGainDiscrete[np.float32]
_zpk_disc_f64: ZerosPolesGainDiscrete[np.float64]
_ss_disc_f32: StateSpaceDiscrete[np.float32]
_ss_disc_f64: StateSpaceDiscrete[np.float64]

_to_tf_cont_f32: tuple[onp.ArrayND[np.float32], _VecF32]
_to_tf_cont_f64: tuple[_ArrF64, _VecF64]
_to_tf_cont_c64: tuple[onp.ArrayND[np.complex64], _VecC64]
_to_tf_cont_c128: tuple[_ArrC128, _VecC128]
_to_tf_disc_f32: tuple[onp.ArrayND[np.float32], _VecF32, float]
_to_tf_disc_f64: tuple[_ArrF64, _VecF64, float]

_to_zpk_cont_f32: tuple[_VecF32, _VecF32, float]
_to_zpk_cont_f64: tuple[_VecF64, _VecF64, float]
_to_zpk_cont_c64: tuple[_VecC64, _VecC64, float]
_to_zpk_cont_c128: tuple[_VecC128, _VecC128, float]
_to_zpk_disc_f32: tuple[_VecF32, _VecF32, float, float]
_to_zpk_disc_f64: tuple[_VecF64, _VecF64, float, float]

_to_ss_cont_f32: tuple[onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array2D[np.float32]]
_to_ss_cont_f64: tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64]]
_to_ss_cont_c64: tuple[onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array2D[np.complex64]]
_to_ss_cont_c128: tuple[
    onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128]
]
_to_ss_disc_f32: tuple[onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array2D[np.float32], float]
_to_ss_disc_f64: tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], float]
_to_ss_disc_c64: tuple[
    onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array2D[np.complex64], float
]
_to_ss_disc_c128: tuple[
    onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], float
]

###
# lsim (same as impulse and step)

# f32
assert_type(lsim(_lti_f32, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_to_tf_cont_f32, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_to_zpk_cont_f32, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_to_ss_cont_f32, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
# c64
assert_type(lsim(_lti_c64, None, _f64_1d), tuple[_VecF64, _VecC128, _ArrC128])
assert_type(lsim(_to_tf_cont_c64, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
assert_type(lsim(_to_zpk_cont_c64, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
assert_type(lsim(_to_ss_cont_c64, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
# f64
assert_type(lsim(_lti_f64, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_to_tf_cont_f64, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_to_zpk_cont_f64, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(lsim(_to_ss_cont_f64, None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
# c128
assert_type(lsim(_lti_c128, None, _f64_1d), tuple[_VecF64, _VecC128, _ArrC128])
assert_type(lsim(_to_tf_cont_c128, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
assert_type(lsim(_to_zpk_cont_c128, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])
assert_type(lsim(_to_ss_cont_c128, None, _f64_1d), tuple[_VecF64, _VecC128ish, _ArrC128ish])

# lti.output method
assert_type(_lti_f32.output(None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(_lti_c64.output(None, _f64_1d), tuple[_VecF64, _VecC128, _ArrC128])
assert_type(_lti_f64.output(None, _f64_1d), tuple[_VecF64, _VecF64, _ArrF64])
assert_type(_lti_c128.output(None, _f64_1d), tuple[_VecF64, _VecC128, _ArrC128])

###
# impulse (same as lsim and step)

# f32
assert_type(impulse(_lti_f32), tuple[_VecF64, _VecF64])
assert_type(impulse(_to_tf_cont_f32), tuple[_VecF64, _VecF64])
assert_type(impulse(_to_zpk_cont_f32), tuple[_VecF64, _VecF64])
assert_type(impulse(_to_ss_cont_f32), tuple[_VecF64, _VecF64])
# c64
assert_type(impulse(_lti_c64), tuple[_VecF64, _VecC128])
assert_type(impulse(_to_tf_cont_c64), tuple[_VecF64, _VecC128ish])
assert_type(impulse(_to_zpk_cont_c64), tuple[_VecF64, _VecC128ish])
assert_type(impulse(_to_ss_cont_c64), tuple[_VecF64, _VecC128ish])
# f64
assert_type(impulse(_lti_f64), tuple[_VecF64, _VecF64])
assert_type(impulse(_to_tf_cont_f64), tuple[_VecF64, _VecF64])
assert_type(impulse(_to_zpk_cont_f64), tuple[_VecF64, _VecF64])
assert_type(impulse(_to_ss_cont_f64), tuple[_VecF64, _VecF64])
# c128
assert_type(impulse(_lti_c128), tuple[_VecF64, _VecC128])
assert_type(impulse(_to_tf_cont_c128), tuple[_VecF64, _VecC128ish])
assert_type(impulse(_to_zpk_cont_c128), tuple[_VecF64, _VecC128ish])
assert_type(impulse(_to_ss_cont_c128), tuple[_VecF64, _VecC128ish])

# lti.impulse method
assert_type(_lti_f32.impulse(), tuple[_VecF64, _VecF64])
assert_type(_lti_c64.impulse(), tuple[_VecF64, _VecC128])
assert_type(_lti_f64.impulse(), tuple[_VecF64, _VecF64])
assert_type(_lti_c128.impulse(), tuple[_VecF64, _VecC128])

###
# step (same as lsim and impulse)

# f32
assert_type(step(_lti_f32), tuple[_VecF64, _VecF64])
assert_type(step(_to_tf_cont_f32), tuple[_VecF64, _VecF64])
assert_type(step(_to_zpk_cont_f32), tuple[_VecF64, _VecF64])
assert_type(step(_to_ss_cont_f32), tuple[_VecF64, _VecF64])
# c64
assert_type(step(_lti_c64), tuple[_VecF64, _VecC128])
assert_type(step(_to_tf_cont_c64), tuple[_VecF64, _VecC128ish])
assert_type(step(_to_zpk_cont_c64), tuple[_VecF64, _VecC128ish])
assert_type(step(_to_ss_cont_c64), tuple[_VecF64, _VecC128ish])
# f64
assert_type(step(_lti_f64), tuple[_VecF64, _VecF64])
assert_type(step(_to_tf_cont_f64), tuple[_VecF64, _VecF64])
assert_type(step(_to_zpk_cont_f64), tuple[_VecF64, _VecF64])
assert_type(step(_to_ss_cont_f64), tuple[_VecF64, _VecF64])
# c128
assert_type(step(_lti_c128), tuple[_VecF64, _VecC128])
assert_type(step(_to_tf_cont_c128), tuple[_VecF64, _VecC128ish])
assert_type(step(_to_zpk_cont_c128), tuple[_VecF64, _VecC128ish])
assert_type(step(_to_ss_cont_c128), tuple[_VecF64, _VecC128ish])

# lti.step method
assert_type(_lti_f32.step(), tuple[_VecF64, _VecF64])
assert_type(_lti_c64.step(), tuple[_VecF64, _VecC128])
assert_type(_lti_f64.step(), tuple[_VecF64, _VecF64])
assert_type(_lti_c128.step(), tuple[_VecF64, _VecC128])

###
# bode

# f32
assert_type(bode(_lti_f32), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_to_tf_cont_f32), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_to_zpk_cont_f32), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_to_ss_cont_f32), tuple[_VecF32, _VecF32, _VecF32])
# c64
assert_type(bode(_lti_c64), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_to_tf_cont_c64), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_to_zpk_cont_c64), tuple[_VecF32, _VecF32, _VecF32])
assert_type(bode(_to_ss_cont_c64), tuple[_VecF32, _VecF32, _VecF32])
# f64
assert_type(bode(_lti_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_to_tf_cont_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_to_zpk_cont_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_to_ss_cont_f64), tuple[_VecF64, _VecF64, _VecF64])
# c128
assert_type(bode(_lti_c128), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_to_tf_cont_c128), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_to_zpk_cont_c128), tuple[_VecF64, _VecF64, _VecF64])
assert_type(bode(_to_ss_cont_c128), tuple[_VecF64, _VecF64, _VecF64])

# lti.bode method
assert_type(_lti_f32.bode(), tuple[_VecF32, _VecF32, _VecF32])
assert_type(_lti_c64.bode(), tuple[_VecF32, _VecF32, _VecF32])
assert_type(_lti_f64.bode(), tuple[_VecF64, _VecF64, _VecF64])
assert_type(_lti_c128.bode(), tuple[_VecF64, _VecF64, _VecF64])

###
# freqresp

# f32
assert_type(freqresp(_lti_f32), tuple[_VecF32, _VecC64])
assert_type(freqresp(_to_tf_cont_f32), tuple[_VecF32, _VecC64])
assert_type(freqresp(_to_zpk_cont_f32), tuple[_VecF32, _VecC64])
assert_type(freqresp(_to_ss_cont_f32), tuple[_VecF32, _VecC64])
# c64
assert_type(freqresp(_lti_c64), tuple[_VecF32, _VecC64])
assert_type(freqresp(_to_tf_cont_c64), tuple[_VecF32, _VecC64])
assert_type(freqresp(_to_zpk_cont_c64), tuple[_VecF32, _VecC64])
assert_type(freqresp(_to_ss_cont_c64), tuple[_VecF32, _VecC64])
# f64
assert_type(freqresp(_lti_f64), tuple[_VecF64, _VecC128])
assert_type(freqresp(_to_tf_cont_f64), tuple[_VecF64, _VecC128])
assert_type(freqresp(_to_zpk_cont_f64), tuple[_VecF64, _VecC128])
assert_type(freqresp(_to_ss_cont_f64), tuple[_VecF64, _VecC128])
# c128
assert_type(freqresp(_lti_c128), tuple[_VecF64, _VecC128])
assert_type(freqresp(_to_tf_cont_c128), tuple[_VecF64, _VecC128])
assert_type(freqresp(_to_zpk_cont_c128), tuple[_VecF64, _VecC128])
assert_type(freqresp(_to_ss_cont_c128), tuple[_VecF64, _VecC128])

# lti.freqresp method
assert_type(_lti_f32.freqresp(), tuple[_VecF32, _VecC64])
assert_type(_lti_c64.freqresp(), tuple[_VecF32, _VecC64])
assert_type(_lti_f64.freqresp(), tuple[_VecF64, _VecC128])
assert_type(_lti_c128.freqresp(), tuple[_VecF64, _VecC128])

###
# dlsim

# f32
assert_type(dlsim(_tf_disc_f32, None, _f64_1d), tuple[_VecF64, _MatF64])
assert_type(dlsim(_zpk_disc_f32, None, _f64_1d), tuple[_VecF64, _MatF64])
assert_type(dlsim(_ss_disc_f32, None, _f64_1d), tuple[_VecF64, _MatF64, _MatF64])
assert_type(dlsim(_to_tf_disc_f32, None, _f64_1d), tuple[_VecF64, _MatF64])
assert_type(dlsim(_to_zpk_disc_f32, None, _f64_1d), tuple[_VecF64, _MatF64])
assert_type(dlsim(_to_ss_disc_f32, None, _f64_1d), tuple[_VecF64, _MatF64, _MatF64])
# f64
assert_type(dlsim(_tf_disc_f64, None, _f64_1d), tuple[_VecF64, _MatF64])
assert_type(dlsim(_zpk_disc_f64, None, _f64_1d), tuple[_VecF64, _MatF64])
assert_type(dlsim(_ss_disc_f64, None, _f64_1d), tuple[_VecF64, _MatF64, _MatF64])
assert_type(dlsim(_to_tf_disc_f64, None, _f64_1d), tuple[_VecF64, _MatF64])
assert_type(dlsim(_to_zpk_disc_f64, None, _f64_1d), tuple[_VecF64, _MatF64])
assert_type(dlsim(_to_ss_disc_f64, None, _f64_1d), tuple[_VecF64, _MatF64, _MatF64])

###
# dimpulse

# f32
assert_type(dimpulse(_dlti_f32), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dimpulse(_to_tf_disc_f32), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dimpulse(_to_zpk_disc_f32), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dimpulse(_to_ss_disc_f32), tuple[_VecF64, tuple[_MatF64, ...]])
# f64
assert_type(dimpulse(_dlti_f64), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dimpulse(_to_tf_disc_f64), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dimpulse(_to_zpk_disc_f64), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dimpulse(_to_ss_disc_f64), tuple[_VecF64, tuple[_MatF64, ...]])

###
# dstep

# f32
assert_type(dstep(_dlti_f32), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dstep(_to_tf_disc_f32), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dstep(_to_zpk_disc_f32), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dstep(_to_ss_disc_f32), tuple[_VecF64, tuple[_MatF64, ...]])
# f64
assert_type(dstep(_dlti_f64), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dstep(_to_tf_disc_f64), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dstep(_to_zpk_disc_f64), tuple[_VecF64, tuple[_MatF64, ...]])
assert_type(dstep(_to_ss_disc_f64), tuple[_VecF64, tuple[_MatF64, ...]])

###
# dbode

# f32
assert_type(dbode(_dlti_f32), tuple[_VecF64, _VecF64, _VecF64])
assert_type(dbode(_to_tf_disc_f32), tuple[_VecF64, _VecF64, _VecF64])
assert_type(dbode(_to_zpk_disc_f32), tuple[_VecF64, _VecF64, _VecF64])
assert_type(dbode(_to_ss_disc_f32), tuple[_VecF64, _VecF64, _VecF64])
# f64
assert_type(dbode(_dlti_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(dbode(_to_tf_disc_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(dbode(_to_zpk_disc_f64), tuple[_VecF64, _VecF64, _VecF64])
assert_type(dbode(_to_ss_disc_f64), tuple[_VecF64, _VecF64, _VecF64])

###
# dfreqresp

# f32
assert_type(dfreqresp(_dlti_f32), tuple[_VecF64, _VecC128])
assert_type(dfreqresp(_to_tf_disc_f32), tuple[_VecF64, _VecC128])
assert_type(dfreqresp(_to_zpk_disc_f32), tuple[_VecF64, _VecC128])
assert_type(dfreqresp(_to_ss_disc_f32), tuple[_VecF64, _VecC128])
# f64
assert_type(dfreqresp(_dlti_f64), tuple[_VecF64, _VecC128])
assert_type(dfreqresp(_to_tf_disc_f64), tuple[_VecF64, _VecC128])
assert_type(dfreqresp(_to_zpk_disc_f64), tuple[_VecF64, _VecC128])
assert_type(dfreqresp(_to_ss_disc_f64), tuple[_VecF64, _VecC128])
