from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import bode, freqresp, impulse, lti, step

###

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c64_2d: onp.Array2D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c128_2d: onp.Array2D[np.complex128]

_lti_f32: lti[np.float32]
_lti_f64: lti[np.float64]
_lti_c64: lti[np.complex64]
_lti_c128: lti[np.complex128]

_tf_cont_f32: tuple[onp.ArrayND[np.float32], onp.Array1D[np.float32]]
_tf_cont_f64: tuple[onp.ArrayND[np.float64], onp.Array1D[np.float64]]
_tf_cont_c64: tuple[onp.ArrayND[np.complex64], onp.Array1D[np.complex64]]
_tf_cont_c128: tuple[onp.ArrayND[np.complex128], onp.Array1D[np.complex128]]

_zpk_cont_f32: tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], float]
_zpk_cont_f64: tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], float]
_zpk_cont_c64: tuple[onp.Array1D[np.complex64], onp.Array1D[np.complex64], float]
_zpk_cont_c128: tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128], float]

_ss_cont_f32: tuple[onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array2D[np.float32], onp.Array2D[np.float32]]
_ss_cont_f64: tuple[onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64], onp.Array2D[np.float64]]
_ss_cont_c64: tuple[onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array2D[np.complex64], onp.Array2D[np.complex64]]
_ss_cont_c128: tuple[
    onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128], onp.Array2D[np.complex128]
]

###
# impulse (same as step)

# f32
assert_type(impulse(_lti_f32), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(impulse(_tf_cont_f32), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(impulse(_zpk_cont_f32), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(impulse(_ss_cont_f32), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
# c64
assert_type(impulse(_lti_c64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(impulse(_tf_cont_c64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
assert_type(impulse(_zpk_cont_c64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
assert_type(impulse(_ss_cont_c64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
# f64
assert_type(impulse(_lti_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(impulse(_tf_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(impulse(_zpk_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(impulse(_ss_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
# c128
assert_type(impulse(_lti_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(impulse(_tf_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
assert_type(impulse(_zpk_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
assert_type(impulse(_ss_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])

###
# step (same as impulse)

# f32
assert_type(step(_lti_f32), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(step(_tf_cont_f32), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(step(_zpk_cont_f32), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(step(_ss_cont_f32), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
# c64
assert_type(step(_lti_c64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(step(_tf_cont_c64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
assert_type(step(_zpk_cont_c64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
assert_type(step(_ss_cont_c64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
# f64
assert_type(step(_lti_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(step(_tf_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(step(_zpk_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(step(_ss_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
# c128
assert_type(step(_lti_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(step(_tf_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
assert_type(step(_zpk_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])
assert_type(step(_ss_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128 | Any]])

###
# bode

# f32
assert_type(bode(_lti_f32), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], onp.Array1D[np.float32]])
assert_type(bode(_tf_cont_f32), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], onp.Array1D[np.float32]])
assert_type(bode(_zpk_cont_f32), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], onp.Array1D[np.float32]])
assert_type(bode(_ss_cont_f32), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], onp.Array1D[np.float32]])
# c64
assert_type(bode(_lti_c64), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], onp.Array1D[np.float32]])
assert_type(bode(_tf_cont_c64), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], onp.Array1D[np.float32]])
assert_type(bode(_zpk_cont_c64), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], onp.Array1D[np.float32]])
assert_type(bode(_ss_cont_c64), tuple[onp.Array1D[np.float32], onp.Array1D[np.float32], onp.Array1D[np.float32]])
# f64
assert_type(bode(_lti_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bode(_tf_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bode(_zpk_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bode(_ss_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]])
# c128
assert_type(bode(_lti_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bode(_tf_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bode(_zpk_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(bode(_ss_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64], onp.Array1D[np.float64]])

###
# freqresp

# f32
assert_type(freqresp(_lti_f32), tuple[onp.Array1D[np.float32], onp.Array1D[np.complex64]])
assert_type(freqresp(_tf_cont_f32), tuple[onp.Array1D[np.float32], onp.Array1D[np.complex64]])
assert_type(freqresp(_zpk_cont_f32), tuple[onp.Array1D[np.float32], onp.Array1D[np.complex64]])
assert_type(freqresp(_ss_cont_f32), tuple[onp.Array1D[np.float32], onp.Array1D[np.complex64]])
# c64
assert_type(freqresp(_lti_c64), tuple[onp.Array1D[np.float32], onp.Array1D[np.complex64]])
assert_type(freqresp(_tf_cont_c64), tuple[onp.Array1D[np.float32], onp.Array1D[np.complex64]])
assert_type(freqresp(_zpk_cont_c64), tuple[onp.Array1D[np.float32], onp.Array1D[np.complex64]])
assert_type(freqresp(_ss_cont_c64), tuple[onp.Array1D[np.float32], onp.Array1D[np.complex64]])
# f64
assert_type(freqresp(_lti_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqresp(_tf_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqresp(_zpk_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqresp(_ss_cont_f64), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
# c128
assert_type(freqresp(_lti_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqresp(_tf_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqresp(_zpk_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
assert_type(freqresp(_ss_cont_c128), tuple[onp.Array1D[np.float64], onp.Array1D[np.complex128]])
