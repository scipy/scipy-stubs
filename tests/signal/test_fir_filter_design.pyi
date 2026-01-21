# type-tests for `signal/_fir_filter_design.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import firls, firwin, firwin2, firwin_2d, kaiser_atten, kaiser_beta, kaiserord, minimum_phase, remez

###

_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]
_u8_1d: onp.Array1D[np.uint8]
_i16_1d: onp.Array1D[np.int16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]
_u8_2d: onp.Array2D[np.uint8]
_i16_2d: onp.Array2D[np.int16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

###

# kaiser*

assert_type(kaiser_beta(5.0), float)
assert_type(kaiser_atten(10, 0.5), float)
assert_type(kaiserord(60.0, 0.1), tuple[int, float])

# firwin

assert_type(firwin(1, 0.5), onp.Array1D[np.float64])
assert_type(firwin(1, _py_i_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _py_f_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _u8_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _i16_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _f32_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _f64_1d), onp.Array1D[np.float64])

# firwin_2d

assert_type(firwin_2d((3, 4), 0.5), onp.Array2D[np.float64])
assert_type(firwin_2d((3, 4), "hann"), onp.Array2D[np.float64])
assert_type(firwin_2d((3, 4), "hann", fc=0.5, circular=True), onp.Array2D[np.float64])
assert_type(firwin_2d((3, 4), "hann", fc=_py_i_1d, circular=True), onp.Array2D[np.float64])
assert_type(firwin_2d((3, 4), "hann", fc=_py_f_1d, circular=True), onp.Array2D[np.float64])
assert_type(firwin_2d((3, 4), "hann", fc=_u8_1d, circular=True), onp.Array2D[np.float64])
assert_type(firwin_2d((3, 4), "hann", fc=_i16_1d, circular=True), onp.Array2D[np.float64])
assert_type(firwin_2d((3, 4), "hann", fc=_f32_1d, circular=True), onp.Array2D[np.float64])
assert_type(firwin_2d((3, 4), "hann", fc=_f64_1d, circular=True), onp.Array2D[np.float64])

# firwin2

assert_type(firwin2(9, _py_i_1d, _py_i_1d), onp.Array1D[np.float64])
assert_type(firwin2(9, _py_f_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(firwin2(9, _u8_1d, _u8_1d), onp.Array1D[np.float64])
assert_type(firwin2(9, _i16_1d, _i16_1d), onp.Array1D[np.float64])
assert_type(firwin2(9, _f32_1d, _f32_1d), onp.Array1D[np.float64])
assert_type(firwin2(9, _f64_1d, _f64_1d), onp.Array1D[np.float64])

# firls

assert_type(firls(9, _py_i_1d, _py_i_1d), onp.Array1D[np.float64])
assert_type(firls(9, _py_f_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(firls(9, _u8_1d, _u8_1d), onp.Array1D[np.float64])
assert_type(firls(9, _i16_1d, _i16_1d), onp.Array1D[np.float64])
assert_type(firls(9, _f32_1d, _f32_1d), onp.Array1D[np.float64])
assert_type(firls(9, _f64_1d, _f64_1d), onp.Array1D[np.float64])

assert_type(firls(9, _py_i_2d, _py_i_2d), onp.Array1D[np.float64])
assert_type(firls(9, _py_f_2d, _py_f_2d), onp.Array1D[np.float64])
assert_type(firls(9, _u8_2d, _u8_2d), onp.Array1D[np.float64])
assert_type(firls(9, _i16_2d, _i16_2d), onp.Array1D[np.float64])
assert_type(firls(9, _f32_2d, _f32_2d), onp.Array1D[np.float64])
assert_type(firls(9, _f64_2d, _f64_2d), onp.Array1D[np.float64])

# remez

assert_type(remez(9, _py_i_1d, _py_i_1d), onp.Array1D[np.float64])
assert_type(remez(9, _py_f_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(remez(9, _u8_1d, _u8_1d), onp.Array1D[np.float64])
assert_type(remez(9, _i16_1d, _i16_1d), onp.Array1D[np.float64])
assert_type(remez(9, _f32_1d, _f32_1d), onp.Array1D[np.float64])
assert_type(remez(9, _f64_1d, _f64_1d), onp.Array1D[np.float64])

# minimum_phase

assert_type(minimum_phase(_py_i_1d), onp.Array1D[np.float64])
assert_type(minimum_phase(_py_f_1d), onp.Array1D[np.float64])
assert_type(minimum_phase(_u8_1d), onp.Array1D[np.float64])
assert_type(minimum_phase(_i16_1d), onp.Array1D[np.float64])
assert_type(minimum_phase(_f32_1d), onp.Array1D[np.float64])
assert_type(minimum_phase(_f64_1d), onp.Array1D[np.float64])
