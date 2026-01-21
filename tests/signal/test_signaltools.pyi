# type-tests for `signal/_signaltools.pyi`

from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import choose_conv_method, convolve, correlate, filtfilt, lfilter, lfilter_zi, lfiltic, sosfiltfilt

###

_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]
_i16_1d: onp.Array1D[np.int16]
_i64_1d: onp.Array1D[np.int64]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]
_i16_2d: onp.Array2D[np.int16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

###

# choose_conv_method

assert_type(choose_conv_method(_py_i_1d, _py_i_1d), Literal["direct"])
assert_type(choose_conv_method(_py_f_1d, _py_f_1d), Literal["direct", "fft"])
assert_type(choose_conv_method(_py_c_1d, _py_c_1d), Literal["direct", "fft"])
assert_type(choose_conv_method(_i16_1d, _i16_1d), Literal["direct"])
assert_type(choose_conv_method(_f32_1d, _f32_1d), Literal["direct", "fft"])
assert_type(choose_conv_method(_f64_1d, _f64_1d), Literal["direct", "fft"])
assert_type(choose_conv_method(_c64_1d, _c64_1d), Literal["direct", "fft"])
assert_type(choose_conv_method(_c128_1d, _c128_1d), Literal["direct", "fft"])

assert_type(choose_conv_method(_py_i_1d, _py_i_1d, measure=True)[0], Literal["direct", "fft"])
assert_type(choose_conv_method(_py_f_1d, _py_f_1d, measure=True)[0], Literal["direct", "fft"])
assert_type(choose_conv_method(_py_c_1d, _py_c_1d, measure=True)[0], Literal["direct", "fft"])
assert_type(choose_conv_method(_i16_1d, _i16_1d, measure=True)[0], Literal["direct", "fft"])
assert_type(choose_conv_method(_f32_1d, _f32_1d, measure=True)[0], Literal["direct", "fft"])
assert_type(choose_conv_method(_f64_1d, _f64_1d, measure=True)[0], Literal["direct", "fft"])
assert_type(choose_conv_method(_c64_1d, _c64_1d, measure=True)[0], Literal["direct", "fft"])
assert_type(choose_conv_method(_c128_1d, _c128_1d, measure=True)[0], Literal["direct", "fft"])

assert_type(choose_conv_method(_py_i_1d, _py_i_1d, measure=True)[1]["direct"], float)
assert_type(choose_conv_method(_py_f_1d, _py_f_1d, measure=True)[1]["direct"], float)
assert_type(choose_conv_method(_py_c_1d, _py_c_1d, measure=True)[1]["direct"], float)
assert_type(choose_conv_method(_i16_1d, _i16_1d, measure=True)[1]["direct"], float)
assert_type(choose_conv_method(_f32_1d, _f32_1d, measure=True)[1]["direct"], float)
assert_type(choose_conv_method(_f64_1d, _f64_1d, measure=True)[1]["direct"], float)
assert_type(choose_conv_method(_c64_1d, _c64_1d, measure=True)[1]["direct"], float)
assert_type(choose_conv_method(_c128_1d, _c128_1d, measure=True)[1]["direct"], float)

assert_type(choose_conv_method(_py_i_1d, _py_i_1d, measure=True)[1]["fft"], float)
assert_type(choose_conv_method(_py_f_1d, _py_f_1d, measure=True)[1]["fft"], float)
assert_type(choose_conv_method(_py_c_1d, _py_c_1d, measure=True)[1]["fft"], float)
assert_type(choose_conv_method(_i16_1d, _i16_1d, measure=True)[1]["fft"], float)
assert_type(choose_conv_method(_f32_1d, _f32_1d, measure=True)[1]["fft"], float)
assert_type(choose_conv_method(_f64_1d, _f64_1d, measure=True)[1]["fft"], float)
assert_type(choose_conv_method(_c64_1d, _c64_1d, measure=True)[1]["fft"], float)
assert_type(choose_conv_method(_c128_1d, _c128_1d, measure=True)[1]["fft"], float)

# convolve (same as correlate)

assert_type(convolve(_py_b_1d, _py_b_1d), onp.ArrayND[np.bool_])
assert_type(convolve(_py_i_1d, _py_i_1d), onp.ArrayND[np.int64])
assert_type(convolve(_py_f_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(convolve(_i64_1d, _i64_1d), onp.Array1D[np.int64])
assert_type(convolve(_f32_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(convolve(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(convolve(_c64_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(convolve(_c128_1d, _c128_1d), onp.Array1D[np.complex128])

# correlate (same as convolve)

assert_type(correlate(_py_b_1d, _py_b_1d), onp.ArrayND[np.bool_])
assert_type(correlate(_py_i_1d, _py_i_1d), onp.ArrayND[np.int64])
assert_type(correlate(_py_f_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(correlate(_i64_1d, _i64_1d), onp.Array1D[np.int64])
assert_type(correlate(_f32_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(correlate(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(correlate(_c64_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(correlate(_c128_1d, _c128_1d), onp.Array1D[np.complex128])

# lfilter_zi

assert_type(lfilter_zi(_py_i_1d, _py_i_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_py_f_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_i16_1d, _i16_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_i16_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(lfilter_zi(_f32_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(lfilter_zi(_f32_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_f32_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(lfilter_zi(_c64_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(lfilter_zi(_c64_1d, _c128_1d), onp.Array1D[np.complex128])
assert_type(lfilter_zi(_c128_1d, _c128_1d), onp.Array1D[np.complex128])

# lfiltic

assert_type(lfiltic(_py_i_1d, _py_i_1d, _py_i_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_py_f_1d, _py_f_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_i16_1d, _i16_1d, _i16_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_f32_1d, _f32_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(lfiltic(_f32_1d, _f32_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_f64_1d, _f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_f32_1d, _f32_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(lfiltic(_c64_1d, _c64_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(lfiltic(_c64_1d, _c64_1d, _c128_1d), onp.Array1D[np.complex128])
assert_type(lfiltic(_c128_1d, _c128_1d, _c128_1d), onp.Array1D[np.complex128])

# lfilter

assert_type(lfilter(_py_i_1d, _py_i_1d, _py_i_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_py_f_1d, _py_f_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_i16_1d, _i16_1d, _i16_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_f32_1d, _f32_1d, _f32_1d), onp.ArrayND[np.float32])
assert_type(lfilter(_f32_1d, _f32_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_f64_1d, _f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_f32_1d, _f32_1d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(lfilter(_c64_1d, _c64_1d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(lfilter(_c64_1d, _c64_1d, _c128_1d), onp.ArrayND[np.complex128])
assert_type(lfilter(_c128_1d, _c128_1d, _c128_1d), onp.ArrayND[np.complex128])

# filtfilt

assert_type(filtfilt(_py_i_1d, _py_i_1d, _py_i_1d), onp.ArrayND[np.float64])
assert_type(filtfilt(_py_f_1d, _py_f_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(filtfilt(_i16_1d, _i16_1d, _i16_1d), onp.ArrayND[np.float64])
assert_type(filtfilt(_f32_1d, _f32_1d, _f32_1d), onp.ArrayND[np.float32])
assert_type(filtfilt(_f32_1d, _f32_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(filtfilt(_f64_1d, _f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(filtfilt(_f32_1d, _f32_1d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(filtfilt(_c64_1d, _c64_1d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(filtfilt(_c64_1d, _c64_1d, _c128_1d), onp.ArrayND[np.complex128])
assert_type(filtfilt(_c128_1d, _c128_1d, _c128_1d), onp.ArrayND[np.complex128])

# sosfiltfilt

assert_type(sosfiltfilt(_py_i_2d, _py_i_1d), onp.ArrayND[np.float64])
assert_type(sosfiltfilt(_py_f_2d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(sosfiltfilt(_i16_2d, _i16_1d), onp.ArrayND[np.float64])
assert_type(sosfiltfilt(_f32_2d, _f32_1d), onp.ArrayND[np.float32])
assert_type(sosfiltfilt(_f32_2d, _f64_1d), onp.ArrayND[np.float64])
assert_type(sosfiltfilt(_f64_2d, _f64_1d), onp.ArrayND[np.float64])
assert_type(sosfiltfilt(_f32_2d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(sosfiltfilt(_c64_2d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(sosfiltfilt(_c64_2d, _c128_1d), onp.ArrayND[np.complex128])
assert_type(sosfiltfilt(_c128_2d, _c128_1d), onp.ArrayND[np.complex128])
