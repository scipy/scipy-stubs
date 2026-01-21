# type-tests for `signal/_signaltools.pyi`

from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp
import optype.numpy.compat as npc

from scipy.signal import (
    choose_conv_method,
    convolve,
    convolve2d,
    correlate,
    correlate2d,
    correlation_lags,
    deconvolve,
    fftconvolve,
    filtfilt,
    hilbert,
    lfilter,
    lfilter_zi,
    lfiltic,
    medfilt,
    medfilt2d,
    oaconvolve,
    order_filter,
    sosfilt_zi,
    sosfiltfilt,
    wiener,
)

###

_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]
_u8_1d: onp.Array1D[np.uint8]
_i16_1d: onp.Array1D[np.int16]
_i64_1d: onp.Array1D[np.int64]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_f80_1d: onp.Array1D[npc.floating80]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]
_c160_1d: onp.Array1D[npc.complexfloating160]

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]
_u8_2d: onp.Array2D[np.uint8]
_i16_2d: onp.Array2D[np.int16]
_f16_2d: onp.Array2D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_f80_2d: onp.Array2D[npc.floating80]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

_u8_nd: onp.ArrayND[np.uint8]
_f16_nd: onp.ArrayND[np.float32]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]
_f80_nd: onp.ArrayND[npc.floating80]

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

# convolve2d (same as correlate2d)

assert_type(convolve2d(_py_i_1d, _py_i_1d), onp.Array2D[np.int64])
assert_type(convolve2d(_py_f_1d, _py_f_1d), onp.Array2D[np.float64])
assert_type(convolve2d(_i64_1d, _i64_1d), onp.Array2D[np.int64])
assert_type(convolve2d(_f32_1d, _f32_1d), onp.Array2D[np.float32])
assert_type(convolve2d(_f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(convolve2d(_c64_1d, _c64_1d), onp.Array2D[np.complex64])
assert_type(convolve2d(_c128_1d, _c128_1d), onp.Array2D[np.complex128])

# correlate2d (same as convolve2d)

assert_type(correlate2d(_py_i_1d, _py_i_1d), onp.Array2D[np.int64])
assert_type(correlate2d(_py_f_1d, _py_f_1d), onp.Array2D[np.float64])
assert_type(correlate2d(_i64_1d, _i64_1d), onp.Array2D[np.int64])
assert_type(correlate2d(_f32_1d, _f32_1d), onp.Array2D[np.float32])
assert_type(correlate2d(_f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(correlate2d(_c64_1d, _c64_1d), onp.Array2D[np.complex64])
assert_type(correlate2d(_c128_1d, _c128_1d), onp.Array2D[np.complex128])

# fftconvolve

assert_type(fftconvolve(_py_i_1d, _py_i_1d), onp.ArrayND[np.float64])
assert_type(fftconvolve(_py_f_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(fftconvolve(_i64_1d, _i64_1d), onp.Array1D[np.float64])
assert_type(fftconvolve(_f32_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(fftconvolve(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(fftconvolve(_c64_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(fftconvolve(_c128_1d, _c128_1d), onp.Array1D[np.complex128])

# oaconvolve

assert_type(oaconvolve(_py_i_1d, _py_i_1d), onp.ArrayND[np.float64])
assert_type(oaconvolve(_py_f_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(oaconvolve(_i64_1d, _i64_1d), onp.Array1D[np.float64])
assert_type(oaconvolve(_f32_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(oaconvolve(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(oaconvolve(_c64_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(oaconvolve(_c128_1d, _c128_1d), onp.Array1D[np.complex128])

# deconvolve

assert_type(deconvolve(_py_i_1d, _py_i_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(deconvolve(_py_f_1d, _py_f_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(deconvolve(_i64_1d, _i64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(deconvolve(_f32_1d, _f32_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(deconvolve(_f64_1d, _f64_1d), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(deconvolve(_c64_1d, _c64_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])
assert_type(deconvolve(_c128_1d, _c128_1d), tuple[onp.Array1D[np.complex128], onp.Array1D[np.complex128]])

# correlation_lags

assert_type(correlation_lags(5, 3), onp.Array1D[np.int_])
assert_type(correlation_lags(5, 3, mode="valid"), onp.Array1D[np.int_])
assert_type(correlation_lags(5, 3, mode="same"), onp.Array1D[np.int_])
assert_type(correlation_lags(5, 3, mode="full"), onp.Array1D[np.int_])

###

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

# sosfilt_zi

assert_type(sosfilt_zi(_py_i_2d), onp.Array2D[np.float64])
assert_type(sosfilt_zi(_py_f_2d), onp.Array2D[np.float64])
assert_type(sosfilt_zi(_i16_2d), onp.Array2D[np.float64])
assert_type(sosfilt_zi(_f32_2d), onp.Array2D[np.float32])
assert_type(sosfilt_zi(_f64_2d), onp.Array2D[np.float64])

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

# order_filter

assert_type(order_filter(_py_i_1d, _i16_2d, 0), onp.ArrayND[np.int_])
assert_type(order_filter(_py_f_1d, _i16_2d, 0), onp.ArrayND[np.float64])
assert_type(order_filter(_i16_1d, _i16_2d, 0), onp.Array1D[np.int16])
assert_type(order_filter(_f32_1d, _i16_2d, 0), onp.Array1D[np.float32])
assert_type(order_filter(_f64_1d, _i16_2d, 0), onp.Array1D[np.float64])

assert_type(order_filter(_py_i_2d, _i16_2d, 0), onp.ArrayND[np.int_])
assert_type(order_filter(_py_f_2d, _i16_2d, 0), onp.ArrayND[np.float64])
assert_type(order_filter(_i16_2d, _i16_2d, 0), onp.Array2D[np.int16])
assert_type(order_filter(_f32_2d, _i16_2d, 0), onp.Array2D[np.float32])
assert_type(order_filter(_f64_2d, _i16_2d, 0), onp.Array2D[np.float64])

# medfilt

assert_type(medfilt(_py_i_1d), onp.ArrayND[np.int_])
assert_type(medfilt(_py_f_1d), onp.ArrayND[np.float64])
assert_type(medfilt(_i16_1d), onp.Array1D[np.int16])
assert_type(medfilt(_f32_1d), onp.Array1D[np.float32])
assert_type(medfilt(_f64_1d), onp.Array1D[np.float64])

assert_type(medfilt(_py_i_2d), onp.ArrayND[np.int_])
assert_type(medfilt(_py_f_2d), onp.ArrayND[np.float64])
assert_type(medfilt(_i16_2d), onp.Array2D[np.int16])
assert_type(medfilt(_f32_2d), onp.Array2D[np.float32])
assert_type(medfilt(_f64_2d), onp.Array2D[np.float64])

# medfilt2d

assert_type(medfilt2d(_py_i_2d), onp.Array2D[np.int_])
assert_type(medfilt2d(_py_f_2d), onp.Array2D[np.float64])
assert_type(medfilt2d(_u8_2d), onp.Array2D[np.uint8])
assert_type(medfilt2d(_f32_2d), onp.Array2D[np.float32])
assert_type(medfilt2d(_f64_2d), onp.Array2D[np.float64])

# wiener

assert_type(wiener(_py_i_1d), onp.ArrayND[np.float64])
assert_type(wiener(_py_f_1d), onp.ArrayND[np.float64])
assert_type(wiener(_u8_1d), onp.ArrayND[np.float64])
assert_type(wiener(_f32_1d), onp.ArrayND[np.float64])
assert_type(wiener(_f64_1d), onp.ArrayND[np.float64])
assert_type(wiener(_f80_1d), onp.ArrayND[npc.floating80])
assert_type(wiener(_c128_1d), onp.ArrayND[np.complex128])
assert_type(wiener(_c160_1d), onp.ArrayND[npc.complexfloating160])

# hilbert

assert_type(hilbert(_py_i_1d), onp.ArrayND[np.complex128])
assert_type(hilbert(_py_f_1d), onp.ArrayND[np.complex128])
assert_type(hilbert(_u8_1d), onp.Array1D[np.complex128])
assert_type(hilbert(_f16_1d), onp.Array1D[np.complex64])
assert_type(hilbert(_f32_1d), onp.Array1D[np.complex64])
assert_type(hilbert(_f64_1d), onp.Array1D[np.complex128])
assert_type(hilbert(_f80_1d), onp.Array1D[npc.complexfloating160])

assert_type(hilbert(_py_i_2d), onp.ArrayND[np.complex128])
assert_type(hilbert(_py_f_2d), onp.ArrayND[np.complex128])
assert_type(hilbert(_u8_2d), onp.Array2D[np.complex128])
assert_type(hilbert(_f16_2d), onp.Array2D[np.complex64])
assert_type(hilbert(_f32_2d), onp.Array2D[np.complex64])
assert_type(hilbert(_f64_2d), onp.Array2D[np.complex128])
assert_type(hilbert(_f80_2d), onp.Array2D[npc.complexfloating160])

assert_type(hilbert(_u8_nd), onp.ArrayND[np.complex128])
assert_type(hilbert(_f16_nd), onp.ArrayND[np.complex64])
assert_type(hilbert(_f32_nd), onp.ArrayND[np.complex64])
assert_type(hilbert(_f64_nd), onp.ArrayND[np.complex128])
assert_type(hilbert(_f80_nd), onp.ArrayND[npc.complexfloating160])
