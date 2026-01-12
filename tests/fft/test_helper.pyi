from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.fft import fftfreq, fftshift, ifftshift, next_fast_len, prev_fast_len, rfftfreq

###

_list_int: list[int] = ...
_list_float: list[float] = ...
_list_complex: list[complex] = ...

_arr_i64: onp.ArrayND[np.int64]
_arr_f16: onp.ArrayND[np.float16]
_arr_f32: onp.ArrayND[np.float32]
_arr_f64: onp.ArrayND[np.float64]
_arr_f80: onp.ArrayND[np.longdouble]
_arr_c64: onp.ArrayND[np.complex64]
_arr_c128: onp.ArrayND[np.complex128]
_arr_c160: onp.ArrayND[np.clongdouble]

###

# next_fast_len (same as prev_fast_len)
assert_type(next_fast_len(42), int)
assert_type(next_fast_len(42, True), int)
assert_type(next_fast_len(target=42, real=True), int)
# prev_fast_len (same as next_fast_len)
assert_type(prev_fast_len(42), int)
assert_type(prev_fast_len(42, True), int)
assert_type(prev_fast_len(target=42, real=True), int)

# fftfreq (same as rfftfreq)
assert_type(fftfreq(8), onp.Array1D[np.float64])
assert_type(fftfreq(8, 2.0), onp.Array1D[np.float64])
assert_type(fftfreq(8, d=2.0), onp.Array1D[np.float64])
# rfftfreq (same as fftfreq)
assert_type(rfftfreq(8), onp.Array1D[np.float64])
assert_type(rfftfreq(8, 2.0), onp.Array1D[np.float64])
assert_type(rfftfreq(8, d=2.0), onp.Array1D[np.float64])

# fftshift (same as ifftshift)
assert_type(fftshift(_list_int), onp.ArrayND[np.float64])
assert_type(fftshift(_list_float), onp.ArrayND[np.float64])
assert_type(fftshift(_list_complex), onp.ArrayND[np.complex128])
assert_type(fftshift(_arr_i64), onp.ArrayND[np.float64])
assert_type(fftshift(_arr_f16), onp.ArrayND[np.float16])
assert_type(fftshift(_arr_f32), onp.ArrayND[np.float32])
assert_type(fftshift(_arr_f64), onp.ArrayND[np.float64])
assert_type(fftshift(_arr_f80), onp.ArrayND[np.longdouble])
assert_type(fftshift(_arr_c64), onp.ArrayND[np.complex64])
assert_type(fftshift(_arr_c128), onp.ArrayND[np.complex128])
assert_type(fftshift(_arr_c160), onp.ArrayND[np.clongdouble])

# ifftshift (same as fftshift)
assert_type(ifftshift(_list_int), onp.ArrayND[np.float64])
assert_type(ifftshift(_list_float), onp.ArrayND[np.float64])
assert_type(ifftshift(_list_complex), onp.ArrayND[np.complex128])
assert_type(ifftshift(_arr_i64), onp.ArrayND[np.float64])
assert_type(ifftshift(_arr_f16), onp.ArrayND[np.float16])
assert_type(ifftshift(_arr_f32), onp.ArrayND[np.float32])
assert_type(ifftshift(_arr_f64), onp.ArrayND[np.float64])
assert_type(ifftshift(_arr_f80), onp.ArrayND[np.longdouble])
assert_type(ifftshift(_arr_c64), onp.ArrayND[np.complex64])
assert_type(ifftshift(_arr_c128), onp.ArrayND[np.complex128])
assert_type(ifftshift(_arr_c160), onp.ArrayND[np.clongdouble])
