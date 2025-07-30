# ruff: noqa: ERA001

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.fft import fft

int_1d: list[int]
int_2d: list[list[int]]
int_3d: list[list[list[int]]]

float_1d: list[float]
float_2d: list[list[float]]
float_3d: list[list[list[float]]]

complex_1d: list[complex]
complex_2d: list[list[complex]]
complex_3d: list[list[list[complex]]]

i16_1d: onp.Array1D[np.int16]
i16_2d: onp.Array2D[np.int16]
i16_3d: onp.Array3D[np.int16]

f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]
f32_3d: onp.Array3D[np.float32]

f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]

c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]
c64_3d: onp.Array3D[np.complex64]

c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

# NOTE: These extended precision types may not exist at runtime, but are used
# here to work around `[c]longdouble` issues on `numpy<2.2`

f80_1d: onp.Array1D[np.float128]
f80_2d: onp.Array2D[np.float128]
f80_3d: onp.Array3D[np.float128]

c160_1d: onp.Array1D[np.complex256]
c160_2d: onp.Array2D[np.complex256]
c160_3d: onp.Array3D[np.complex256]

###
# fft

# NOTE: the commented out assertions only work on numpy 2.1+, so we instead check for assignability

assert_type(fft(int_1d), onp.Array1D[np.complex128])
assert_type(fft(float_1d), onp.Array1D[np.complex128])
assert_type(fft(complex_1d), onp.Array1D[np.complex128])

assert_type(fft(int_2d), onp.ArrayND[np.complex128])
assert_type(fft(float_2d), onp.ArrayND[np.complex128])
assert_type(fft(complex_2d), onp.ArrayND[np.complex128])

# assert_type(fft(i16_1d), onp.Array1D[np.complex128])
# assert_type(fft(f32_1d), onp.Array1D[np.complex64])
# assert_type(fft(f64_1d), onp.Array1D[np.complex128])
# assert_type(fft(f80_1d), onp.Array1D[np.clongdouble])
# assert_type(fft(c64_1d), onp.Array1D[np.complex64])
# assert_type(fft(c128_1d), onp.Array1D[np.complex128])
# assert_type(fft(c160_1d), onp.Array1D[np.clongdouble])
_10: onp.Array1D[np.complex128] = fft(i16_1d)
_11: onp.Array1D[np.complex64] = fft(f32_1d)
_12: onp.Array1D[np.complex128] = fft(f64_1d)
_13: onp.Array1D[np.clongdouble] = fft(f80_1d)
_14: onp.Array1D[np.complex64] = fft(c64_1d)
_15: onp.Array1D[np.complex128] = fft(c128_1d)
_16: onp.Array1D[np.clongdouble] = fft(c160_1d)

# assert_type(fft(i16_2d), onp.Array2D[np.complex128])
# assert_type(fft(f32_2d), onp.Array2D[np.complex64])
# assert_type(fft(f64_2d), onp.Array2D[np.complex128])
# assert_type(fft(f80_2d), onp.Array2D[np.clongdouble])
# assert_type(fft(c64_2d), onp.Array2D[np.complex64])
# assert_type(fft(c128_2d), onp.Array2D[np.complex128])
# assert_type(fft(c160_2d), onp.Array2D[np.clongdouble])
_20: onp.Array2D[np.complex128] = fft(i16_2d)
_21: onp.Array2D[np.complex64] = fft(f32_2d)
_22: onp.Array2D[np.complex128] = fft(f64_2d)
_23: onp.Array2D[np.clongdouble] = fft(f80_2d)
_24: onp.Array2D[np.complex64] = fft(c64_2d)
_25: onp.Array2D[np.complex128] = fft(c128_2d)
_26: onp.Array2D[np.clongdouble] = fft(c160_2d)
