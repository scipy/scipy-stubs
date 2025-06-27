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

f80_1d: onp.Array1D[np.longdouble]
f80_2d: onp.Array2D[np.longdouble]
f80_3d: onp.Array3D[np.longdouble]

c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]
c64_3d: onp.Array3D[np.complex64]

c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c128_3d: onp.Array3D[np.complex128]

c160_1d: onp.Array1D[np.clongdouble]
c160_2d: onp.Array2D[np.clongdouble]
c160_3d: onp.Array3D[np.clongdouble]

###
# fft

assert_type(fft(int_1d), onp.Array1D[np.complex128])
assert_type(fft(int_2d), onp.ArrayND[np.complex128])
assert_type(fft(float_1d), onp.Array1D[np.complex128])
assert_type(fft(float_2d), onp.ArrayND[np.complex128])
assert_type(fft(complex_1d), onp.Array1D[np.complex128])
assert_type(fft(complex_2d), onp.ArrayND[np.complex128])

assert_type(fft(i16_1d), onp.Array1D[np.complex128])
assert_type(fft(i16_2d), onp.Array2D[np.complex128])

assert_type(fft(f32_1d), onp.Array1D[np.complex64])
assert_type(fft(f32_2d), onp.Array2D[np.complex64])

assert_type(fft(f64_1d), onp.Array1D[np.complex128])
assert_type(fft(f64_2d), onp.Array2D[np.complex128])

assert_type(fft(f80_1d), onp.Array1D[np.clongdouble])
assert_type(fft(f80_2d), onp.Array2D[np.clongdouble])

assert_type(fft(c64_1d), onp.Array1D[np.complex64])
assert_type(fft(c64_2d), onp.Array2D[np.complex64])

assert_type(fft(c128_1d), onp.Array1D[np.complex128])
assert_type(fft(c128_2d), onp.Array2D[np.complex128])

assert_type(fft(c160_1d), onp.Array1D[np.clongdouble])
assert_type(fft(c160_2d), onp.Array2D[np.clongdouble])
