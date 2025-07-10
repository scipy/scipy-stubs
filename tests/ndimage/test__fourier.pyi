# type-tests for `ndimage/_fourier.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.ndimage import fourier_gaussian, fourier_shift

i8_nd: npt.NDArray[np.int8]
f16_nd: npt.NDArray[np.float16]
f32_nd: npt.NDArray[np.float32]
f64_nd: npt.NDArray[np.float64]
c64_nd: npt.NDArray[np.complex64]
c128_nd: npt.NDArray[np.complex128]

int_2d: list[list[int]]
float_2d: list[list[float]]
complex_2d: list[list[complex]]

_OutputArray: TypeAlias = onp.Array2D[np.complex64]
_OutputArrayND: TypeAlias = onp.ArrayND[np.complex64]
output_array: _OutputArray
output_sctype: type[np.complex64]

###
# `fourier_gaussian` (also `fourier_ellipsoid` and `fourier_uniform`)
# NOTE: `fourier_uniform` and `fourier_ellipsoid` have the same signature, so no need to also test those.

assert_type(fourier_gaussian(i8_nd, 4), onp.ArrayND[np.float64])
assert_type(fourier_gaussian(f16_nd, 4), onp.ArrayND[np.float32])
assert_type(fourier_gaussian(f32_nd, 4), onp.ArrayND[np.float32])
assert_type(fourier_gaussian(f64_nd, 4), onp.ArrayND[np.float64])
assert_type(fourier_gaussian(c64_nd, 4), onp.ArrayND[np.complex64])
assert_type(fourier_gaussian(c128_nd, 4), onp.ArrayND[np.complex128])
assert_type(fourier_gaussian(int_2d, 4), onp.ArrayND[np.float64])
assert_type(fourier_gaussian(float_2d, 4), onp.ArrayND[np.float64])
assert_type(fourier_gaussian(complex_2d, 4), onp.ArrayND[np.complex128])

assert_type(fourier_gaussian(i8_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_gaussian(f16_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_gaussian(f32_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_gaussian(f64_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_gaussian(c64_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_gaussian(c128_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_gaussian(int_2d, 4, output=output_array), _OutputArray)
assert_type(fourier_gaussian(float_2d, 4, output=output_array), _OutputArray)
assert_type(fourier_gaussian(complex_2d, 4, output=output_array), _OutputArray)

assert_type(fourier_gaussian(i8_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_gaussian(f16_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_gaussian(f32_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_gaussian(f64_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_gaussian(c64_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_gaussian(c128_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_gaussian(int_2d, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_gaussian(float_2d, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_gaussian(complex_2d, 4, output=output_sctype), _OutputArrayND)

###
# `fourier_shift`
# NOTE: Unlike the other three functions, this always returns complex output.

assert_type(fourier_shift(i8_nd, 4), onp.ArrayND[np.complex128])
assert_type(fourier_shift(f16_nd, 4), onp.ArrayND[np.complex128])
assert_type(fourier_shift(f32_nd, 4), onp.ArrayND[np.complex128])
assert_type(fourier_shift(f64_nd, 4), onp.ArrayND[np.complex128])
assert_type(fourier_shift(c64_nd, 4), onp.ArrayND[np.complex64])
assert_type(fourier_shift(c128_nd, 4), onp.ArrayND[np.complex128])
assert_type(fourier_shift(int_2d, 4), onp.ArrayND[np.complex128])
assert_type(fourier_shift(float_2d, 4), onp.ArrayND[np.complex128])
assert_type(fourier_shift(complex_2d, 4), onp.ArrayND[np.complex128])

assert_type(fourier_shift(i8_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_shift(f16_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_shift(f32_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_shift(f64_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_shift(c64_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_shift(c128_nd, 4, output=output_array), _OutputArray)
assert_type(fourier_shift(int_2d, 4, output=output_array), _OutputArray)
assert_type(fourier_shift(float_2d, 4, output=output_array), _OutputArray)
assert_type(fourier_shift(complex_2d, 4, output=output_array), _OutputArray)

assert_type(fourier_shift(i8_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_shift(f16_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_shift(f32_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_shift(f64_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_shift(c64_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_shift(c128_nd, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_shift(int_2d, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_shift(float_2d, 4, output=output_sctype), _OutputArrayND)
assert_type(fourier_shift(complex_2d, 4, output=output_sctype), _OutputArrayND)
