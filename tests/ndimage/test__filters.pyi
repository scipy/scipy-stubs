# type-tests for `ndimage/_filters.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.ndimage import (
    convolve,
    convolve1d,
    correlate,
    correlate1d,
    gaussian_filter,
    gaussian_filter1d,
    gaussian_gradient_magnitude,
    gaussian_laplace,
    laplace,
    maximum_filter,
    maximum_filter1d,
    median_filter,
    minimum_filter,
    minimum_filter1d,
    percentile_filter,
    prewitt,
    rank_filter,
    sobel,
    uniform_filter,
    uniform_filter1d,
)

###
# Test variables

# typed numpy arrays - CanArray[_ShapeT, _DTypeT] -> shape+dtype are preserved
i32_2d: onp.Array2D[np.int32]
f32_2d: onp.Array2D[np.float32]
f64_2d: onp.Array2D[np.float64]
f64_nd: onp.ArrayND[np.float64]
c64_nd: onp.ArrayND[np.complex64]
c128_nd: onp.ArrayND[np.complex128]

# weights for convolve/correlate
weights_1d: onp.Array1D[np.float64]
weights_nd: onp.ArrayND[np.float64]

# plain-Python sequences -> the JustInt / JustFloat64 / JustComplex128 overloads
int_2d: list[list[int]]
float_2d: list[list[float]]
complex_2d: list[list[complex]]

###
# correlate1d

# CanArray[ShapeT, DTypeT] overload: shape AND dtype are preserved
assert_type(correlate1d(f64_2d, weights_1d), onp.Array2D[np.float64])
assert_type(correlate1d(f32_2d, weights_1d), onp.Array2D[np.float32])
assert_type(correlate1d(i32_2d, weights_1d), onp.Array2D[np.int32])

# plain-Python list inputs
assert_type(correlate1d(int_2d, weights_1d), onp.ArrayND[np.intp])
assert_type(correlate1d(float_2d, weights_1d), onp.ArrayND[np.float64])
assert_type(correlate1d(complex_2d, weights_1d), onp.ArrayND[np.complex128])

# ArrayND: shape loosens to tuple[int, ...]
assert_type(correlate1d(f64_nd, weights_1d), onp.ArrayND[np.float64])
assert_type(correlate1d(c128_nd, weights_1d), onp.ArrayND[np.complex128])

###
# correlate

assert_type(correlate(f64_2d, weights_nd), onp.Array2D[np.float64])
assert_type(correlate(f32_2d, weights_nd), onp.Array2D[np.float32])
assert_type(correlate(int_2d, weights_nd), onp.ArrayND[np.intp])
assert_type(correlate(float_2d, weights_nd), onp.ArrayND[np.float64])
assert_type(correlate(complex_2d, weights_nd), onp.ArrayND[np.complex128])
assert_type(correlate(f64_nd, weights_nd), onp.ArrayND[np.float64])
assert_type(correlate(c128_nd, weights_nd), onp.ArrayND[np.complex128])

###
# convolve1d

assert_type(convolve1d(f64_2d, weights_1d), onp.Array2D[np.float64])
assert_type(convolve1d(f32_2d, weights_1d), onp.Array2D[np.float32])
assert_type(convolve1d(int_2d, weights_1d), onp.ArrayND[np.intp])
assert_type(convolve1d(float_2d, weights_1d), onp.ArrayND[np.float64])
assert_type(convolve1d(complex_2d, weights_1d), onp.ArrayND[np.complex128])
assert_type(convolve1d(f64_nd, weights_1d), onp.ArrayND[np.float64])
assert_type(convolve1d(c128_nd, weights_1d), onp.ArrayND[np.complex128])

###
# convolve

assert_type(convolve(f64_2d, weights_nd), onp.Array2D[np.float64])
assert_type(convolve(f32_2d, weights_nd), onp.Array2D[np.float32])
assert_type(convolve(int_2d, weights_nd), onp.ArrayND[np.intp])
assert_type(convolve(float_2d, weights_nd), onp.ArrayND[np.float64])
assert_type(convolve(complex_2d, weights_nd), onp.ArrayND[np.complex128])
assert_type(convolve(f64_nd, weights_nd), onp.ArrayND[np.float64])
assert_type(convolve(c128_nd, weights_nd), onp.ArrayND[np.complex128])

###
# prewitt

assert_type(prewitt(f64_2d), onp.Array2D[np.float64])
assert_type(prewitt(f32_2d), onp.Array2D[np.float32])
assert_type(prewitt(int_2d), onp.ArrayND[np.intp])
assert_type(prewitt(float_2d), onp.ArrayND[np.float64])
assert_type(prewitt(complex_2d), onp.ArrayND[np.complex128])
assert_type(prewitt(f64_nd), onp.ArrayND[np.float64])
assert_type(prewitt(c128_nd), onp.ArrayND[np.complex128])

###
# sobel

assert_type(sobel(f64_2d), onp.Array2D[np.float64])
assert_type(sobel(f32_2d), onp.Array2D[np.float32])
assert_type(sobel(int_2d), onp.ArrayND[np.intp])
assert_type(sobel(float_2d), onp.ArrayND[np.float64])
assert_type(sobel(complex_2d), onp.ArrayND[np.complex128])
assert_type(sobel(f64_nd), onp.ArrayND[np.float64])
assert_type(sobel(c128_nd), onp.ArrayND[np.complex128])

###
# laplace

assert_type(laplace(f64_2d), onp.Array2D[np.float64])
assert_type(laplace(f32_2d), onp.Array2D[np.float32])
assert_type(laplace(int_2d), onp.ArrayND[np.intp])
assert_type(laplace(float_2d), onp.ArrayND[np.float64])
assert_type(laplace(complex_2d), onp.ArrayND[np.complex128])
assert_type(laplace(f64_nd), onp.ArrayND[np.float64])
assert_type(laplace(c128_nd), onp.ArrayND[np.complex128])

###
# gaussian_laplace

assert_type(gaussian_laplace(f64_2d, sigma=1), onp.Array2D[np.float64])
assert_type(gaussian_laplace(f32_2d, sigma=1), onp.Array2D[np.float32])
assert_type(gaussian_laplace(int_2d, sigma=1), onp.ArrayND[np.intp])
assert_type(gaussian_laplace(float_2d, sigma=1), onp.ArrayND[np.float64])
assert_type(gaussian_laplace(complex_2d, sigma=1), onp.ArrayND[np.complex128])
assert_type(gaussian_laplace(f64_nd, sigma=1), onp.ArrayND[np.float64])
assert_type(gaussian_laplace(c128_nd, sigma=1), onp.ArrayND[np.complex128])

###
# gaussian_gradient_magnitude

assert_type(gaussian_gradient_magnitude(f64_2d, sigma=1), onp.Array2D[np.float64])
assert_type(gaussian_gradient_magnitude(f32_2d, sigma=1), onp.Array2D[np.float32])
assert_type(gaussian_gradient_magnitude(int_2d, sigma=1), onp.ArrayND[np.intp])
assert_type(gaussian_gradient_magnitude(float_2d, sigma=1), onp.ArrayND[np.float64])
assert_type(gaussian_gradient_magnitude(complex_2d, sigma=1), onp.ArrayND[np.complex128])
assert_type(gaussian_gradient_magnitude(f64_nd, sigma=1), onp.ArrayND[np.float64])
assert_type(gaussian_gradient_magnitude(c128_nd, sigma=1), onp.ArrayND[np.complex128])

###
# gaussian_filter1d

assert_type(gaussian_filter1d(f64_2d, sigma=1), onp.Array2D[np.float64])
assert_type(gaussian_filter1d(f32_2d, sigma=1), onp.Array2D[np.float32])
assert_type(gaussian_filter1d(int_2d, sigma=1), onp.ArrayND[np.intp])
assert_type(gaussian_filter1d(float_2d, sigma=1), onp.ArrayND[np.float64])
assert_type(gaussian_filter1d(complex_2d, sigma=1), onp.ArrayND[np.complex128])
assert_type(gaussian_filter1d(f64_nd, sigma=1), onp.ArrayND[np.float64])
assert_type(gaussian_filter1d(c128_nd, sigma=1), onp.ArrayND[np.complex128])

###
# gaussian_filter

assert_type(gaussian_filter(f64_2d, sigma=1), onp.Array2D[np.float64])
assert_type(gaussian_filter(f32_2d, sigma=1), onp.Array2D[np.float32])
assert_type(gaussian_filter(int_2d, sigma=1), onp.ArrayND[np.intp])
assert_type(gaussian_filter(float_2d, sigma=1), onp.ArrayND[np.float64])
assert_type(gaussian_filter(complex_2d, sigma=1), onp.ArrayND[np.complex128])
assert_type(gaussian_filter(f64_nd, sigma=1), onp.ArrayND[np.float64])
assert_type(gaussian_filter(c128_nd, sigma=1), onp.ArrayND[np.complex128])

###
# uniform_filter1d

assert_type(uniform_filter1d(f64_2d, size=3), onp.Array2D[np.float64])
assert_type(uniform_filter1d(f32_2d, size=3), onp.Array2D[np.float32])
assert_type(uniform_filter1d(int_2d, size=3), onp.ArrayND[np.intp])
assert_type(uniform_filter1d(float_2d, size=3), onp.ArrayND[np.float64])
assert_type(uniform_filter1d(complex_2d, size=3), onp.ArrayND[np.complex128])
assert_type(uniform_filter1d(f64_nd, size=3), onp.ArrayND[np.float64])
assert_type(uniform_filter1d(c128_nd, size=3), onp.ArrayND[np.complex128])

###
# uniform_filter

assert_type(uniform_filter(f64_2d), onp.Array2D[np.float64])
assert_type(uniform_filter(f32_2d), onp.Array2D[np.float32])
assert_type(uniform_filter(int_2d), onp.ArrayND[np.intp])
assert_type(uniform_filter(float_2d), onp.ArrayND[np.float64])
assert_type(uniform_filter(complex_2d), onp.ArrayND[np.complex128])
assert_type(uniform_filter(f64_nd), onp.ArrayND[np.float64])
assert_type(uniform_filter(c128_nd), onp.ArrayND[np.complex128])

###
# maximum_filter1d

assert_type(maximum_filter1d(f64_2d, size=3), onp.Array2D[np.float64])
assert_type(maximum_filter1d(f32_2d, size=3), onp.Array2D[np.float32])
assert_type(maximum_filter1d(int_2d, size=3), onp.ArrayND[np.intp])
assert_type(maximum_filter1d(float_2d, size=3), onp.ArrayND[np.float64])
assert_type(maximum_filter1d(complex_2d, size=3), onp.ArrayND[np.complex128])
assert_type(maximum_filter1d(f64_nd, size=3), onp.ArrayND[np.float64])
assert_type(maximum_filter1d(c128_nd, size=3), onp.ArrayND[np.complex128])

###
# maximum_filter

assert_type(maximum_filter(f64_2d), onp.Array2D[np.float64])
assert_type(maximum_filter(f32_2d), onp.Array2D[np.float32])
assert_type(maximum_filter(int_2d), onp.ArrayND[np.intp])
assert_type(maximum_filter(float_2d), onp.ArrayND[np.float64])
assert_type(maximum_filter(complex_2d), onp.ArrayND[np.complex128])
assert_type(maximum_filter(f64_nd), onp.ArrayND[np.float64])
assert_type(maximum_filter(c128_nd), onp.ArrayND[np.complex128])

###
# minimum_filter1d

assert_type(minimum_filter1d(f64_2d, size=3), onp.Array2D[np.float64])
assert_type(minimum_filter1d(f32_2d, size=3), onp.Array2D[np.float32])
assert_type(minimum_filter1d(int_2d, size=3), onp.ArrayND[np.intp])
assert_type(minimum_filter1d(float_2d, size=3), onp.ArrayND[np.float64])
assert_type(minimum_filter1d(c128_nd, size=3), onp.ArrayND[np.complex128])
assert_type(minimum_filter1d(f64_nd, size=3), onp.ArrayND[np.float64])

###
# minimum_filter

assert_type(minimum_filter(f64_2d), onp.Array2D[np.float64])
assert_type(minimum_filter(f32_2d), onp.Array2D[np.float32])
assert_type(minimum_filter(int_2d), onp.ArrayND[np.intp])
assert_type(minimum_filter(float_2d), onp.ArrayND[np.float64])
assert_type(minimum_filter(c128_nd), onp.ArrayND[np.complex128])
assert_type(minimum_filter(f64_nd), onp.ArrayND[np.float64])

###
# median_filter

assert_type(median_filter(f64_2d), onp.Array2D[np.float64])
assert_type(median_filter(f32_2d), onp.Array2D[np.float32])
assert_type(median_filter(int_2d), onp.ArrayND[np.intp])
assert_type(median_filter(float_2d), onp.ArrayND[np.float64])
assert_type(median_filter(c128_nd), onp.ArrayND[np.complex128])
assert_type(median_filter(f64_nd), onp.ArrayND[np.float64])

###
# rank_filter

assert_type(rank_filter(f64_2d, rank=1, size=3), onp.Array2D[np.float64])
assert_type(rank_filter(f32_2d, rank=1, size=3), onp.Array2D[np.float32])
assert_type(rank_filter(int_2d, rank=1, size=3), onp.ArrayND[np.intp])
assert_type(rank_filter(float_2d, rank=1, size=3), onp.ArrayND[np.float64])
assert_type(rank_filter(c128_nd, rank=1, size=3), onp.ArrayND[np.complex128])
assert_type(rank_filter(f64_nd, rank=1, size=3), onp.ArrayND[np.float64])

###
# percentile_filter

assert_type(percentile_filter(f64_2d, percentile=50, size=3), onp.Array2D[np.float64])
assert_type(percentile_filter(f32_2d, percentile=50, size=3), onp.Array2D[np.float32])
assert_type(percentile_filter(int_2d, percentile=50, size=3), onp.ArrayND[np.intp])
assert_type(percentile_filter(float_2d, percentile=50, size=3), onp.ArrayND[np.float64])
assert_type(percentile_filter(c128_nd, percentile=50, size=3), onp.ArrayND[np.complex128])
assert_type(percentile_filter(f64_nd, percentile=50, size=3), onp.ArrayND[np.float64])

###
# output=type[ScalarT]: matches ToDType[_ScalarT] -> returned as ArrayND[_ScalarT]

assert_type(gaussian_filter(f64_nd, sigma=1, output=np.float32), onp.ArrayND[np.float32])
assert_type(uniform_filter(f64_nd, output=np.float32), onp.ArrayND[np.float32])
