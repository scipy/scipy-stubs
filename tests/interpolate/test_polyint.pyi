from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import (
    BarycentricInterpolator,
    KroghInterpolator,
    approximate_taylor_polynomial,
    barycentric_interpolate,
    krogh_interpolate,
)

py_int_1d: list[float]
py_int_2d: list[list[float]]
py_float_1d: list[float]
py_float_2d: list[list[float]]
py_complex_2d: list[list[complex]]

i32_1d: onp.Array1D[np.int32]
f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]
f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
c64_2d: onp.Array2D[np.complex64]
c128_2d: onp.Array2D[np.complex128]

###
# KroghInterpolator

assert_type(KroghInterpolator(py_int_1d, py_float_2d), KroghInterpolator[np.float64, np.float64])
assert_type(KroghInterpolator(py_int_1d, py_complex_2d), KroghInterpolator[np.complex128, np.float64])
assert_type(KroghInterpolator(py_int_1d, f32_2d), KroghInterpolator[np.float64, np.float64])
assert_type(KroghInterpolator(py_int_1d, c64_2d), KroghInterpolator[np.complex128, np.float64])
assert_type(KroghInterpolator(py_float_1d, py_float_2d), KroghInterpolator[np.float64, np.float64])
assert_type(KroghInterpolator(py_float_1d, py_complex_2d), KroghInterpolator[np.complex128, np.float64])
assert_type(KroghInterpolator(py_float_1d, f32_2d), KroghInterpolator[np.float64, np.float64])
assert_type(KroghInterpolator(py_float_1d, c64_2d), KroghInterpolator[np.complex128, np.float64])
assert_type(KroghInterpolator(i32_1d, py_float_2d), KroghInterpolator[np.float64, np.int32])
assert_type(KroghInterpolator(i32_1d, py_complex_2d), KroghInterpolator[np.complex128, np.int32])
assert_type(KroghInterpolator(i32_1d, f32_2d), KroghInterpolator[np.float64, np.int32])
assert_type(KroghInterpolator(i32_1d, c64_2d), KroghInterpolator[np.complex128, np.int32])

krogh_f_f32: KroghInterpolator[np.float64, np.float32]
krogh_c_f32: KroghInterpolator[np.complex128, np.float32]

assert_type(krogh_f_f32(0), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])
assert_type(krogh_c_f32(0), np.ndarray[tuple[Any, ...], np.dtype[np.complex128]])

assert_type(krogh_f_f32.derivative(0), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])
assert_type(krogh_c_f32.derivative(0), np.ndarray[tuple[Any, ...], np.dtype[np.complex128]])

assert_type(krogh_f_f32.derivatives(0), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])
assert_type(krogh_c_f32.derivatives(0), np.ndarray[tuple[Any, ...], np.dtype[np.complex128]])

###
# BarycentricInterpolator

assert_type(BarycentricInterpolator(py_int_1d, py_float_2d), BarycentricInterpolator[np.float64])
assert_type(BarycentricInterpolator(py_int_1d, py_complex_2d), BarycentricInterpolator[np.complex128])
assert_type(BarycentricInterpolator(py_int_1d, f32_2d), BarycentricInterpolator[np.float64])
assert_type(BarycentricInterpolator(py_int_1d, c64_2d), BarycentricInterpolator[np.complex128])
assert_type(BarycentricInterpolator(py_float_1d, py_float_2d), BarycentricInterpolator[np.float64])
assert_type(BarycentricInterpolator(py_float_1d, py_complex_2d), BarycentricInterpolator[np.complex128])
assert_type(BarycentricInterpolator(py_float_1d, f32_2d), BarycentricInterpolator[np.float64])
assert_type(BarycentricInterpolator(py_float_1d, c64_2d), BarycentricInterpolator[np.complex128])
assert_type(BarycentricInterpolator(i32_1d, py_float_2d), BarycentricInterpolator[np.float64])
assert_type(BarycentricInterpolator(i32_1d, py_complex_2d), BarycentricInterpolator[np.complex128])
assert_type(BarycentricInterpolator(i32_1d, f32_2d), BarycentricInterpolator[np.float64])
assert_type(BarycentricInterpolator(i32_1d, c64_2d), BarycentricInterpolator[np.complex128])

bary_f_f32: BarycentricInterpolator[np.float64]
bary_c_f32: BarycentricInterpolator[np.complex128]

assert_type(bary_f_f32(0), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])
assert_type(bary_c_f32(0), np.ndarray[tuple[Any, ...], np.dtype[np.complex128]])

assert_type(bary_f_f32.derivative(0), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])
assert_type(bary_c_f32.derivative(0), np.ndarray[tuple[Any, ...], np.dtype[np.complex128]])

assert_type(bary_f_f32.derivatives(0), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])
assert_type(bary_c_f32.derivatives(0), np.ndarray[tuple[Any, ...], np.dtype[np.complex128]])

###
# krogh_interpolate

assert_type(krogh_interpolate(py_float_1d, py_float_2d, 0.0), onp.ArrayND[np.float64])
assert_type(krogh_interpolate(py_float_1d, f32_2d, f64_1d, der=1, axis=1), onp.ArrayND[np.float64])
assert_type(krogh_interpolate(py_float_1d, py_complex_2d, 0.0), onp.ArrayND[np.complex128])
assert_type(krogh_interpolate(py_float_1d, c64_2d, 0.0, der=[0, 1]), onp.ArrayND[np.complex128])

###
# barycentric_interpolate

assert_type(barycentric_interpolate(py_float_1d, py_float_2d, 0.0), onp.ArrayND[np.float64])
assert_type(barycentric_interpolate(py_float_1d, f32_2d, f64_2d, 1, der=1), onp.ArrayND[np.float64])
assert_type(barycentric_interpolate(py_float_1d, py_complex_2d, 0.0), onp.ArrayND[np.complex128])
assert_type(barycentric_interpolate(py_float_1d, c128_2d, 0.0, rng=0), onp.ArrayND[np.complex128])

###
# approximate_taylor_polynomial

assert_type(approximate_taylor_polynomial(np.sin, 0.0, 3, 1.0), np.poly1d)  # pyright:ignore[reportDeprecated] # pyrefly:ignore[deprecated]
