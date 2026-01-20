from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import (
    block_diag,
    circulant,
    companion,
    convolution_matrix,
    dft,
    fiedler,
    fiedler_companion,
    hadamard,
    hankel,
    helmert,
    hilbert,
    leslie,
    toeplitz,
)

###

_py_b_1d: list[bool]
_py_b_2d: list[list[bool]]
_py_b_3d: list[list[list[bool]]]

_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_i_3d: list[list[list[int]]]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]

_py_c_1d: list[complex]
_py_c_2d: list[list[complex]]
_py_c_3d: list[list[list[complex]]]

_i8_1d: onp.Array1D[np.int8]
_i8_2d: onp.Array2D[np.int8]
_i8_3d: onp.Array3D[np.int8]
_i8_nd: onp.ArrayND[np.int8]

_f32_1d: onp.Array1D[np.float32]
_f32_2d: onp.Array2D[np.float32]
_f32_3d: onp.Array3D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]

###

# toeplitz

assert_type(toeplitz(_py_b_1d), onp.Array2D[np.bool_])
assert_type(toeplitz(_py_b_2d), onp.Array3D[np.bool_])
assert_type(toeplitz(_py_b_3d), onp.ArrayND[np.bool_])

assert_type(toeplitz(_py_i_1d), onp.Array2D[np.int_])
assert_type(toeplitz(_py_i_2d), onp.Array3D[np.int_])
assert_type(toeplitz(_py_i_3d), onp.ArrayND[np.int_])

assert_type(toeplitz(_py_f_1d), onp.Array2D[np.float64])
assert_type(toeplitz(_py_f_2d), onp.Array3D[np.float64])
assert_type(toeplitz(_py_f_3d), onp.ArrayND[np.float64])

assert_type(toeplitz(_py_c_1d), onp.Array2D[np.complex128])
assert_type(toeplitz(_py_c_2d), onp.Array3D[np.complex128])
assert_type(toeplitz(_py_c_3d), onp.ArrayND[np.complex128])

assert_type(toeplitz(_i8_1d), onp.Array2D[np.int8])
assert_type(toeplitz(_i8_2d), onp.Array3D[np.int8])
assert_type(toeplitz(_i8_3d), onp.ArrayND[np.int8])
assert_type(toeplitz(_i8_nd), onp.ArrayND[np.int8])

assert_type(toeplitz(_f32_1d), onp.Array2D[np.float32])
assert_type(toeplitz(_f32_2d), onp.Array3D[np.float32])
assert_type(toeplitz(_f32_3d), onp.ArrayND[np.float32])
assert_type(toeplitz(_f32_nd), onp.ArrayND[np.float32])

assert_type(toeplitz(_f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(toeplitz(_f64_2d, _f64_2d), onp.Array3D[np.float64])
assert_type(toeplitz(_f64_3d, _f64_3d), onp.ArrayND[np.float64])
assert_type(toeplitz(_f64_nd, _f64_nd), onp.ArrayND[np.float64])

# circulant

assert_type(circulant(_py_b_1d), onp.Array2D[np.bool_])
assert_type(circulant(_py_b_2d), onp.Array3D[np.bool_])
assert_type(circulant(_py_b_3d), onp.ArrayND[np.bool_])

assert_type(circulant(_py_i_1d), onp.Array2D[np.int_])
assert_type(circulant(_py_i_2d), onp.Array3D[np.int_])
assert_type(circulant(_py_i_3d), onp.ArrayND[np.int_])

assert_type(circulant(_py_f_1d), onp.Array2D[np.float64])
assert_type(circulant(_py_f_2d), onp.Array3D[np.float64])
assert_type(circulant(_py_f_3d), onp.ArrayND[np.float64])

assert_type(circulant(_py_c_1d), onp.Array2D[np.complex128])
assert_type(circulant(_py_c_2d), onp.Array3D[np.complex128])
assert_type(circulant(_py_c_3d), onp.ArrayND[np.complex128])

assert_type(circulant(_i8_1d), onp.Array2D[np.int8])
assert_type(circulant(_i8_2d), onp.Array3D[np.int8])
assert_type(circulant(_i8_3d), onp.ArrayND[np.int8])
assert_type(circulant(_i8_nd), onp.ArrayND[np.int8])

assert_type(circulant(_f32_1d), onp.Array2D[np.float32])
assert_type(circulant(_f32_2d), onp.Array3D[np.float32])
assert_type(circulant(_f32_3d), onp.ArrayND[np.float32])
assert_type(circulant(_f32_nd), onp.ArrayND[np.float32])

# convolution_matrix

assert_type(convolution_matrix(_py_b_1d, 4), onp.Array2D[np.bool_])
assert_type(convolution_matrix(_py_b_2d, 4), onp.Array3D[np.bool_])
assert_type(convolution_matrix(_py_b_3d, 4), onp.ArrayND[np.bool_])

assert_type(convolution_matrix(_py_i_1d, 4), onp.Array2D[np.int_])
assert_type(convolution_matrix(_py_i_2d, 4), onp.Array3D[np.int_])
assert_type(convolution_matrix(_py_i_3d, 4), onp.ArrayND[np.int_])

assert_type(convolution_matrix(_py_f_1d, 4), onp.Array2D[np.float64])
assert_type(convolution_matrix(_py_f_2d, 4), onp.Array3D[np.float64])
assert_type(convolution_matrix(_py_f_3d, 4), onp.ArrayND[np.float64])

assert_type(convolution_matrix(_py_c_1d, 4), onp.Array2D[np.complex128])
assert_type(convolution_matrix(_py_c_2d, 4), onp.Array3D[np.complex128])
assert_type(convolution_matrix(_py_c_3d, 4), onp.ArrayND[np.complex128])

assert_type(convolution_matrix(_i8_1d, 4), onp.Array2D[np.int8])
assert_type(convolution_matrix(_i8_2d, 4), onp.Array3D[np.int8])
assert_type(convolution_matrix(_i8_3d, 4), onp.ArrayND[np.int8])
assert_type(convolution_matrix(_i8_nd, 4), onp.ArrayND[np.int8])

assert_type(convolution_matrix(_f32_1d, 4), onp.Array2D[np.float32])
assert_type(convolution_matrix(_f32_2d, 4), onp.Array3D[np.float32])
assert_type(convolution_matrix(_f32_3d, 4), onp.ArrayND[np.float32])
assert_type(convolution_matrix(_f32_nd, 4), onp.ArrayND[np.float32])

# fiedler

assert_type(fiedler(_py_i_1d), onp.Array2D[np.int_])
assert_type(fiedler(_py_i_2d), onp.Array3D[np.int_])
assert_type(fiedler(_py_i_3d), onp.ArrayND[np.int_])

assert_type(fiedler(_py_f_1d), onp.Array2D[np.float64])
assert_type(fiedler(_py_f_2d), onp.Array3D[np.float64])
assert_type(fiedler(_py_f_3d), onp.ArrayND[np.float64])

assert_type(fiedler(_py_c_1d), onp.Array2D[np.complex128])
assert_type(fiedler(_py_c_2d), onp.Array3D[np.complex128])
assert_type(fiedler(_py_c_3d), onp.ArrayND[np.complex128])

assert_type(fiedler(_i8_1d), onp.Array2D[np.int8])
assert_type(fiedler(_i8_2d), onp.Array3D[np.int8])
assert_type(fiedler(_i8_3d), onp.ArrayND[np.int8])
assert_type(fiedler(_i8_nd), onp.ArrayND[np.int8])

assert_type(fiedler(_f32_1d), onp.Array2D[np.float32])
assert_type(fiedler(_f32_2d), onp.Array3D[np.float32])
assert_type(fiedler(_f32_3d), onp.ArrayND[np.float32])
assert_type(fiedler(_f32_nd), onp.ArrayND[np.float32])

# companion

assert_type(companion(_py_i_1d), onp.Array2D[np.float64])
assert_type(companion(_py_i_2d), onp.Array3D[np.float64])
assert_type(companion(_py_i_3d), onp.ArrayND[np.float64])

assert_type(companion(_py_f_1d), onp.Array2D[np.float64])
assert_type(companion(_py_f_2d), onp.Array3D[np.float64])
assert_type(companion(_py_f_3d), onp.ArrayND[np.float64])

assert_type(companion(_py_c_1d), onp.Array2D[np.complex128])
assert_type(companion(_py_c_2d), onp.Array3D[np.complex128])
assert_type(companion(_py_c_3d), onp.ArrayND[np.complex128])

assert_type(companion(_i8_1d), onp.Array2D[np.float64])
assert_type(companion(_i8_2d), onp.Array3D[np.float64])
assert_type(companion(_i8_3d), onp.ArrayND[np.float64])
assert_type(companion(_i8_nd), onp.ArrayND[np.float64])

assert_type(companion(_f32_1d), onp.Array2D[np.float32])
assert_type(companion(_f32_2d), onp.Array3D[np.float32])
assert_type(companion(_f32_3d), onp.ArrayND[np.float32])
assert_type(companion(_f32_nd), onp.ArrayND[np.float32])

# fiedler_companion

assert_type(fiedler_companion(_py_i_1d), onp.Array2D[np.float64])
assert_type(fiedler_companion(_py_i_2d), onp.Array3D[np.float64])
assert_type(fiedler_companion(_py_i_3d), onp.ArrayND[np.float64])

assert_type(fiedler_companion(_py_f_1d), onp.Array2D[np.float64])
assert_type(fiedler_companion(_py_f_2d), onp.Array3D[np.float64])
assert_type(fiedler_companion(_py_f_3d), onp.ArrayND[np.float64])

assert_type(fiedler_companion(_py_c_1d), onp.Array2D[np.complex128])
assert_type(fiedler_companion(_py_c_2d), onp.Array3D[np.complex128])
assert_type(fiedler_companion(_py_c_3d), onp.ArrayND[np.complex128])

assert_type(fiedler_companion(_i8_1d), onp.Array2D[np.float64])
assert_type(fiedler_companion(_i8_2d), onp.Array3D[np.float64])
assert_type(fiedler_companion(_i8_3d), onp.ArrayND[np.float64])
assert_type(fiedler_companion(_i8_nd), onp.ArrayND[np.float64])

assert_type(fiedler_companion(_f32_1d), onp.Array2D[np.float32])
assert_type(fiedler_companion(_f32_2d), onp.Array3D[np.float32])
assert_type(fiedler_companion(_f32_3d), onp.ArrayND[np.float32])
assert_type(fiedler_companion(_f32_nd), onp.ArrayND[np.float32])

# leslie

assert_type(leslie(_py_b_1d, _py_b_1d), onp.Array2D[np.bool_])
assert_type(leslie(_py_b_1d, _py_i_1d), onp.Array2D[np.int_])
assert_type(leslie(_py_i_1d, _py_b_1d), onp.Array2D[np.int_])
assert_type(leslie(_py_i_1d, _py_i_1d), onp.Array2D[np.int_])
assert_type(leslie(_py_i_1d, _py_f_1d), onp.Array2D[np.float64])
assert_type(leslie(_py_f_1d, _py_i_1d), onp.Array2D[np.float64])
assert_type(leslie(_py_f_1d, _py_f_1d), onp.Array2D[np.float64])
assert_type(leslie(_py_f_1d, _py_c_1d), onp.Array2D[np.complex128])
assert_type(leslie(_py_c_1d, _py_f_1d), onp.Array2D[np.complex128])
assert_type(leslie(_py_c_1d, _py_c_1d), onp.Array2D[np.complex128])
assert_type(leslie(_i8_1d, _i8_1d), onp.Array2D[np.int8])
assert_type(leslie(_f32_1d, _f32_1d), onp.Array2D[np.float32])

# block_diag

assert_type(block_diag(), onp.Array2D[np.float64])
assert_type(block_diag(_py_b_1d), onp.Array2D[np.bool_])
assert_type(block_diag(_py_i_1d), onp.Array2D[np.int_])
assert_type(block_diag(_py_i_1d, _py_b_1d), onp.Array2D[np.int_])
assert_type(block_diag(_py_f_1d), onp.Array2D[np.float64])
assert_type(block_diag(_py_f_1d, _py_i_1d), onp.Array2D[np.float64])
assert_type(block_diag(_py_c_1d), onp.Array2D[np.complex128])
assert_type(block_diag(_py_c_1d, _py_f_1d), onp.Array2D[np.complex128])
assert_type(block_diag(_i8_1d), onp.Array2D[np.int8])
assert_type(block_diag(_f32_1d), onp.Array2D[np.float32])

# dft

assert_type(dft(4), onp.Array2D[np.complex128])
assert_type(dft(4.0), onp.Array2D[np.complex128])

# hadamard

assert_type(hadamard(4), onp.Array2D[np.int_])
assert_type(hadamard(4, int), onp.Array2D[np.int_])
assert_type(hadamard(4, float), onp.Array2D[np.float64])
assert_type(hadamard(4, complex), onp.Array2D[np.complex128])
assert_type(hadamard(4, object), onp.Array2D[np.object_])
assert_type(hadamard(4, np.float16), onp.Array2D[np.float16])

# hankel

assert_type(hankel(_py_i_1d), onp.Array2D[np.int_])
assert_type(hankel(_py_f_1d), onp.Array2D[np.float64])
assert_type(hankel(_py_c_1d), onp.Array2D[np.complex128])
assert_type(hankel(_f64_1d), onp.Array2D[np.float64])
assert_type(hankel(_f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_f64_1d, _f64_1d), onp.Array2D[np.float64])
assert_type(hankel(_f64_1d, _f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_f64_nd, _f64_nd), onp.Array2D[np.float64])
assert_type(hankel(_py_i_2d), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]
assert_type(hankel(_py_i_3d), onp.Array2D[np.int_])  # pyright: ignore[reportDeprecated]  # pyrefly: ignore[deprecated]

# helmert

assert_type(helmert(4), onp.Array2D[np.float64])
assert_type(helmert(4, True), onp.Array2D[np.float64])
assert_type(helmert(4, full=True), onp.Array2D[np.float64])

# hilbert

assert_type(hilbert(4), onp.Array2D[np.float64])
