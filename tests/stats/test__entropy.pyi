# type-tests for `stats/_entropy.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import differential_entropy, entropy

i8_0d: np.int8
i8_1d: onp.Array1D[np.int8]
i8_2d: onp.Array2D[np.int8]
i8_3d: onp.Array3D[np.int8]
i8_nd: onp.ArrayND[np.int8, tuple[int, ...]]

f16_0d: np.float16
f16_1d: onp.Array1D[np.float16]
f16_2d: onp.Array2D[np.float16]
f16_3d: onp.Array3D[np.float16]
f16_nd: onp.ArrayND[np.float16, tuple[int, ...]]

f32_0d: np.float32
f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]
f32_3d: onp.Array3D[np.float32]
f32_nd: onp.ArrayND[np.float32, tuple[int, ...]]

f64_0d: np.float64
f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
f64_nd: onp.ArrayND[np.float64, tuple[int, ...]]

f80_0d: np.float128
f80_1d: onp.Array1D[np.float128]
f80_2d: onp.Array2D[np.float128]
f80_3d: onp.Array3D[np.float128]
f80_nd: onp.ArrayND[np.float128, tuple[int, ...]]

c64_0d: np.complex64
c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]
c64_3d: onp.Array3D[np.complex64]
c64_nd: onp.ArrayND[np.complex64, tuple[int, ...]]

py_f_0d: float
py_f_1d: list[float]
py_f_2d: list[list[float]]
py_f_3d: list[list[list[float]]]
py_f_nd: list[list[list[list[list[list[float]]]]]]

py_c_0d: complex
py_c_1d: list[complex]
py_c_2d: list[list[complex]]
py_c_3d: list[list[list[complex]]]
py_c_nd: list[list[list[list[list[list[complex]]]]]]

###
# entropy

assert_type(entropy(i8_0d), np.float64)
assert_type(entropy(i8_1d), np.float64)
assert_type(entropy(i8_2d), onp.Array1D[np.float64])  # X
assert_type(entropy(i8_3d), onp.Array2D[np.float64])  # X
assert_type(entropy(i8_nd), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(entropy(i8_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(i8_nd, axis=None), np.float64)

assert_type(entropy(f16_0d), np.float32)
assert_type(entropy(f16_1d), np.float32)
assert_type(entropy(f16_2d), onp.Array1D[np.float32])  # X
assert_type(entropy(f16_3d), onp.Array2D[np.float32])  # X
assert_type(entropy(f16_nd), np.float32 | onp.ArrayND[np.float32])  # X
assert_type(entropy(f16_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(entropy(f16_nd, axis=None), np.float32)

assert_type(entropy(f32_0d), np.float32)
assert_type(entropy(f32_1d), np.float32)
assert_type(entropy(f32_2d), onp.Array1D[np.float32])  # X
assert_type(entropy(f32_3d), onp.Array2D[np.float32])  # X
assert_type(entropy(f32_nd), np.float32 | onp.ArrayND[np.float32])  # X
assert_type(entropy(f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(entropy(f32_nd, axis=None), np.float32)

assert_type(entropy(f64_0d), np.float64)
assert_type(entropy(f64_1d), np.float64)
assert_type(entropy(f64_2d), onp.Array1D[np.float64])  # X
assert_type(entropy(f64_3d), onp.Array2D[np.float64])  # X
assert_type(entropy(f64_nd), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(entropy(f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(f64_nd, axis=None), np.float64)

assert_type(entropy(py_f_0d), np.float64)
assert_type(entropy(py_f_1d), np.float64)
assert_type(entropy(py_f_2d), onp.Array1D[np.float64])
assert_type(entropy(py_f_3d), onp.Array2D[np.float64])
assert_type(entropy(py_f_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(entropy(py_f_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(py_f_nd, axis=None), np.float64)

assert_type(entropy(i8_0d, f16_0d), np.float64)
assert_type(entropy(i8_1d, f16_0d), np.float64)
assert_type(entropy(i8_2d, f16_0d), onp.Array1D[np.float64])  # X
assert_type(entropy(i8_3d, f16_0d), onp.Array2D[np.float64])  # X
assert_type(entropy(i8_nd, f16_0d), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(entropy(i8_nd, f16_0d, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(i8_nd, f16_0d, axis=None), np.float64)

assert_type(entropy(f16_0d, f16_0d), np.float32)
assert_type(entropy(f16_1d, f16_0d), np.float32)
assert_type(entropy(f16_2d, f16_0d), onp.Array1D[np.float32])  # X
assert_type(entropy(f16_3d, f16_0d), onp.Array2D[np.float32])  # X
assert_type(entropy(f16_nd, f16_0d), np.float32 | onp.ArrayND[np.float32])  # X
assert_type(entropy(f16_nd, f16_0d, keepdims=True), onp.ArrayND[np.float32])
assert_type(entropy(f16_nd, f16_0d, axis=None), np.float32)

assert_type(entropy(f32_0d, f16_0d), np.float32)
assert_type(entropy(f32_1d, f16_0d), np.float32)
assert_type(entropy(f32_2d, f16_0d), onp.Array1D[np.float32])  # X
assert_type(entropy(f32_3d, f16_0d), onp.Array2D[np.float32])  # X
assert_type(entropy(f32_nd, f16_0d), np.float32 | onp.ArrayND[np.float32])  # X
assert_type(entropy(f32_nd, f16_0d, keepdims=True), onp.ArrayND[np.float32])
assert_type(entropy(f32_nd, f16_0d, axis=None), np.float32)

assert_type(entropy(f64_0d, f16_0d), np.float64)
assert_type(entropy(f64_1d, f16_0d), np.float64)
assert_type(entropy(f64_2d, f16_0d), onp.Array1D[np.float64])  # X
assert_type(entropy(f64_3d, f16_0d), onp.Array2D[np.float64])  # X
assert_type(entropy(f64_nd, f16_0d), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(entropy(f64_nd, f16_0d, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(f64_nd, f16_0d, axis=None), np.float64)

assert_type(entropy(py_f_0d, f16_0d), np.float64)
assert_type(entropy(py_f_1d, f16_0d), np.float64)
assert_type(entropy(py_f_2d, f16_0d), onp.Array1D[np.float64])
assert_type(entropy(py_f_3d, f16_0d), onp.Array2D[np.float64])
assert_type(entropy(py_f_nd, f16_0d), np.float64 | onp.ArrayND[np.float64])
assert_type(entropy(py_f_nd, f16_0d, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(py_f_nd, f16_0d, axis=None), np.float64)

assert_type(entropy(i8_0d, py_f_0d), np.float64)
assert_type(entropy(i8_1d, py_f_0d), np.float64)
assert_type(entropy(i8_2d, py_f_0d), onp.Array1D[np.float64])  # X
assert_type(entropy(i8_3d, py_f_0d), onp.Array2D[np.float64])  # X
assert_type(entropy(i8_nd, py_f_0d), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(entropy(i8_nd, py_f_0d, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(i8_nd, py_f_0d, axis=None), np.float64)

assert_type(entropy(f16_0d, py_f_0d), np.float64)
assert_type(entropy(f16_1d, py_f_0d), np.float64)
assert_type(entropy(f16_2d, py_f_0d), onp.Array1D[np.float64])  # X
assert_type(entropy(f16_3d, py_f_0d), onp.Array2D[np.float64])  # X
assert_type(entropy(f16_nd, py_f_0d), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(entropy(f16_nd, py_f_0d, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(f16_nd, py_f_0d, axis=None), np.float64)

assert_type(entropy(f32_0d, py_f_0d), np.float64)
assert_type(entropy(f32_1d, py_f_0d), np.float64)
assert_type(entropy(f32_2d, py_f_0d), onp.Array1D[np.float64])  # X
assert_type(entropy(f32_3d, py_f_0d), onp.Array2D[np.float64])  # X
assert_type(entropy(f32_nd, py_f_0d), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(entropy(f32_nd, py_f_0d, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(f32_nd, py_f_0d, axis=None), np.float64)

assert_type(entropy(f64_0d, py_f_0d), np.float64)
assert_type(entropy(f64_1d, py_f_0d), np.float64)
assert_type(entropy(f64_2d, py_f_0d), onp.Array1D[np.float64])  # X
assert_type(entropy(f64_3d, py_f_0d), onp.Array2D[np.float64])  # X
assert_type(entropy(f64_nd, py_f_0d), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(entropy(f64_nd, py_f_0d, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(f64_nd, py_f_0d, axis=None), np.float64)

assert_type(entropy(py_f_0d, py_f_0d), np.float64)
assert_type(entropy(py_f_1d, py_f_0d), np.float64)
assert_type(entropy(py_f_2d, py_f_0d), onp.Array1D[np.float64])
assert_type(entropy(py_f_3d, py_f_0d), onp.Array2D[np.float64])
assert_type(entropy(py_f_nd, py_f_0d), np.float64 | onp.ArrayND[np.float64])
assert_type(entropy(py_f_nd, py_f_0d, keepdims=True), onp.ArrayND[np.float64])
assert_type(entropy(py_f_nd, py_f_0d, axis=None), np.float64)

###
# differential_entropy

assert_type(differential_entropy(i8_0d), np.float64)
assert_type(differential_entropy(i8_1d), np.float64)
assert_type(differential_entropy(i8_2d), onp.Array1D[np.float64])  # X
assert_type(differential_entropy(i8_3d), onp.Array2D[np.float64])  # X
assert_type(differential_entropy(i8_nd), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(differential_entropy(i8_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(differential_entropy(i8_nd, axis=None), np.float64)

assert_type(differential_entropy(f16_0d), np.float16)
assert_type(differential_entropy(f16_1d), np.float16)
assert_type(differential_entropy(f16_2d), onp.Array1D[np.float16])  # X
assert_type(differential_entropy(f16_3d), onp.Array2D[np.float16])  # X
assert_type(differential_entropy(f16_nd), np.float16 | onp.ArrayND[np.float16])  # X
assert_type(differential_entropy(f16_nd, keepdims=True), onp.ArrayND[np.float16])
assert_type(differential_entropy(f16_nd, axis=None), np.float16)

assert_type(differential_entropy(f32_0d), np.float32)
assert_type(differential_entropy(f32_1d), np.float32)
assert_type(differential_entropy(f32_2d), onp.Array1D[np.float32])  # X
assert_type(differential_entropy(f32_3d), onp.Array2D[np.float32])  # X
assert_type(differential_entropy(f32_nd), np.float32 | onp.ArrayND[np.float32])  # X
assert_type(differential_entropy(f32_nd, keepdims=True), onp.ArrayND[np.float32])
assert_type(differential_entropy(f32_nd, axis=None), np.float32)

assert_type(differential_entropy(f64_0d), np.float64)
assert_type(differential_entropy(f64_1d), np.float64)
assert_type(differential_entropy(f64_2d), onp.Array1D[np.float64])  # X
assert_type(differential_entropy(f64_3d), onp.Array2D[np.float64])  # X
assert_type(differential_entropy(f64_nd), np.float64 | onp.ArrayND[np.float64])  # X
assert_type(differential_entropy(f64_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(differential_entropy(f64_nd, axis=None), np.float64)

assert_type(differential_entropy(f80_0d), np.float128)
assert_type(differential_entropy(f80_1d), np.float128)
assert_type(differential_entropy(f80_2d), onp.Array1D[np.float128])  # X
assert_type(differential_entropy(f80_3d), onp.Array2D[np.float128])  # X
assert_type(differential_entropy(f80_nd), np.float128 | onp.ArrayND[np.float128])  # X
assert_type(differential_entropy(f80_nd, keepdims=True), onp.ArrayND[np.float128])
assert_type(differential_entropy(f80_nd, axis=None), np.float128)

assert_type(differential_entropy(c64_0d), np.complex64)
assert_type(differential_entropy(c64_1d), np.complex64)
assert_type(differential_entropy(c64_2d), onp.Array1D[np.complex64])  # X
assert_type(differential_entropy(c64_3d), onp.Array2D[np.complex64])  # X
assert_type(differential_entropy(c64_nd), np.complex64 | onp.ArrayND[np.complex64])  # X
assert_type(differential_entropy(c64_nd, keepdims=True), onp.ArrayND[np.complex64])
assert_type(differential_entropy(c64_nd, axis=None), np.complex64)

assert_type(differential_entropy(py_f_0d), np.float64)
assert_type(differential_entropy(py_f_1d), np.float64)
assert_type(differential_entropy(py_f_2d), onp.Array1D[np.float64])
assert_type(differential_entropy(py_f_3d), onp.Array2D[np.float64])
assert_type(differential_entropy(py_f_nd), np.float64 | onp.ArrayND[np.float64])
assert_type(differential_entropy(py_f_nd, keepdims=True), onp.ArrayND[np.float64])
assert_type(differential_entropy(py_f_nd, axis=None), np.float64)

assert_type(differential_entropy(py_c_0d), np.complex128)
assert_type(differential_entropy(py_c_1d), np.complex128)
assert_type(differential_entropy(py_c_2d), onp.Array1D[np.complex128])
assert_type(differential_entropy(py_c_3d), onp.Array2D[np.complex128])
assert_type(differential_entropy(py_c_nd), np.complex128 | onp.ArrayND[np.complex128])
assert_type(differential_entropy(py_c_nd, keepdims=True), onp.ArrayND[np.complex128])
assert_type(differential_entropy(py_c_nd, axis=None), np.complex128)
