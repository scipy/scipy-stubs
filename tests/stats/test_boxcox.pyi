# type-tests for `boxcox*` from `stats/_morestats.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import boxcox, boxcox_llf, boxcox_normmax, boxcox_normplot

###

_py_float_1d: list[float]
_i8_1d: onp.Array1D[np.int8]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]

_py_float_2d: list[list[float]]
_i8_2d: onp.Array2D[np.int8]
_f16_2d: onp.Array2D[np.float16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]

###

# boxcox_llf

assert_type(boxcox_llf(0.5, _py_float_1d), np.float64)
assert_type(boxcox_llf(0.5, _i8_1d), np.float64)
assert_type(boxcox_llf(0.5, _f16_1d), np.float16)
assert_type(boxcox_llf(0.5, _f32_1d), np.float32)
assert_type(boxcox_llf(0.5, _f64_1d), np.float64)

assert_type(boxcox_llf(0.5, _py_float_1d, axis=None), np.float64)
assert_type(boxcox_llf(0.5, _i8_1d, axis=None), np.float64)
assert_type(boxcox_llf(0.5, _f16_1d, axis=None), np.float16)
assert_type(boxcox_llf(0.5, _f32_1d, axis=None), np.float32)
assert_type(boxcox_llf(0.5, _f64_1d, axis=None), np.float64)

assert_type(boxcox_llf(0.5, _py_float_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(boxcox_llf(0.5, _i8_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(boxcox_llf(0.5, _f16_1d, keepdims=True), onp.ArrayND[np.float16])
assert_type(boxcox_llf(0.5, _f32_1d, keepdims=True), onp.ArrayND[np.float32])
assert_type(boxcox_llf(0.5, _f64_1d, keepdims=True), onp.ArrayND[np.float64])

assert_type(boxcox_llf(0.5, _py_float_2d), onp.Array1D[np.float64])
assert_type(boxcox_llf(0.5, _i8_2d), onp.Array1D[np.float64])
assert_type(boxcox_llf(0.5, _f16_2d), onp.Array1D[np.float16])
assert_type(boxcox_llf(0.5, _f32_2d), onp.Array1D[np.float32])
assert_type(boxcox_llf(0.5, _f64_2d), onp.Array1D[np.float64])

assert_type(boxcox_llf(0.5, _py_float_2d, axis=None), np.float64)
assert_type(boxcox_llf(0.5, _i8_2d, axis=None), np.float64)
assert_type(boxcox_llf(0.5, _f16_2d, axis=None), np.float16)
assert_type(boxcox_llf(0.5, _f32_2d, axis=None), np.float32)
assert_type(boxcox_llf(0.5, _f64_2d, axis=None), np.float64)

assert_type(boxcox_llf(0.5, _py_float_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(boxcox_llf(0.5, _i8_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(boxcox_llf(0.5, _f16_2d, keepdims=True), onp.ArrayND[np.float16])
assert_type(boxcox_llf(0.5, _f32_2d, keepdims=True), onp.ArrayND[np.float32])
assert_type(boxcox_llf(0.5, _f64_2d, keepdims=True), onp.ArrayND[np.float64])

# boxcox

assert_type(boxcox(_i8_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(boxcox(_f16_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(boxcox(_f32_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(boxcox(_f64_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(boxcox(_i8_1d, alpha=0.1), tuple[onp.Array1D[np.float64], np.float64, tuple[float, float]])
assert_type(boxcox(_f16_1d, alpha=0.1), tuple[onp.Array1D[np.float64], np.float64, tuple[float, float]])
assert_type(boxcox(_f32_1d, alpha=0.1), tuple[onp.Array1D[np.float64], np.float64, tuple[float, float]])
assert_type(boxcox(_f64_1d, alpha=0.1), tuple[onp.Array1D[np.float64], np.float64, tuple[float, float]])

# boxcox_normmax

assert_type(boxcox_normmax(_i8_1d), np.float64)
assert_type(boxcox_normmax(_f16_1d), np.float64)
assert_type(boxcox_normmax(_f32_1d), np.float64)
assert_type(boxcox_normmax(_f64_1d), np.float64)
assert_type(boxcox_normmax(_i8_1d, method="all"), onp.Array1D[np.float64])
assert_type(boxcox_normmax(_f16_1d, method="all"), onp.Array1D[np.float64])
assert_type(boxcox_normmax(_f32_1d, method="all"), onp.Array1D[np.float64])
assert_type(boxcox_normmax(_f64_1d, method="all"), onp.Array1D[np.float64])

# boxcox_plot

assert_type(boxcox_normplot(_i8_1d, 0.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(boxcox_normplot(_f16_1d, 0.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(boxcox_normplot(_f32_1d, 0.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(boxcox_normplot(_f64_1d, 0.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
