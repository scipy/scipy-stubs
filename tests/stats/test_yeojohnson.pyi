# type-tests for `yeojohnson*` from `stats/_morestats.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import yeojohnson, yeojohnson_llf, yeojohnson_normmax, yeojohnson_normplot

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

_i8_nd: onp.ArrayND[np.int8]
_f16_nd: onp.ArrayND[np.float16]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]

###

# yeojohnson_llf

assert_type(yeojohnson_llf(0.5, _py_float_1d), np.float64)
assert_type(yeojohnson_llf(0.5, _i8_1d), np.float64)
assert_type(yeojohnson_llf(0.5, _f16_1d), np.float16)
assert_type(yeojohnson_llf(0.5, _f32_1d), np.float32)
assert_type(yeojohnson_llf(0.5, _f64_1d), np.float64)

assert_type(yeojohnson_llf(0.5, _py_float_1d, axis=None), np.float64)
assert_type(yeojohnson_llf(0.5, _i8_1d, axis=None), np.float64)
assert_type(yeojohnson_llf(0.5, _f16_1d, axis=None), np.float16)
assert_type(yeojohnson_llf(0.5, _f32_1d, axis=None), np.float32)
assert_type(yeojohnson_llf(0.5, _f64_1d, axis=None), np.float64)

assert_type(yeojohnson_llf(0.5, _py_float_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(yeojohnson_llf(0.5, _i8_1d, keepdims=True), onp.ArrayND[np.float64])
assert_type(yeojohnson_llf(0.5, _f16_1d, keepdims=True), onp.ArrayND[np.float16])
assert_type(yeojohnson_llf(0.5, _f32_1d, keepdims=True), onp.ArrayND[np.float32])
assert_type(yeojohnson_llf(0.5, _f64_1d, keepdims=True), onp.ArrayND[np.float64])

assert_type(yeojohnson_llf(0.5, _py_float_2d), onp.Array1D[np.float64])
assert_type(yeojohnson_llf(0.5, _i8_2d), onp.Array1D[np.float64])
assert_type(yeojohnson_llf(0.5, _f16_2d), onp.Array1D[np.float16])
assert_type(yeojohnson_llf(0.5, _f32_2d), onp.Array1D[np.float32])
assert_type(yeojohnson_llf(0.5, _f64_2d), onp.Array1D[np.float64])

assert_type(yeojohnson_llf(0.5, _py_float_2d, axis=None), np.float64)
assert_type(yeojohnson_llf(0.5, _i8_2d, axis=None), np.float64)
assert_type(yeojohnson_llf(0.5, _f16_2d, axis=None), np.float16)
assert_type(yeojohnson_llf(0.5, _f32_2d, axis=None), np.float32)
assert_type(yeojohnson_llf(0.5, _f64_2d, axis=None), np.float64)

assert_type(yeojohnson_llf(0.5, _py_float_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(yeojohnson_llf(0.5, _i8_2d, keepdims=True), onp.ArrayND[np.float64])
assert_type(yeojohnson_llf(0.5, _f16_2d, keepdims=True), onp.ArrayND[np.float16])
assert_type(yeojohnson_llf(0.5, _f32_2d, keepdims=True), onp.ArrayND[np.float32])
assert_type(yeojohnson_llf(0.5, _f64_2d, keepdims=True), onp.ArrayND[np.float64])

# yeojohnson

assert_type(yeojohnson(_i8_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(yeojohnson(_f16_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(yeojohnson(_f32_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(yeojohnson(_f64_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(yeojohnson(_i8_1d, 0.1), onp.Array1D[np.float64])
assert_type(yeojohnson(_f16_1d, 0.1), onp.Array1D[np.float64])
assert_type(yeojohnson(_f32_1d, 0.1), onp.Array1D[np.float64])
assert_type(yeojohnson(_f64_1d, 0.1), onp.Array1D[np.float64])

# yeojohnson_normmax

assert_type(yeojohnson_normmax(_py_float_1d), np.float64)
assert_type(yeojohnson_normmax(_i8_1d), np.float64)
assert_type(yeojohnson_normmax(_f16_1d), np.float64)
assert_type(yeojohnson_normmax(_f32_1d), np.float64)
assert_type(yeojohnson_normmax(_f64_1d), np.float64)

assert_type(yeojohnson_normmax(_py_float_2d), onp.Array1D[np.float64])
assert_type(yeojohnson_normmax(_i8_2d), onp.Array1D[np.float64])
assert_type(yeojohnson_normmax(_f16_2d), onp.Array1D[np.float64])
assert_type(yeojohnson_normmax(_f32_2d), onp.Array1D[np.float64])
assert_type(yeojohnson_normmax(_f64_2d), onp.Array1D[np.float64])

# NOTE: Pyrefly doesn't seem to be able to intersect the return types in case of multiple matching overloads,
# and in this case both return types are even identical (Array1D[float64] | float64).
assert_type(yeojohnson_normmax(_i8_nd), onp.Array1D[np.float64] | np.float64)  # pyrefly:ignore[assert-type]
assert_type(yeojohnson_normmax(_f16_nd), onp.Array1D[np.float64] | np.float64)  # pyrefly:ignore[assert-type]
assert_type(yeojohnson_normmax(_f32_nd), onp.Array1D[np.float64] | np.float64)  # pyrefly:ignore[assert-type]
assert_type(yeojohnson_normmax(_f64_nd), onp.Array1D[np.float64] | np.float64)  # pyrefly:ignore[assert-type]

# yeojohnson_plot

assert_type(yeojohnson_normplot(_i8_1d, 0.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(yeojohnson_normplot(_f16_1d, 0.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(yeojohnson_normplot(_f32_1d, 0.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
assert_type(yeojohnson_normplot(_f64_1d, 0.0, 1.0), tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]])
