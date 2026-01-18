from typing import TypeAlias, TypeVar, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import CensoredData

###

_ScalarT = TypeVar("_ScalarT", bound=np.float64 | np.complex128)

_LeftCensored: TypeAlias = CensoredData[_ScalarT, _ScalarT, np.float64, np.float64]
_RightCensored: TypeAlias = CensoredData[_ScalarT, np.float64, _ScalarT, np.float64]
_IntervalCensored: TypeAlias = CensoredData[_ScalarT, np.float64, np.float64, _ScalarT]

###

_py_bool_1d: list[bool]
_py_float_1d: list[float]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_py_complex_1d: list[complex]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]

_py_float_2d: list[list[float]]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_py_complex_2d: list[list[complex]]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

###

# left_censored
assert_type(CensoredData.left_censored(_py_float_1d, _py_bool_1d), _LeftCensored[np.float64])
assert_type(CensoredData.left_censored(_f32_1d, _py_bool_1d), _LeftCensored[np.float64])
assert_type(CensoredData.left_censored(_f64_1d, _py_bool_1d), _LeftCensored[np.float64])
assert_type(CensoredData.left_censored(_py_complex_1d, _py_bool_1d), _LeftCensored[np.complex128])
assert_type(CensoredData.left_censored(_c64_1d, _py_bool_1d), _LeftCensored[np.complex128])
assert_type(CensoredData.left_censored(_c128_1d, _py_bool_1d), _LeftCensored[np.complex128])

# right_censored
assert_type(CensoredData.right_censored(_py_float_1d, _py_bool_1d), _RightCensored[np.float64])
assert_type(CensoredData.right_censored(_f32_1d, _py_bool_1d), _RightCensored[np.float64])
assert_type(CensoredData.right_censored(_f64_1d, _py_bool_1d), _RightCensored[np.float64])
assert_type(CensoredData.right_censored(_py_complex_1d, _py_bool_1d), _RightCensored[np.complex128])
assert_type(CensoredData.right_censored(_c64_1d, _py_bool_1d), _RightCensored[np.complex128])
assert_type(CensoredData.right_censored(_c128_1d, _py_bool_1d), _RightCensored[np.complex128])

# interval_censored
assert_type(CensoredData.interval_censored(_py_float_1d, _py_float_1d), _IntervalCensored[np.float64])
assert_type(CensoredData.interval_censored(_f32_2d, _f32_2d), _IntervalCensored[np.float64])
assert_type(CensoredData.interval_censored(_f64_2d, _f64_2d), _IntervalCensored[np.float64])
assert_type(CensoredData.interval_censored(_py_complex_1d, _py_float_1d), _IntervalCensored[np.complex128])
assert_type(CensoredData.interval_censored(_py_float_1d, _py_complex_1d), _IntervalCensored[np.complex128])
assert_type(CensoredData.interval_censored(_c64_1d, _c64_1d), _IntervalCensored[np.complex128])
assert_type(CensoredData.interval_censored(_c128_1d, _c128_1d), _IntervalCensored[np.complex128])
