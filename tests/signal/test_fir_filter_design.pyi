# type-tests for `signal/_fir_filter_design.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import firwin

###

_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]
_u8_1d: onp.Array1D[np.uint8]
_i16_1d: onp.Array1D[np.int16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]

_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]
_u8_2d: onp.Array2D[np.uint8]
_i16_2d: onp.Array2D[np.int16]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

###

# firwin
assert_type(firwin(1, 0.5), onp.Array1D[np.float64])
assert_type(firwin(1, _py_i_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _py_f_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _u8_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _i16_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _f32_1d), onp.Array1D[np.float64])
assert_type(firwin(1, _f64_1d), onp.Array1D[np.float64])
