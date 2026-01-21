# type-tests for `signal/_signaltools.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import lfilter, lfilter_zi, lfiltic

###

_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]

_i16_1d: onp.Array1D[np.int16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]
_c64_1d: onp.Array1D[np.complex64]
_c128_1d: onp.Array1D[np.complex128]

###

# lfilter_zi
assert_type(lfilter_zi(_py_i_1d, _py_i_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_py_f_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_i16_1d, _i16_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_i16_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(lfilter_zi(_f32_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(lfilter_zi(_f32_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(lfilter_zi(_f32_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(lfilter_zi(_c64_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(lfilter_zi(_c64_1d, _c128_1d), onp.Array1D[np.complex128])
assert_type(lfilter_zi(_c128_1d, _c128_1d), onp.Array1D[np.complex128])

# lfiltic
assert_type(lfiltic(_py_i_1d, _py_i_1d, _py_i_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_py_f_1d, _py_f_1d, _py_f_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_i16_1d, _i16_1d, _i16_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_f32_1d, _f32_1d, _f32_1d), onp.Array1D[np.float32])
assert_type(lfiltic(_f32_1d, _f32_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_f64_1d, _f64_1d, _f64_1d), onp.Array1D[np.float64])
assert_type(lfiltic(_f32_1d, _f32_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(lfiltic(_c64_1d, _c64_1d, _c64_1d), onp.Array1D[np.complex64])
assert_type(lfiltic(_c64_1d, _c64_1d, _c128_1d), onp.Array1D[np.complex128])
assert_type(lfiltic(_c128_1d, _c128_1d, _c128_1d), onp.Array1D[np.complex128])

# lfilter
assert_type(lfilter(_py_i_1d, _py_i_1d, _py_i_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_py_f_1d, _py_f_1d, _py_f_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_i16_1d, _i16_1d, _i16_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_f32_1d, _f32_1d, _f32_1d), onp.ArrayND[np.float32])
assert_type(lfilter(_f32_1d, _f32_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_f64_1d, _f64_1d, _f64_1d), onp.ArrayND[np.float64])
assert_type(lfilter(_f32_1d, _f32_1d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(lfilter(_c64_1d, _c64_1d, _c64_1d), onp.ArrayND[np.complex64])
assert_type(lfilter(_c64_1d, _c64_1d, _c128_1d), onp.ArrayND[np.complex128])
assert_type(lfilter(_c128_1d, _c128_1d, _c128_1d), onp.ArrayND[np.complex128])
