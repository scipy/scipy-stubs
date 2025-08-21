# type-tests for `resample` from `signal/_signaltools.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.signal import resample

num: int

py_i_1d: list[int]
py_i_2d: list[list[int]]
py_f_1d: list[float]
py_f_2d: list[list[float]]

i8_1d: onp.Array1D[np.int8]
i8_2d: onp.Array2D[np.int8]

f16_1d: onp.Array1D[np.float16]
f16_2d: onp.Array2D[np.float16]
f32_1d: onp.Array1D[np.float32]
f32_2d: onp.Array2D[np.float32]
f64_1d: onp.Array1D[np.float64]
f64_2d: onp.Array2D[np.float64]
f80_1d: onp.Array1D[np.float128]
f80_2d: onp.Array2D[np.float128]

c64_1d: onp.Array1D[np.complex64]
c64_2d: onp.Array2D[np.complex64]
c128_1d: onp.Array1D[np.complex128]
c128_2d: onp.Array2D[np.complex128]
c160_1d: onp.Array1D[np.complex256]
c160_2d: onp.Array2D[np.complex256]

###

assert_type(resample(py_i_1d, num), onp.ArrayND[np.float64])
assert_type(resample(py_f_1d, num), onp.ArrayND[np.float64])
assert_type(resample(i8_1d, num), onp.Array1D[np.float64])
assert_type(resample(f16_1d, num), onp.Array1D[np.float32])
assert_type(resample(f32_1d, num), onp.Array1D[np.float32])
assert_type(resample(f64_1d, num), onp.Array1D[np.float64])
assert_type(resample(f80_1d, num), onp.Array1D[np.float128])
assert_type(resample(c64_1d, num), onp.Array1D[np.complex64])
assert_type(resample(c128_1d, num), onp.Array1D[np.complex128])
assert_type(resample(c160_1d, num), onp.Array1D[np.complex256])

assert_type(resample(py_i_2d, num), onp.ArrayND[np.float64])
assert_type(resample(py_f_2d, num), onp.ArrayND[np.float64])
assert_type(resample(i8_2d, num), onp.Array2D[np.float64])
assert_type(resample(f16_2d, num), onp.Array2D[np.float32])
assert_type(resample(f32_2d, num), onp.Array2D[np.float32])
assert_type(resample(f64_2d, num), onp.Array2D[np.float64])
assert_type(resample(f80_2d, num), onp.Array2D[np.float128])
assert_type(resample(c64_2d, num), onp.Array2D[np.complex64])
assert_type(resample(c128_2d, num), onp.Array2D[np.complex128])
assert_type(resample(c160_2d, num), onp.Array2D[np.complex256])
