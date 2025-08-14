# type-tests for `mode` from `stats/_stats_py.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import mode

py_f_0d: float
py_f_1d: list[float]
py_f_2d: list[list[float]]

i8_0d: np.int8
i8_1d: onp.Array1D[np.int8]
i8_2d: onp.Array2D[np.int8]

f16_0d: np.float16
f16_1d: onp.Array1D[np.float16]
f16_2d: onp.Array2D[np.float16]

###

assert_type(mode(1)[0], np.intp)
assert_type(mode(py_f_0d)[0], np.float64)
assert_type(mode(i8_0d)[0], np.int8)
assert_type(mode(f16_0d)[0], np.float16)

assert_type(mode([1])[0], np.intp)
assert_type(mode(py_f_1d)[0], np.float64)
assert_type(mode(i8_1d)[0], np.int8)
assert_type(mode(f16_1d)[0], np.float16)

assert_type(mode([[1]])[0], onp.ArrayND[np.intp])
assert_type(mode(py_f_2d)[0], onp.ArrayND[np.float64])
assert_type(mode(i8_2d)[0], onp.ArrayND[np.int8])
assert_type(mode(f16_2d)[0], onp.ArrayND[np.float16])

assert_type(mode(1, axis=None)[0], np.intp)
assert_type(mode(py_f_0d, axis=None)[0], np.float64)
assert_type(mode(i8_0d, axis=None)[0], np.int8)
assert_type(mode(f16_0d, axis=None)[0], np.float16)

assert_type(mode([1], axis=None)[0], np.intp)
assert_type(mode(py_f_1d, axis=None)[0], np.float64)
assert_type(mode(i8_1d, axis=None)[0], np.int8)
assert_type(mode(f16_1d, axis=None)[0], np.float16)

assert_type(mode([[1]], axis=None)[0], np.intp)
assert_type(mode(py_f_2d, axis=None)[0], np.float64)
assert_type(mode(i8_2d, axis=None)[0], np.int8)
assert_type(mode(f16_2d, axis=None)[0], np.float16)

assert_type(mode(1, keepdims=True)[0], onp.ArrayND[np.intp])
assert_type(mode(py_f_0d, keepdims=True)[0], onp.ArrayND[np.float64])
assert_type(mode(i8_0d, keepdims=True)[0], onp.ArrayND[np.int8])
assert_type(mode(f16_0d, keepdims=True)[0], onp.ArrayND[np.float16])

assert_type(mode([1], keepdims=True)[0], onp.ArrayND[np.intp])
assert_type(mode(py_f_1d, keepdims=True)[0], onp.ArrayND[np.float64])
assert_type(mode(i8_1d, keepdims=True)[0], onp.ArrayND[np.int8])
assert_type(mode(f16_1d, keepdims=True)[0], onp.ArrayND[np.float16])

assert_type(mode([[1]], keepdims=True)[0], onp.ArrayND[np.intp])
assert_type(mode(py_f_2d, keepdims=True)[0], onp.ArrayND[np.float64])
assert_type(mode(i8_2d, keepdims=True)[0], onp.ArrayND[np.int8])
assert_type(mode(f16_2d, keepdims=True)[0], onp.ArrayND[np.float16])
