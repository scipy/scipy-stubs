from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import coo_array

###

_py_i_1d: list[int]
_py_i_2d: list[list[int]]

###
# shape-like `arg1`; the shape-type of the tuple is reused as the output shape-type

assert_type(coo_array((2,)), coo_array[np.float64, tuple[int]])
assert_type(coo_array((2, 3)), coo_array[np.float64, tuple[int, int]])
assert_type(coo_array((2, 3, 4)), coo_array[np.float64, onp.AtLeast3D])

assert_type(coo_array((2,), dtype=np.bool), coo_array[np.bool, tuple[int]])
assert_type(coo_array((2, 3), dtype=np.bool), coo_array[np.bool, tuple[int, int]])
assert_type(coo_array((2,), dtype=np.int64), coo_array[np.int64, tuple[int]])
assert_type(coo_array((2, 3), dtype=np.int64), coo_array[np.int64, tuple[int, int]])
assert_type(coo_array((2,), dtype=np.float64), coo_array[np.float64, tuple[int]])
assert_type(coo_array((2, 3), dtype=np.float64), coo_array[np.float64, tuple[int, int]])
assert_type(coo_array((2,), dtype=np.complex128), coo_array[np.complex128, tuple[int]])
assert_type(coo_array((2, 3), dtype=np.complex128), coo_array[np.complex128, tuple[int, int]])

# dtypes without a dedicated overload family
assert_type(coo_array((2,), dtype=np.int8), coo_array[np.int8, tuple[int]])
assert_type(coo_array((2, 3), dtype=np.int8), coo_array[np.int8, tuple[int, int]])
assert_type(coo_array((2, 3, 4), dtype=np.int8), coo_array[np.int8, onp.AtLeast3D])
assert_type(coo_array((2, 3), dtype=np.uint8), coo_array[np.uint8, tuple[int, int]])
assert_type(coo_array((2, 3), dtype=np.float32), coo_array[np.float32, tuple[int, int]])
assert_type(coo_array((2, 3), dtype=np.complex64), coo_array[np.complex64, tuple[int, int]])

###
# array-like `arg1` keeps being read as data, not as a shape

assert_type(coo_array(_py_i_1d, dtype=np.int8), coo_array[np.int8, tuple[int]])
assert_type(coo_array(_py_i_2d, dtype=np.int8), coo_array[np.int8, tuple[int, int]])
