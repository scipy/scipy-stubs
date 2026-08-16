from typing import assert_type

import numpy as np
import optype.numpy as onp

from ._types import coo_arr, coo_vec
from scipy.sparse import coo_array, coo_matrix

###

_py_i_1d: list[int]
_py_i_2d: list[list[int]]

###
# coo_array

assert_type(coo_array([True]), coo_array[np.bool, tuple[int]])
assert_type(coo_array([1]), coo_array[np.int_, tuple[int]])
assert_type(coo_array([1.0]), coo_array[np.float64, tuple[int]])
assert_type(coo_array([1j]), coo_array[np.complex128, tuple[int]])
assert_type(coo_array([[True]]), coo_array[np.bool, tuple[int, int]])
assert_type(coo_array([[1]]), coo_array[np.int_, tuple[int, int]])
assert_type(coo_array([[1.0]]), coo_array[np.float64, tuple[int, int]])
assert_type(coo_array([[1j]]), coo_array[np.complex128, tuple[int, int]])

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

assert_type(coo_array((2,), dtype=np.int8), coo_array[np.int8, tuple[int]])
assert_type(coo_array((2, 3), dtype=np.int8), coo_array[np.int8, tuple[int, int]])
assert_type(coo_array((2, 3, 4), dtype=np.int8), coo_array[np.int8, onp.AtLeast3D])
assert_type(coo_array((2, 3), dtype=np.uint8), coo_array[np.uint8, tuple[int, int]])
assert_type(coo_array((2, 3), dtype=np.float32), coo_array[np.float32, tuple[int, int]])
assert_type(coo_array((2, 3), dtype=np.complex64), coo_array[np.complex64, tuple[int, int]])

assert_type(coo_array(_py_i_1d, dtype=np.int8), coo_array[np.int8, tuple[int]])
assert_type(coo_array(_py_i_2d, dtype=np.int8), coo_array[np.int8, tuple[int, int]])

assert_type(coo_vec.count_nonzero(), np.intp)
assert_type(coo_arr.count_nonzero(), np.intp)
assert_type(coo_arr.count_nonzero(axis=0), onp.Array1D[np.intp])
# pyrefly: ignore [no-matching-overload]
coo_vec.count_nonzero(axis=0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

###
# coo_matrix

assert_type(coo_matrix([[True]]), coo_matrix[np.bool])
assert_type(coo_matrix([[1]]), coo_matrix[np.int_])
assert_type(coo_matrix([[1.0]]), coo_matrix[np.float64])
assert_type(coo_matrix([[1j]]), coo_matrix[np.complex128])

assert_type(coo_matrix((2, 3)), coo_matrix[np.float64])
assert_type(coo_matrix((2, 3), dtype=np.bool), coo_matrix[np.bool])
assert_type(coo_matrix((2, 3), dtype=np.int64), coo_matrix[np.int64])
assert_type(coo_matrix((2, 3), dtype=np.float64), coo_matrix[np.float64])
assert_type(coo_matrix((2, 3), dtype=np.complex128), coo_matrix[np.complex128])
assert_type(coo_matrix((2, 3), dtype=np.int8), coo_matrix[np.int8])
assert_type(coo_matrix((2, 3), dtype=np.uint8), coo_matrix[np.uint8])
assert_type(coo_matrix((2, 3), dtype=np.float32), coo_matrix[np.float32])
assert_type(coo_matrix((2, 3), dtype=np.complex64), coo_matrix[np.complex64])
