# regression tests for https://github.com/scipy/scipy-stubs/issues/817

from typing import assert_type

import numpy as np

from ._types import ScalarType, lil_arr, lil_mat
from scipy.sparse import lil_array, lil_matrix

dtype: np.dtype[ScalarType]

shape2: tuple[int, int]

data2: np.ndarray[tuple[int, int], np.dtype[ScalarType]]

###
# LIL matrix constructor
# ruff: noqa: ERA001

# lil_matrix(D)
assert_type(lil_matrix(data2), lil_matrix[ScalarType])
assert_type(lil_matrix(data2, dtype=dtype), lil_matrix[ScalarType])
assert_type(lil_matrix(data2, dtype=bool), lil_matrix[np.bool_])
assert_type(lil_matrix(data2, dtype=int), lil_matrix[np.int_])
assert_type(lil_matrix(data2, dtype=float), lil_matrix[np.float64])
assert_type(lil_matrix(data2, dtype=complex), lil_matrix[np.complex128])

# lil_matrix(S)
assert_type(lil_matrix(lil_arr), lil_matrix[ScalarType])
assert_type(lil_matrix(lil_mat), lil_matrix[ScalarType])

# lil_matrix((M, N), [dtype])
assert_type(lil_matrix(shape2), lil_matrix[np.float64])
assert_type(lil_matrix(shape2, dtype=dtype), lil_matrix[ScalarType])
assert_type(lil_matrix(shape2, dtype=bool), lil_matrix[np.bool_])
assert_type(lil_matrix(shape2, dtype=int), lil_matrix[np.int_])
assert_type(lil_matrix(shape2, dtype=float), lil_matrix[np.float64])
assert_type(lil_matrix(shape2, dtype=complex), lil_matrix[np.complex128])

###
# LIL array constructor

# lil_array(D)
assert_type(lil_array(data2), lil_array[ScalarType])
assert_type(lil_array(data2, dtype=dtype), lil_array[ScalarType])
assert_type(lil_array(data2, dtype=bool), lil_array[np.bool_])
assert_type(lil_array(data2, dtype=int), lil_array[np.int_])
assert_type(lil_array(data2, dtype=float), lil_array[np.float64])
assert_type(lil_array(data2, dtype=complex), lil_array[np.complex128])

# lil_array(S)
assert_type(lil_array(lil_arr), lil_array[ScalarType])
assert_type(lil_array(lil_mat), lil_array[ScalarType])

# lil_array((M, N), [dtype])
assert_type(lil_array(shape2), lil_array[np.float64])
assert_type(lil_array(shape2, dtype=dtype), lil_array[ScalarType])
assert_type(lil_array(shape2, dtype=bool), lil_array[np.bool_])
assert_type(lil_array(shape2, dtype=int), lil_array[np.int_])
assert_type(lil_array(shape2, dtype=float), lil_array[np.float64])
assert_type(lil_array(shape2, dtype=complex), lil_array[np.complex128])
