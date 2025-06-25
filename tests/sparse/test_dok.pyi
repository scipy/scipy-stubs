from typing import Literal, assert_type

import numpy as np

from ._types import ScalarType, dok_arr, dok_mat, dok_vec
from scipy.sparse import dok_array, dok_matrix
from scipy.sparse._coo import coo_array

###

seq_bool: list[bool]
seq_int: list[int]
seq_float: list[float]
seq_complex: list[complex]

seq_seq_bool: list[list[bool]]
seq_seq_int: list[list[int]]
seq_seq_float: list[list[float]]
seq_seq_complex: list[list[complex]]

###
# NOTE: Keep these tests in sync with the `csr` tests.

dok_array(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
dok_matrix(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(dok_array((2,)), dok_array[np.float64, tuple[int]])
assert_type(dok_array((2, 3)), dok_array[np.float64, tuple[int, int]])
assert_type(dok_matrix((2, 3)), dok_matrix[np.float64])

assert_type(dok_array(seq_bool), dok_array[np.bool_, tuple[int]])
assert_type(dok_array(seq_int), dok_array[np.int64, tuple[int]])
assert_type(dok_array(seq_float), dok_array[np.float64, tuple[int]])
assert_type(dok_array(seq_complex), dok_array[np.complex128, tuple[int]])

assert_type(dok_array(seq_seq_bool), dok_array[np.bool_, tuple[int, int]])
assert_type(dok_array(seq_seq_int), dok_array[np.int64, tuple[int, int]])
assert_type(dok_array(seq_seq_float), dok_array[np.float64, tuple[int, int]])
assert_type(dok_array(seq_seq_complex), dok_array[np.complex128, tuple[int, int]])

assert_type(dok_matrix(seq_seq_bool), dok_matrix[np.bool_])
assert_type(dok_matrix(seq_seq_int), dok_matrix[np.int64])
assert_type(dok_matrix(seq_seq_float), dok_matrix[np.float64])
assert_type(dok_matrix(seq_seq_complex), dok_matrix[np.complex128])

###
# DOK-specific tests

# __len__
assert_type(len(dok_vec), int)
assert_type(len(dok_arr), int)
assert_type(len(dok_mat), int)

# __getitem__
assert_type(dok_vec[0], ScalarType)
assert_type(dok_arr[0], coo_array[ScalarType, tuple[int]])
assert_type(dok_mat[0], dok_matrix[ScalarType])
dok_vec[0, 0]  # type: ignore[index]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(dok_arr[0, 0], ScalarType)
assert_type(dok_mat[0, 0], ScalarType)

# .format
assert_type(dok_arr.format, Literal["dok"])
assert_type(dok_mat.format, Literal["dok"])

# .ndim
assert_type(dok_arr.ndim, Literal[1, 2])
assert_type(dok_mat.ndim, Literal[2])

# .count_nonzero() (defined in `_dok_base`), so no need to check for `dok_matrix`
assert_type(dok_vec.count_nonzero(), int)
assert_type(dok_arr.count_nonzero(), int)
dok_vec.count_nonzero(0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
dok_arr.count_nonzero(0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
dok_vec.count_nonzero(axis=0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
dok_arr.count_nonzero(axis=0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

# .get()
assert_type(dok_vec.get((0,)), ScalarType | float)
dok_arr.get((0,))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
dok_mat.get((0,))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
dok_vec.get((0, 1))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
assert_type(dok_arr.get((0, 0)), ScalarType | float)
assert_type(dok_mat.get((0, 0)), ScalarType | float)

# .T
assert_type(dok_vec.T, dok_array[ScalarType, tuple[int]])
assert_type(dok_arr.T, dok_array[ScalarType])
assert_type(dok_mat.T, dok_matrix[ScalarType])

# .transpose()
assert_type(dok_vec.transpose(), dok_array[ScalarType, tuple[int]])
assert_type(dok_arr.transpose(), dok_array[ScalarType])
assert_type(dok_mat.transpose(), dok_matrix[ScalarType])
