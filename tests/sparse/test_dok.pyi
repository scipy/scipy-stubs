from typing import assert_type

import numpy as np

from scipy.sparse import dok_array, dok_matrix

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
