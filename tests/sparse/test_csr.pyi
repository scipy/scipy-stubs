from typing import Literal, TypeAlias, assert_type

import numpy as np

from ._types import csr_arr, csr_mat, csr_vec
from scipy.sparse import csr_array, csr_matrix, isspmatrix

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
# NOTE: Keep these tests in sync with the `dok` tests.

csr_array(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
csr_matrix(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(csr_array((2,)), csr_array[np.float64, tuple[int]])
assert_type(csr_array((2, 3)), csr_array[np.float64, tuple[int, int]])
assert_type(csr_matrix((2, 3)), csr_matrix[np.float64])

assert_type(csr_array(seq_bool), csr_array[np.bool_, tuple[int]])
assert_type(csr_array(seq_int), csr_array[np.int64, tuple[int]])
assert_type(csr_array(seq_float), csr_array[np.float64, tuple[int]])
assert_type(csr_array(seq_complex), csr_array[np.complex128, tuple[int]])

assert_type(csr_array(seq_seq_bool), csr_array[np.bool_, tuple[int, int]])
assert_type(csr_array(seq_seq_int), csr_array[np.int64, tuple[int, int]])
assert_type(csr_array(seq_seq_float), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array(seq_seq_complex), csr_array[np.complex128, tuple[int, int]])

assert_type(csr_matrix(seq_seq_bool), csr_matrix[np.bool_])
assert_type(csr_matrix(seq_seq_int), csr_matrix[np.int64])
assert_type(csr_matrix(seq_seq_float), csr_matrix[np.float64])
assert_type(csr_matrix(seq_seq_complex), csr_matrix[np.complex128])

###
# CSR-specific tests

_Index1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.intp]]

# .format
assert_type(csr_arr.format, Literal["csr"])
assert_type(csr_mat.format, Literal["csr"])

# .ndim
assert_type(csr_arr.ndim, Literal[1, 2])
assert_type(csr_mat.ndim, Literal[2])

# .count_nonzero() (defined in `_csr_base`), so no need to check for `csr_matrix`
assert_type(csr_vec.count_nonzero(), np.intp)
assert_type(csr_arr.count_nonzero(), np.intp)
assert_type(csr_vec.count_nonzero(0), np.intp)
assert_type(csr_arr.count_nonzero(0), _Index1D)
assert_type(csr_vec.count_nonzero(axis=0), np.intp)
assert_type(csr_arr.count_nonzero(axis=0), _Index1D)

# .getnnz() (only matrix)
assert_type(csr_mat.getnnz(), int)
assert_type(csr_mat.getnnz(None), int)
assert_type(csr_mat.getnnz(1), _Index1D)
assert_type(csr_mat.getnnz(0), _Index1D)
assert_type(csr_mat.getnnz(-1), _Index1D)
assert_type(csr_mat.getnnz(-2), _Index1D)
csr_mat.getnnz(2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
csr_mat.getnnz(-3)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(isspmatrix(csr_arr), bool)
assert_type(isspmatrix(csr_mat), bool)
assert_type(isspmatrix(object()), bool)
