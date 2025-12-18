from typing import Any, Literal, TypeAlias, assert_type

import numpy as np

from ._types import ScalarType, csr_arr, csr_mat, csr_vec
from scipy.sparse import coo_array, csc_array, csc_matrix, csr_array, csr_matrix, isspmatrix

###

seq_bool: list[bool]
seq_int: list[int]
seq_float: list[float]
seq_complex: list[complex]

seq_seq_bool: list[list[bool]]
seq_seq_int: list[list[int]]
seq_seq_float: list[list[float]]
seq_seq_complex: list[list[complex]]

arr_f32_nd: np.ndarray[tuple[Any, ...], np.dtype[np.float32]]
arr_f32_1d: np.ndarray[tuple[int], np.dtype[np.float32]]

###
# NOTE: Keep these tests in sync with the `dok` tests.

csr_array(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
csr_matrix(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(csr_array((2,)), csr_array[np.float64, tuple[int]])
assert_type(csr_array((2, 3)), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array((2,), dtype=np.bool_), csr_array[np.bool_, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.bool_), csr_array[np.bool_, tuple[int, int]])
assert_type(csr_array((2,), dtype=bool), csr_array[np.bool_, tuple[int]])
assert_type(csr_array((2, 3), dtype=bool), csr_array[np.bool_, tuple[int, int]])
assert_type(csr_array((2,), dtype="bool"), csr_array[np.bool_, tuple[int]])
assert_type(csr_array((2, 3), dtype="bool"), csr_array[np.bool_, tuple[int, int]])
assert_type(csr_array((2,), dtype=np.int64), csr_array[np.int64, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.int64), csr_array[np.int64, tuple[int, int]])
assert_type(csr_array((2,), dtype=int), csr_array[np.int64, tuple[int]])
assert_type(csr_array((2, 3), dtype=int), csr_array[np.int64, tuple[int, int]])
assert_type(csr_array((2,), dtype="int"), csr_array[np.int64, tuple[int]])
assert_type(csr_array((2, 3), dtype="int"), csr_array[np.int64, tuple[int, int]])
assert_type(csr_array((2,), dtype=np.float64), csr_array[np.float64, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.float64), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array((2,), dtype=float), csr_array[np.float64, tuple[int]])
assert_type(csr_array((2, 3), dtype=float), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array((2,), dtype="float"), csr_array[np.float64, tuple[int]])
assert_type(csr_array((2, 3), dtype="float"), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array((2,), dtype=np.complex128), csr_array[np.complex128, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.complex128), csr_array[np.complex128, tuple[int, int]])
assert_type(csr_array((2,), dtype=complex), csr_array[np.complex128, tuple[int]])
assert_type(csr_array((2, 3), dtype=complex), csr_array[np.complex128, tuple[int, int]])
assert_type(csr_array((2,), dtype="complex"), csr_array[np.complex128, tuple[int]])
assert_type(csr_array((2, 3), dtype="complex"), csr_array[np.complex128, tuple[int, int]])
assert_type(csr_array((2,), None, None), csr_array[np.float64, tuple[int]])
assert_type(csr_array((2, 3), None, None), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array((2,), None, np.bool_), csr_array[np.bool_, tuple[int]])
assert_type(csr_array((2, 3), None, np.bool_), csr_array[np.bool_, tuple[int, int]])
assert_type(csr_array((2,), None, np.int64), csr_array[np.int64, tuple[int]])
assert_type(csr_array((2, 3), None, np.int64), csr_array[np.int64, tuple[int, int]])
assert_type(csr_array((2,), None, np.float64), csr_array[np.float64, tuple[int]])
assert_type(csr_array((2, 3), None, np.float64), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array((2,), None, np.complex128), csr_array[np.complex128, tuple[int]])
assert_type(csr_array((2, 3), None, np.complex128), csr_array[np.complex128, tuple[int, int]])
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

# https://github.com/scipy/scipy-stubs/issues/1060

assert_type(csr_array((arr_f32_nd, (seq_int, seq_int))), csr_array[np.float32])
assert_type(csr_array((arr_f32_1d, (seq_int, seq_int))), csr_array[np.float32])
assert_type(csr_array((seq_bool, (seq_int, seq_int))), csr_array[np.bool_])
assert_type(csr_array((seq_int, (seq_int, seq_int))), csr_array[np.int64])
assert_type(csr_array((seq_float, (seq_int, seq_int))), csr_array[np.float64])
assert_type(csr_array((seq_complex, (seq_int, seq_int))), csr_array[np.complex128])
csr_array((seq_seq_bool, (seq_int, seq_int)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
csr_array((seq_seq_int, (seq_int, seq_int)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
csr_array((seq_seq_float, (seq_int, seq_int)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
csr_array((seq_seq_complex, (seq_int, seq_int)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(csr_matrix((arr_f32_nd, (seq_int, seq_int))), csr_matrix[np.float32])
assert_type(csr_matrix((arr_f32_1d, (seq_int, seq_int))), csr_matrix[np.float32])
assert_type(csr_matrix((seq_bool, (seq_int, seq_int))), csr_matrix[np.bool_])
assert_type(csr_matrix((seq_int, (seq_int, seq_int))), csr_matrix[np.int64])
assert_type(csr_matrix((seq_float, (seq_int, seq_int))), csr_matrix[np.float64])
assert_type(csr_matrix((seq_complex, (seq_int, seq_int))), csr_matrix[np.complex128])
csr_matrix((seq_seq_bool, (seq_int, seq_int)))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType, reportCallIssue]
csr_matrix((seq_seq_int, (seq_int, seq_int)))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType, reportCallIssue]
csr_matrix((seq_seq_float, (seq_int, seq_int)))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType, reportCallIssue]
csr_matrix((seq_seq_complex, (seq_int, seq_int)))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType, reportCallIssue]

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

assert_type(isspmatrix(csr_arr), bool)  # pyrefly: ignore[assert-type]
assert_type(isspmatrix(csr_mat), bool)  # pyrefly: ignore[assert-type]
assert_type(isspmatrix(object()), bool)  # pyrefly: ignore[assert-type]

# __getitem__
assert_type(csr_vec[0], ScalarType)
assert_type(csr_vec[()], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[:], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[...], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[seq_bool], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[seq_int], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[0, None], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[None, 0], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[None], coo_array[ScalarType, tuple[int, int]])

csr_arr[None]  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(csr_arr[0, 0], ScalarType)
assert_type(csr_arr[0], coo_array[ScalarType, tuple[int]])
assert_type(csr_arr[0, seq_int], coo_array[ScalarType, tuple[int]])
assert_type(csr_arr[seq_int, 0], coo_array[ScalarType, tuple[int]])
assert_type(csr_arr[()], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[:], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[...], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[seq_bool], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[seq_int], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[0, None], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[None, 0], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[seq_int, seq_int], np.ndarray[tuple[int], np.dtype[ScalarType]])  # pyrefly: ignore[assert-type]

csr_mat[None]  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(csr_mat[0, 0], ScalarType)
assert_type(csr_mat[0], csr_matrix[ScalarType])
assert_type(csr_mat[0, seq_int], csr_matrix[ScalarType])
assert_type(csr_mat[seq_int, 0], csr_matrix[ScalarType])
assert_type(csr_mat[()], csr_matrix[ScalarType])
assert_type(csr_mat[:], csr_matrix[ScalarType])
assert_type(csr_mat[...], csr_matrix[ScalarType])
assert_type(csr_mat[seq_bool], csr_matrix[ScalarType])
assert_type(csr_mat[seq_int], csr_matrix[ScalarType])
assert_type(csr_mat[0, None], csr_matrix[ScalarType])
assert_type(csr_mat[None, 0], csr_matrix[ScalarType])
assert_type(csr_mat[seq_int, seq_int], np.matrix[tuple[int, int], np.dtype[ScalarType]])  # pyrefly: ignore[assert-type]

# T
assert_type(csr_vec.T, csr_array[ScalarType, tuple[int]])  # pyrefly: ignore[assert-type]
assert_type(csr_arr.T, csc_array[ScalarType])
assert_type(csr_mat.T, csc_matrix[ScalarType])

# transpose
assert_type(csr_vec.transpose(), csr_array[ScalarType, tuple[int]])
assert_type(csr_arr.transpose(), csc_array[ScalarType])
assert_type(csr_mat.transpose(), csc_matrix[ScalarType])
