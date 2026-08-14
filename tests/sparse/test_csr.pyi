from typing import Any, Literal, assert_type

import numpy as np
import optype.numpy as onp

from ._types import ScalarType, csr_arr, csr_mat, csr_vec
from scipy.sparse import coo_array, csc_array, csc_matrix, csr_array, csr_matrix, isspmatrix

###

_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]

_py_b_2d: list[list[bool]]
_py_i_2d: list[list[int]]
_py_f_2d: list[list[float]]
_py_c_2d: list[list[complex]]

_f32_1d: onp.Array1D[np.float32]
_f32_nd: onp.ArrayND[np.float32]

###
# NOTE: Keep these tests in sync with the `dok` tests.

# pyrefly: ignore [no-matching-overload]
csr_array(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
csr_matrix(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(csr_array((2,)), csr_array[np.float64, tuple[int]])
assert_type(csr_array((2, 3)), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array((2,), dtype=np.bool), csr_array[np.bool, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.bool), csr_array[np.bool, tuple[int, int]])
assert_type(csr_array((2,), dtype=bool), csr_array[np.bool, tuple[int]])
assert_type(csr_array((2, 3), dtype=bool), csr_array[np.bool, tuple[int, int]])
assert_type(csr_array((2,), dtype="bool"), csr_array[np.bool, tuple[int]])
assert_type(csr_array((2, 3), dtype="bool"), csr_array[np.bool, tuple[int, int]])
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
assert_type(csr_array((2,), dtype=np.int8), csr_array[np.int8, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.int8), csr_array[np.int8, tuple[int, int]])
assert_type(csr_array((2,), dtype=np.uint8), csr_array[np.uint8, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.uint8), csr_array[np.uint8, tuple[int, int]])
assert_type(csr_array((2,), dtype=np.float32), csr_array[np.float32, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.float32), csr_array[np.float32, tuple[int, int]])
assert_type(csr_array((2,), dtype=np.complex64), csr_array[np.complex64, tuple[int]])
assert_type(csr_array((2, 3), dtype=np.complex64), csr_array[np.complex64, tuple[int, int]])
assert_type(csr_matrix((2, 3)), csr_matrix[np.float64])
assert_type(csr_matrix((2, 3), dtype=np.bool), csr_matrix[np.bool])
assert_type(csr_matrix((2, 3), dtype=np.int64), csr_matrix[np.int64])
assert_type(csr_matrix((2, 3), dtype=np.float64), csr_matrix[np.float64])
assert_type(csr_matrix((2, 3), dtype=np.complex128), csr_matrix[np.complex128])
assert_type(csr_matrix((2, 3), dtype=np.int8), csr_matrix[np.int8])
assert_type(csr_matrix((2, 3), dtype=np.uint8), csr_matrix[np.uint8])
assert_type(csr_matrix((2, 3), dtype=np.float32), csr_matrix[np.float32])
assert_type(csr_matrix((2, 3), dtype=np.complex64), csr_matrix[np.complex64])

assert_type(csr_array(_py_b_1d), csr_array[np.bool, tuple[int]])
assert_type(csr_array(_py_i_1d), csr_array[np.int64, tuple[int]])
assert_type(csr_array(_py_f_1d), csr_array[np.float64, tuple[int]])
assert_type(csr_array(_py_c_1d), csr_array[np.complex128, tuple[int]])

assert_type(csr_array(_py_b_2d), csr_array[np.bool, tuple[int, int]])
assert_type(csr_array(_py_i_2d), csr_array[np.int64, tuple[int, int]])
assert_type(csr_array(_py_f_2d), csr_array[np.float64, tuple[int, int]])
assert_type(csr_array(_py_c_2d), csr_array[np.complex128, tuple[int, int]])

assert_type(csr_matrix(_py_b_2d), csr_matrix[np.bool])
assert_type(csr_matrix(_py_i_2d), csr_matrix[np.int64])
assert_type(csr_matrix(_py_f_2d), csr_matrix[np.float64])
assert_type(csr_matrix(_py_c_2d), csr_matrix[np.complex128])

# https://github.com/scipy/scipy-stubs/issues/1060

assert_type(csr_array((_f32_nd, (_py_i_1d, _py_i_1d))), csr_array[np.float32])
assert_type(csr_array((_f32_1d, (_py_i_1d, _py_i_1d))), csr_array[np.float32])
assert_type(csr_array((_py_b_1d, (_py_i_1d, _py_i_1d))), csr_array[np.bool])
assert_type(csr_array((_py_i_1d, (_py_i_1d, _py_i_1d))), csr_array[np.int64])
assert_type(csr_array((_py_f_1d, (_py_i_1d, _py_i_1d))), csr_array[np.float64])
assert_type(csr_array((_py_c_1d, (_py_i_1d, _py_i_1d))), csr_array[np.complex128])
# pyrefly: ignore [no-matching-overload]
csr_array((_py_b_2d, (_py_i_1d, _py_i_1d)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
csr_array((_py_i_2d, (_py_i_1d, _py_i_1d)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
csr_array((_py_f_2d, (_py_i_1d, _py_i_1d)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
csr_array((_py_c_2d, (_py_i_1d, _py_i_1d)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(csr_matrix((_f32_nd, (_py_i_1d, _py_i_1d))), csr_matrix[np.float32])
assert_type(csr_matrix((_f32_1d, (_py_i_1d, _py_i_1d))), csr_matrix[np.float32])
assert_type(csr_matrix((_py_b_1d, (_py_i_1d, _py_i_1d))), csr_matrix[np.bool])
assert_type(csr_matrix((_py_i_1d, (_py_i_1d, _py_i_1d))), csr_matrix[np.int64])
assert_type(csr_matrix((_py_f_1d, (_py_i_1d, _py_i_1d))), csr_matrix[np.float64])
assert_type(csr_matrix((_py_c_1d, (_py_i_1d, _py_i_1d))), csr_matrix[np.complex128])
# pyrefly: ignore [no-matching-overload]
csr_matrix((_py_b_2d, (_py_i_1d, _py_i_1d)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
csr_matrix((_py_i_2d, (_py_i_1d, _py_i_1d)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
csr_matrix((_py_f_2d, (_py_i_1d, _py_i_1d)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
csr_matrix((_py_c_2d, (_py_i_1d, _py_i_1d)))  # type: ignore[type-var] # pyright: ignore[reportArgumentType, reportCallIssue]

###
# CSR-specific tests

type _Index1D = onp.Array1D[np.intp]

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
# pyrefly: ignore [no-matching-overload]
csr_mat.getnnz(2)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
csr_mat.getnnz(-3)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]

assert_type(isspmatrix(csr_arr), bool)
assert_type(isspmatrix(csr_mat), bool)
assert_type(isspmatrix(object()), bool)

# __getitem__
assert_type(csr_vec[0], ScalarType)
assert_type(csr_vec[()], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[:], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[...], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[_py_b_1d], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[_py_i_1d], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[0, None], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[None, 0], csr_array[ScalarType, tuple[int]])
assert_type(csr_vec[None], coo_array[ScalarType, tuple[int, int]])

# pyrefly: ignore [bad-index]
csr_arr[None]  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(csr_arr[0, 0], ScalarType)
assert_type(csr_arr[0], coo_array[ScalarType, tuple[int]])
assert_type(csr_arr[0, _py_i_1d], coo_array[ScalarType, tuple[int]])
assert_type(csr_arr[_py_i_1d, 0], coo_array[ScalarType, tuple[int]])
assert_type(csr_arr[()], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[:], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[...], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[_py_b_1d], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[_py_i_1d], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[0, None], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[None, 0], csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr[_py_i_1d, _py_i_1d], np.ndarray[tuple[int], np.dtype[ScalarType]])

# pyrefly: ignore [bad-index]
csr_mat[None]  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
assert_type(csr_mat[0, 0], ScalarType)
assert_type(csr_mat[0], csr_matrix[ScalarType])
assert_type(csr_mat[0, _py_i_1d], csr_matrix[ScalarType])
assert_type(csr_mat[_py_i_1d, 0], csr_matrix[ScalarType])
assert_type(csr_mat[()], csr_matrix[ScalarType])
assert_type(csr_mat[:], csr_matrix[ScalarType])
assert_type(csr_mat[...], csr_matrix[ScalarType])
assert_type(csr_mat[_py_b_1d], csr_matrix[ScalarType])
assert_type(csr_mat[_py_i_1d], csr_matrix[ScalarType])
assert_type(csr_mat[0, None], csr_matrix[ScalarType])
assert_type(csr_mat[None, 0], csr_matrix[ScalarType])
assert_type(csr_mat[_py_i_1d, _py_i_1d], np.matrix[tuple[int, int], np.dtype[ScalarType]])

# T
assert_type(csr_vec.T, csr_array[ScalarType, tuple[int]])
assert_type(csr_arr.T, csc_array[ScalarType])
assert_type(csr_mat.T, csc_matrix[ScalarType])

# transpose
assert_type(csr_vec.transpose(), csr_array[ScalarType, tuple[int]])
assert_type(csr_arr.transpose(), csc_array[ScalarType])
assert_type(csr_mat.transpose(), csc_matrix[ScalarType])

# sum
_csr_arr_any: csr_array[Any]
_csr_arr_bool: csr_array[np.bool]
_csr_arr_i8: csr_array[np.int8]
_csr_arr_u8: csr_array[np.uint8]
_csr_arr_f32: csr_array[np.float32]

assert_type(_csr_arr_any.sum(), Any)
assert_type(_csr_arr_bool.sum(), np.int_)
assert_type(_csr_arr_i8.sum(), np.int_)
assert_type(_csr_arr_u8.sum(), np.uint64)
assert_type(_csr_arr_f32.sum(), np.float32)

assert_type(_csr_arr_any.sum(0), onp.Array1D[Any])
assert_type(_csr_arr_bool.sum(0), onp.Array1D[np.int_])
assert_type(_csr_arr_i8.sum(0), onp.Array1D[np.int_])
assert_type(_csr_arr_u8.sum(0), onp.Array1D[np.uint64])
assert_type(_csr_arr_f32.sum(0), onp.Array1D[np.float32])  # type: ignore[assert-type]  # mypy bug

_csr_arr_c64: csr_array[np.complex64]
assert_type(_csr_arr_any.mean(), Any)
assert_type(_csr_arr_bool.mean(), np.float64)
assert_type(_csr_arr_i8.mean(), np.float64)
assert_type(_csr_arr_u8.mean(), np.float64)
assert_type(_csr_arr_f32.mean(), np.float32)
assert_type(_csr_arr_c64.mean(), np.complex64)
assert_type(_csr_arr_i8.mean(0), onp.Array1D[np.float64])
assert_type(_csr_arr_any.mean(0), onp.Array1D[Any])
