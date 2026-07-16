# ruff: noqa: ERA001

from typing import assert_type

import numpy as np
import optype.numpy.compat as npc

import scipy.sparse as sparse
from ._types import ScalarType, any_arr, any_mat, coo_arr, coo_mat, csc_mat, csr_arr, csr_mat, dia_mat, dok_mat, lil_mat

i64_1d: np.ndarray[tuple[int], np.dtype[np.int64]]
i64_2d: np.ndarray[tuple[int, int], np.dtype[np.int64]]
f64_2d: np.ndarray[tuple[int, int], np.dtype[np.float64]]

shape_2d: tuple[int, int]
dense_2d: np.ndarray[tuple[int, int], np.dtype[ScalarType]]

_csr_mat_bool: sparse.csr_matrix[np.bool]
_csr_mat_i16: sparse.csr_matrix[np.int16]
_csr_mat_i64: sparse.csr_matrix[np.int64]
_csr_mat_f64: sparse.csr_matrix[np.float64]
_csr_mat_f80: sparse.csr_matrix[npc.floating80]

type _SpMatrix[ScalarT: npc.number | np.bool] = (
    sparse.bsr_matrix[ScalarT]
    | sparse.coo_matrix[ScalarT]
    | sparse.csc_matrix[ScalarT]
    | sparse.csr_matrix[ScalarT]
    | sparse.dia_matrix[ScalarT]
    | sparse.dok_matrix[ScalarT]
    | sparse.lil_matrix[ScalarT]
)

###
# utility functions

assert_type(sparse.issparse(coo_mat), bool)
assert_type(sparse.issparse(coo_arr), bool)
assert_type(sparse.issparse(i64_2d), bool)

assert_type(sparse.isspmatrix_csr(csr_mat), bool)
assert_type(sparse.isspmatrix_csc(csr_mat), bool)
assert_type(sparse.isspmatrix_coo(coo_mat), bool)
assert_type(sparse.isspmatrix_dia(dia_mat), bool)
assert_type(sparse.isspmatrix_dok(dok_mat), bool)
assert_type(sparse.isspmatrix_lil(lil_mat), bool)

assert_type(sparse.issparse("duck"), bool)

###
# constructors with different formats

assert_type(sparse.coo_matrix((i64_1d, (i64_1d, i64_1d)), shape=shape_2d), sparse.coo_matrix[np.int64])
assert_type(sparse.csr_matrix(f64_2d), sparse.csr_matrix[np.float64])
assert_type(sparse.csc_matrix(f64_2d), sparse.csc_matrix[np.float64])

assert_type(sparse.bsr_matrix(shape_2d), sparse.bsr_matrix[np.float64])
assert_type(sparse.coo_matrix(shape_2d), sparse.coo_matrix[np.float64])
assert_type(sparse.csc_matrix(shape_2d), sparse.csc_matrix[np.float64])
assert_type(sparse.csr_matrix(shape_2d), sparse.csr_matrix[np.float64])
assert_type(sparse.dia_matrix(shape_2d), sparse.dia_matrix[np.float64])
assert_type(sparse.dok_matrix(shape_2d), sparse.dok_matrix[np.float64])
assert_type(sparse.lil_matrix(shape_2d), sparse.lil_matrix[np.float64])

assert_type(sparse.coo_array((i64_1d, (i64_1d, i64_1d)), shape=shape_2d), sparse.coo_array[np.int64, tuple[int, int]])
assert_type(sparse.bsr_array(f64_2d), sparse.bsr_array[np.float64])
assert_type(sparse.coo_array(f64_2d), sparse.coo_array[np.float64, tuple[int, int]])
assert_type(sparse.csc_array(f64_2d), sparse.csc_array[np.float64])
assert_type(sparse.csr_array(f64_2d), sparse.csr_array[np.float64])
assert_type(sparse.dia_array(f64_2d), sparse.dia_array[np.float64])
assert_type(sparse.lil_array(f64_2d), sparse.lil_array[np.float64])

###
# format conversion

assert_type(any_mat.tobsr(), sparse.bsr_matrix[ScalarType])
assert_type(any_mat.tocoo(), sparse.coo_matrix[ScalarType])
assert_type(any_mat.tocsc(), sparse.csc_matrix[ScalarType])
assert_type(coo_mat.tocsr(), sparse.csr_matrix[ScalarType])
assert_type(coo_mat.todia(), sparse.dia_matrix[ScalarType])
assert_type(coo_mat.todok(), sparse.dok_matrix[ScalarType])
assert_type(coo_mat.toarray(), np.ndarray[tuple[int, int], np.dtype[ScalarType]])

assert_type(any_arr.tobsr(), sparse.bsr_array[ScalarType])
assert_type(any_arr.tocoo(), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(any_arr.tocsc(), sparse.csc_array[ScalarType])
assert_type(any_arr.tocsr(), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(any_arr.todia(), sparse.dia_array[ScalarType])
assert_type(any_arr.todok(), sparse.dok_array[ScalarType, tuple[int, int]])
assert_type(any_arr.tolil(), sparse.lil_array[ScalarType])
assert_type(any_arr.toarray(), np.ndarray[tuple[int, int], np.dtype[ScalarType]])

###
# arithmetic operations

# CSR matrix
assert_type(-csr_mat, sparse.csr_matrix[ScalarType])
assert_type(round(csr_mat), sparse.csr_matrix[ScalarType])
assert_type(csr_mat + 0, sparse.csr_matrix[ScalarType])
assert_type(csr_mat - 0, sparse.csr_matrix[ScalarType])
assert_type(csr_mat * 3, sparse.csr_matrix[ScalarType])
assert_type(csr_mat**3, sparse.csr_matrix[ScalarType])
# pyrefly: ignore [unsupported-operation]
csr_mat + 1  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
# pyrefly: ignore [unsupported-operation]
csr_mat - 1  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
assert_type(csr_mat + csr_mat, sparse.csr_matrix[ScalarType])
assert_type(csr_mat - csr_mat, sparse.csr_matrix[ScalarType])
assert_type(csr_mat * csr_mat, sparse.csr_matrix[ScalarType])
assert_type(csr_mat @ csr_mat, sparse.csr_matrix[ScalarType])
assert_type(csr_mat / csr_mat, np.matrix[tuple[int, int], np.dtype[np.float64]])
assert_type(csr_mat.multiply(3), sparse.csr_matrix[ScalarType])
assert_type(csr_mat.multiply(csr_mat), sparse.csr_matrix[ScalarType])
assert_type(coo_mat.multiply(coo_mat), sparse.csr_matrix[ScalarType])
assert_type(csr_mat.multiply(dense_2d), sparse.coo_matrix[ScalarType])
assert_type(_csr_mat_bool.multiply(3), _SpMatrix[np.int_])
assert_type(_csr_mat_bool.multiply(_csr_mat_i64), _SpMatrix[np.int64])
assert_type(_csr_mat_bool.multiply(i64_2d), sparse.coo_matrix[np.int64])
assert_type(_csr_mat_i64.multiply(0.5), _SpMatrix[np.float64])
assert_type(csr_mat.multiply([[1j]]), sparse.coo_matrix[np.complex128])
assert_type(_csr_mat_i64.multiply(dense_2d), sparse.coo_matrix[np.float64])
assert_type(_csr_mat_i64.multiply(np.float32(2)), _SpMatrix[np.float64])
assert_type(csr_mat.multiply(1.5j), _SpMatrix[np.complex64])
assert_type(lil_mat.multiply(3), sparse.lil_matrix[ScalarType])
assert_type(dia_mat.multiply(dia_mat), sparse.dia_matrix[ScalarType])
assert_type(csr_mat.multiply(np.complex64(1j)), _SpMatrix[np.complex64])
assert_type(_csr_mat_bool.multiply(np.int8(1)), _SpMatrix[np.int8])
assert_type(_csr_mat_bool.multiply(i64_1d), sparse.coo_matrix[np.int64])
assert_type(_csr_mat_i16.multiply(np.float32(2)), _SpMatrix[np.float32])
assert_type(_csr_mat_i64.multiply(csc_mat), _SpMatrix[np.float64])
assert_type(_csr_mat_i64.multiply(1j), _SpMatrix[np.complex128])
assert_type(_csr_mat_f64.multiply(np.complex64(1j)), _SpMatrix[np.complex128])
assert_type(_csr_mat_f80.multiply(1j), _SpMatrix[np.clongdouble])

# CSR array
# assert_type(sparse.csr_array(dense_2d), sparse.csr_array[ScalarType])
_3: sparse.csr_array[ScalarType] = sparse.csr_array(dense_2d)

assert_type(-csr_arr, sparse.csr_array[ScalarType])
assert_type(round(csr_arr), sparse.csr_array[ScalarType])
assert_type(csr_arr + 0, sparse.csr_array[ScalarType])
assert_type(csr_arr - 0, sparse.csr_array[ScalarType])
assert_type(csr_arr * 3, sparse.csr_array[ScalarType])
assert_type(csr_arr**3, sparse.csr_array[ScalarType])
# pyrefly: ignore [unsupported-operation]
csr_arr + 1  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
# pyrefly: ignore [unsupported-operation]
csr_arr - 1  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
assert_type(csr_arr + csr_arr, sparse.csr_array[ScalarType])
assert_type(csr_arr - csr_arr, sparse.csr_array[ScalarType])
assert_type(csr_arr * csr_arr, sparse.csr_array[ScalarType])
assert_type(csr_arr @ csr_arr, sparse.csr_array[ScalarType])
assert_type(csr_arr / csr_arr, np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(csr_arr.multiply(3), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr.multiply(csr_arr), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(csr_arr.multiply(dense_2d), sparse.coo_array[ScalarType, tuple[int, int]])

# TODO(jorenham): test other arithmetic operations for all formats
