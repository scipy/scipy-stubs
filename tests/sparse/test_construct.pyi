from typing import assert_type

import numpy as np

import scipy.sparse as sparse

from ._types import (
    ScalarType,
    any_arr,
    any_mat,
    bsr_arr,
    bsr_mat,
    coo_arr,
    coo_mat,
    csc_arr,
    csc_mat,
    csr_arr,
    csr_mat,
    dia_arr,
    dia_mat,
    dok_arr,
    dok_mat,
    lil_arr,
    lil_mat,
)

shape_1d: tuple[int]
shape_2d: tuple[int, int]
shape_3d: tuple[int, int, int]

dense_1d: np.ndarray[tuple[int], np.dtype[ScalarType]]
dense_2d: np.ndarray[tuple[int, int], np.dtype[ScalarType]]

sctype: type[ScalarType]

int_list: list[int]

###
# diags_array
assert_type(sparse.diags_array(dense_1d), sparse.dia_array[ScalarType])
assert_type(sparse.diags_array([dense_1d, dense_1d]), sparse.dia_array[ScalarType])
# diags (legacy, `diags_array` is preferred)
assert_type(sparse.diags(dense_1d), sparse.dia_matrix[ScalarType])
assert_type(sparse.diags([dense_1d, dense_1d]), sparse.dia_matrix[ScalarType])
# spdiags (legacy, `diags_array` is preferred)
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d), sparse.dia_matrix[ScalarType])

###
# eye_array
assert_type(sparse.eye_array(5), sparse.dia_array[np.float64])
assert_type(sparse.eye_array(5, 4, dtype=sctype), sparse.dia_array[ScalarType])
# eye (legacy, `eye_array` is preferred)
assert_type(sparse.eye(5), sparse.dia_matrix[np.float64])
assert_type(sparse.eye(5, 4), sparse.dia_matrix[np.float64])
assert_type(sparse.eye(5, 4, dtype=sctype), sparse.dia_matrix[ScalarType])
# identity (legacy, `eye_array` is preferred)
assert_type(sparse.identity(5), sparse.dia_matrix[np.float64])
assert_type(sparse.identity(5, dtype=sctype), sparse.dia_matrix[ScalarType])

###
# kron
assert_type(sparse.kron(any_mat, any_mat), sparse.bsr_matrix[ScalarType])
assert_type(sparse.kron(any_mat, any_arr), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, any_mat), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, any_arr), sparse.bsr_array[ScalarType])
# kronsum
assert_type(sparse.kronsum(any_mat, any_mat), sparse.csr_matrix[ScalarType])
assert_type(sparse.kronsum(any_mat, any_arr), sparse.csr_array[ScalarType])
assert_type(sparse.kronsum(any_arr, any_mat), sparse.csr_array[ScalarType])
assert_type(sparse.kronsum(any_arr, any_arr), sparse.csr_array[ScalarType])
assert_type(sparse.kronsum(any_mat, [[1, 2], [3, 4]]), sparse.csr_matrix[ScalarType])
assert_type(sparse.kronsum(any_arr, [[1, 2], [3, 4]]), sparse.csr_array[ScalarType])

###
# hstack (same as vstack)
assert_type(sparse.hstack([bsr_mat, bsr_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.hstack([coo_mat, coo_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.hstack([csc_mat, csc_mat]), sparse.csc_matrix[ScalarType])
assert_type(sparse.hstack([csr_mat, csr_mat]), sparse.csr_matrix[ScalarType])
assert_type(sparse.hstack([dia_mat, dia_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.hstack([dok_mat, dok_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.hstack([lil_mat, lil_mat]), sparse.coo_matrix[ScalarType])

assert_type(sparse.hstack([bsr_arr, bsr_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.hstack([coo_arr, coo_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.hstack([csc_arr, csc_arr]), sparse.csc_array[ScalarType])
assert_type(sparse.hstack([csr_arr, csr_arr]), sparse.csr_array[ScalarType])
assert_type(sparse.hstack([dia_arr, dia_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.hstack([dok_arr, dok_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.hstack([lil_arr, lil_arr]), sparse.coo_array[ScalarType, tuple[int, int]])

assert_type(sparse.hstack([bsr_arr, bsr_arr], dtype=np.complex64), sparse.coo_array[np.complex64, tuple[int, int]])
assert_type(sparse.hstack([coo_arr, coo_arr], dtype=np.complex64), sparse.coo_array[np.complex64, tuple[int, int]])
assert_type(sparse.hstack([csc_arr, csc_arr], dtype=np.complex64), sparse.csc_array[np.complex64])
assert_type(sparse.hstack([csr_arr, csr_arr], dtype=np.complex64), sparse.csr_array[np.complex64])
assert_type(sparse.hstack([dia_arr, dia_arr], dtype=np.complex64), sparse.coo_array[np.complex64, tuple[int, int]])
assert_type(sparse.hstack([dok_arr, dok_arr], dtype=np.complex64), sparse.coo_array[np.complex64, tuple[int, int]])
assert_type(sparse.hstack([lil_arr, lil_arr], dtype=np.complex64), sparse.coo_array[np.complex64, tuple[int, int]])

assert_type(sparse.hstack([bsr_arr, bsr_arr], dtype=sctype), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.hstack([coo_arr, coo_arr], dtype=float), sparse.coo_array[np.float64, tuple[int, int]])
assert_type(sparse.hstack([csc_arr, csc_arr], dtype=float), sparse.csc_array[np.float64])
assert_type(sparse.hstack([csr_arr, csr_arr], dtype=float), sparse.csr_array[np.float64])
assert_type(sparse.hstack([dia_arr, dia_arr], dtype=complex), sparse.coo_array[np.complex128, tuple[int, int]])
assert_type(sparse.hstack([dok_arr, dok_arr], dtype=complex), sparse.coo_array[np.complex128, tuple[int, int]])
assert_type(sparse.hstack([lil_arr, lil_arr], dtype=complex), sparse.coo_array[np.complex128, tuple[int, int]])

###
# block_array
assert_type(sparse.block_array([[any_mat]]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.block_array([[any_mat]], dtype=sctype), sparse.coo_array[ScalarType, tuple[int, int]])
# bmat (legacy, `block_array` is preferred)
assert_type(sparse.bmat([[any_mat]]), sparse.coo_matrix[ScalarType])
assert_type(sparse.bmat([[any_arr]]), sparse.coo_array[ScalarType, tuple[int, int]])

# block_diag
assert_type(sparse.block_diag([any_mat, any_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.block_diag([any_arr, any_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.block_diag([any_arr, any_mat]), sparse.coo_array[ScalarType, tuple[int, int]] | sparse.coo_matrix[ScalarType])

###
# random_array
assert_type(sparse.random_array(shape_1d), sparse.coo_array[np.float64, tuple[int]])
assert_type(sparse.random_array(shape_2d), sparse.coo_array[np.float64, tuple[int, int]])
assert_type(sparse.random_array(shape_3d), sparse.coo_array[np.float64, tuple[int, int, int]])
assert_type(sparse.random_array(shape_1d, dtype=sctype), sparse.coo_array[ScalarType, tuple[int]])
assert_type(sparse.random_array(shape_2d, dtype=sctype), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.random_array(shape_3d, dtype=sctype), sparse.coo_array[ScalarType, tuple[int, int, int]])
# random (legacy, `random_array` is preferred)
assert_type(sparse.random(4, 2), sparse.coo_matrix[np.float64])
assert_type(sparse.random(4, 2, dtype=sctype), sparse.coo_matrix[ScalarType])
# rand (legacy, `random_array` is preferred)
assert_type(sparse.rand(4, 2), sparse.coo_matrix[np.float64])
assert_type(sparse.rand(4, 2, dtype=sctype), sparse.coo_matrix[ScalarType])
