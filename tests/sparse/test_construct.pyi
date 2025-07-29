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

# TODO(julvandenbroeck): add tests for arrays with unknown shape, like np.ndarray[tuple[int, ...], np.dtype[ScalarType]]
dense_1d: np.ndarray[tuple[int], np.dtype[ScalarType]]
dense_2d: np.ndarray[tuple[int, int], np.dtype[ScalarType]]

sctype: type[ScalarType]

int_list: list[int]

###
# diags_array
assert_type(sparse.diags_array([1, 2]), sparse.dia_array[np.float64] | sparse.dia_array[np.complex128])
assert_type(sparse.diags_array([[1, 2], 2.0]), sparse.dia_array[np.float64] | sparse.dia_array[np.complex128])
assert_type(sparse.diags_array([[1, 2.0], [2]]), sparse.dia_array[np.float64] | sparse.dia_array[np.complex128])
assert_type(sparse.diags_array([3j, 5j]), sparse.dia_array[np.float64] | sparse.dia_array[np.complex128])
assert_type(sparse.diags_array([[1, 2.0], [3j]]), sparse.dia_array[np.float64] | sparse.dia_array[np.complex128])
assert_type(sparse.diags_array([[1, 2.0], 3j]), sparse.dia_array[np.float64] | sparse.dia_array[np.complex128])
assert_type(sparse.diags_array(dense_1d), sparse.dia_array[ScalarType])
assert_type(sparse.diags_array(dense_1d.astype(np.float128)), sparse.dia_array[np.float128])
assert_type(sparse.diags_array(dense_1d.astype(np.complex128)), sparse.dia_array[np.complex128])
assert_type(sparse.diags_array(dense_2d, offsets=int_list), sparse.dia_array[ScalarType])
assert_type(sparse.diags_array(dense_1d, dtype="bool"), sparse.dia_array[np.bool_])
assert_type(sparse.diags_array(dense_2d, dtype="bool"), sparse.dia_array[np.bool_])
assert_type(sparse.diags_array(dense_1d, dtype=int), sparse.dia_array[np.int_])
assert_type(sparse.diags_array(dense_2d, dtype=int), sparse.dia_array[np.int_])
assert_type(sparse.diags_array(dense_1d, dtype=np.float128), sparse.dia_array[np.float128])
assert_type(sparse.diags_array(dense_2d, dtype=np.float128), sparse.dia_array[np.float128])
assert_type(sparse.diags_array(dense_1d, dtype="<c16"), sparse.dia_array[np.complex128])
assert_type(sparse.diags_array(dense_2d, dtype="<c16"), sparse.dia_array[np.complex128])
assert_type(sparse.diags_array(dense_2d, format="bsr"), sparse.bsr_array[ScalarType])
assert_type(sparse.diags_array(dense_2d, format="coo"), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.diags_array(dense_2d, format="csc"), sparse.csc_array[ScalarType])
assert_type(sparse.diags_array(dense_2d, format="csr"), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(sparse.diags_array(dense_2d, format="dia"), sparse.dia_array[ScalarType])
assert_type(sparse.diags_array(dense_2d, format="dok"), sparse.dok_array[ScalarType, tuple[int, int]])
assert_type(sparse.diags_array(dense_2d, format="lil"), sparse.lil_array[ScalarType])
# diags (legacy, `diags_array` is preferred)
assert_type(sparse.diags(dense_1d), sparse.dia_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, int_list, (5, 5), format="lil"), sparse.lil_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, int_list, (5, 5), format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, int_list, (5, 5), format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, int_list, (5, 5), format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, int_list, (5, 5), format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, int_list, (5, 5), format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, int_list, (5, 5), format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.diags(dense_1d, int_list, (5, 5), format="lil"), sparse.lil_matrix[ScalarType])
assert_type(sparse.diags(dense_2d), sparse.dia_matrix[ScalarType])
assert_type(sparse.diags(dense_2d, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.diags(dense_2d, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.diags(dense_2d, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.diags(dense_2d, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.diags(dense_2d, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.diags(dense_2d, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.diags(dense_2d, format="lil"), sparse.lil_matrix[ScalarType])
# spdiags (legacy, `diags_array` is preferred)
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, 4, 4, format="lil"), sparse.lil_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.spdiags(dense_1d, int_list, shape_2d, format="lil"), sparse.lil_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, 4, 4, format="lil"), sparse.lil_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.spdiags(dense_2d, int_list, shape_2d, format="lil"), sparse.lil_matrix[ScalarType])

###
# eye_array
assert_type(sparse.eye_array(5), sparse.dia_array[np.float64])
assert_type(sparse.eye_array(5, format="bsr"), sparse.bsr_array[np.float64])
assert_type(sparse.eye_array(5, format="coo"), sparse.coo_array[np.float64, tuple[int, int]])
assert_type(sparse.eye_array(5, format="csc"), sparse.csc_array[np.float64])
assert_type(sparse.eye_array(5, format="csr"), sparse.csr_array[np.float64, tuple[int, int]])
assert_type(sparse.eye_array(5, format="dia"), sparse.dia_array[np.float64])
assert_type(sparse.eye_array(5, format="dok"), sparse.dok_array[np.float64, tuple[int, int]])
assert_type(sparse.eye_array(5, format="lil"), sparse.lil_array[np.float64])
assert_type(sparse.eye_array(5, 4, dtype=sctype), sparse.dia_array[ScalarType])
assert_type(sparse.eye_array(5, 4, dtype=sctype, format="bsr"), sparse.bsr_array[ScalarType])
assert_type(sparse.eye_array(5, 4, dtype=sctype, format="coo"), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.eye_array(5, 4, dtype=sctype, format="csc"), sparse.csc_array[ScalarType])
assert_type(sparse.eye_array(5, 4, dtype=sctype, format="csr"), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(sparse.eye_array(5, 4, dtype=sctype, format="dia"), sparse.dia_array[ScalarType])
assert_type(sparse.eye_array(5, 4, dtype=sctype, format="dok"), sparse.dok_array[ScalarType, tuple[int, int]])
assert_type(sparse.eye_array(5, 4, dtype=sctype, format="lil"), sparse.lil_array[ScalarType])
# eye (legacy, `eye_array` is preferred)
assert_type(sparse.eye(5), sparse.dia_matrix[np.float64])
assert_type(sparse.eye(5, format="bsr"), sparse.bsr_matrix[np.float64])
assert_type(sparse.eye(5, format="coo"), sparse.coo_matrix[np.float64])
assert_type(sparse.eye(5, format="csc"), sparse.csc_matrix[np.float64])
assert_type(sparse.eye(5, format="csr"), sparse.csr_matrix[np.float64])
assert_type(sparse.eye(5, format="dia"), sparse.dia_matrix[np.float64])
assert_type(sparse.eye(5, format="dok"), sparse.dok_matrix[np.float64])
assert_type(sparse.eye(5, format="lil"), sparse.lil_matrix[np.float64])
assert_type(sparse.eye(5, 4), sparse.dia_matrix[np.float64])
assert_type(sparse.eye(5, 4, format="bsr"), sparse.bsr_matrix[np.float64])
assert_type(sparse.eye(5, 4, format="coo"), sparse.coo_matrix[np.float64])
assert_type(sparse.eye(5, 4, format="csc"), sparse.csc_matrix[np.float64])
assert_type(sparse.eye(5, 4, format="csr"), sparse.csr_matrix[np.float64])
assert_type(sparse.eye(5, 4, format="dia"), sparse.dia_matrix[np.float64])
assert_type(sparse.eye(5, 4, format="dok"), sparse.dok_matrix[np.float64])
assert_type(sparse.eye(5, 4, format="lil"), sparse.lil_matrix[np.float64])
assert_type(sparse.eye(5, 4, dtype=sctype), sparse.dia_matrix[ScalarType])
assert_type(sparse.eye(5, 4, dtype=sctype, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.eye(5, 4, dtype=sctype, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.eye(5, 4, dtype=sctype, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.eye(5, 4, dtype=sctype, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.eye(5, 4, dtype=sctype, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.eye(5, 4, dtype=sctype, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.eye(5, 4, dtype=sctype, format="lil"), sparse.lil_matrix[ScalarType])
# identity (legacy, `eye_array` is preferred)
assert_type(sparse.identity(5), sparse.dia_matrix[np.float64])
assert_type(sparse.identity(5, format="bsr"), sparse.bsr_matrix[np.float64])
assert_type(sparse.identity(5, format="coo"), sparse.coo_matrix[np.float64])
assert_type(sparse.identity(5, format="csc"), sparse.csc_matrix[np.float64])
assert_type(sparse.identity(5, format="csr"), sparse.csr_matrix[np.float64])
assert_type(sparse.identity(5, format="dia"), sparse.dia_matrix[np.float64])
assert_type(sparse.identity(5, format="dok"), sparse.dok_matrix[np.float64])
assert_type(sparse.identity(5, format="lil"), sparse.lil_matrix[np.float64])
assert_type(sparse.identity(5, dtype=sctype), sparse.dia_matrix[ScalarType])
assert_type(sparse.identity(5, dtype=sctype, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.identity(5, dtype=sctype, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.identity(5, dtype=sctype, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.identity(5, dtype=sctype, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.identity(5, dtype=sctype, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.identity(5, dtype=sctype, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.identity(5, dtype=sctype, format="lil"), sparse.lil_matrix[ScalarType])

###
# kron
assert_type(sparse.kron(any_mat, any_mat), sparse.bsr_matrix[ScalarType])
assert_type(sparse.kron(any_mat, any_arr), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, any_mat), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, any_arr), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(dense_2d, any_arr), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, dense_2d), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, any_arr, format="bsr"), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, any_arr, format="coo"), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.kron(any_arr, any_arr, format="csc"), sparse.csc_array[ScalarType])
assert_type(sparse.kron(any_arr, any_arr, format="csr"), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(sparse.kron(any_arr, any_arr, format="dia"), sparse.dia_array[ScalarType])
assert_type(sparse.kron(any_arr, any_arr, format="dok"), sparse.dok_array[ScalarType, tuple[int, int]])
assert_type(sparse.kron(any_arr, any_arr, format="lil"), sparse.lil_array[ScalarType])
assert_type(sparse.kron(any_arr, dense_2d, format="lil"), sparse.lil_array[ScalarType])
assert_type(sparse.kron(dense_2d, any_arr, format="lil"), sparse.lil_array[ScalarType])
# kronsum
assert_type(sparse.kronsum(any_mat, any_mat), sparse.csr_matrix[ScalarType])
assert_type(sparse.kronsum(any_mat, any_arr), sparse.csr_array[ScalarType])
assert_type(sparse.kronsum(any_arr, any_mat), sparse.csr_array[ScalarType])
assert_type(sparse.kronsum(any_arr, any_arr), sparse.csr_array[ScalarType])
assert_type(sparse.kronsum(dense_2d, any_arr), sparse.csr_array[ScalarType])
assert_type(sparse.kronsum(any_arr, dense_2d), sparse.csr_array[ScalarType])
assert_type(sparse.kronsum(any_arr, any_arr, format="bsr"), sparse.bsr_array[ScalarType])
assert_type(sparse.kronsum(any_arr, any_arr, format="coo"), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.kronsum(any_arr, any_arr, format="csc"), sparse.csc_array[ScalarType])
assert_type(sparse.kronsum(any_arr, any_arr, format="csr"), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(sparse.kronsum(any_arr, any_arr, format="dia"), sparse.dia_array[ScalarType])
assert_type(sparse.kronsum(any_arr, any_arr, format="dok"), sparse.dok_array[ScalarType, tuple[int, int]])
assert_type(sparse.kronsum(any_arr, any_arr, format="lil"), sparse.lil_array[ScalarType])
assert_type(sparse.kronsum(any_arr, dense_2d, format="lil"), sparse.lil_array[ScalarType])
assert_type(sparse.kronsum(dense_2d, any_arr, format="lil"), sparse.lil_array[ScalarType])

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

assert_type(sparse.hstack([coo_arr, coo_arr], format="bsr"), sparse.bsr_array[ScalarType])
assert_type(sparse.hstack([csc_arr, csc_arr], format="bsr", dtype=np.complex64), sparse.bsr_array[np.complex64])
assert_type(sparse.hstack([csr_arr, csr_arr], format="bsr", dtype=int), sparse.bsr_array[np.int_])
assert_type(sparse.hstack([dia_arr, dia_arr], format="bsr", dtype=float), sparse.bsr_array[np.float64])
assert_type(sparse.hstack([dok_arr, dok_arr], format="bsr", dtype=complex), sparse.bsr_array[np.complex128])
assert_type(sparse.hstack([csc_arr, csc_arr], format="coo", dtype=np.complex64), sparse.coo_array[np.complex64, tuple[int, int]])
assert_type(sparse.hstack([csc_arr, csc_arr], format="csc", dtype=np.complex64), sparse.csc_array[np.complex64])
assert_type(sparse.hstack([csc_arr, csc_arr], format="csr", dtype=np.complex64), sparse.csr_array[np.complex64, tuple[int, int]])
assert_type(sparse.hstack([csc_arr, csc_arr], format="dia", dtype=np.complex64), sparse.dia_array[np.complex64])
assert_type(sparse.hstack([csc_arr, csc_arr], format="dok", dtype=np.complex64), sparse.dok_array[np.complex64, tuple[int, int]])
assert_type(sparse.hstack([csc_arr, csc_arr], format="lil", dtype=np.complex64), sparse.lil_array[np.complex64])

###
# block_array
assert_type(sparse.block_array([[bsr_mat]]), sparse.coo_matrix[ScalarType])
assert_type(sparse.block_array([[bsr_arr]]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.block_array([[csr_arr, None], [None, dok_arr]], format="dia"), sparse.dia_array[ScalarType])
assert_type(sparse.block_array([[coo_arr]], dtype=int), sparse.coo_array[np.int_, tuple[int, int]])
assert_type(sparse.block_array([[csr_arr]], dtype=sctype), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(sparse.block_array([[lil_arr]], dtype=np.complex64), sparse.coo_array[np.complex64, tuple[int, int]])
assert_type(sparse.block_array([[csr_arr]], format="lil"), sparse.lil_array[ScalarType])
assert_type(sparse.block_array([[coo_arr, None]], format="bsr", dtype="float"), sparse.bsr_array[np.float64])
assert_type(sparse.block_array([[dia_arr], [None]], format="dok", dtype=complex), sparse.dok_array[np.complex128])
# bmat (legacy, `block_array` is p`referred)
assert_type(sparse.bmat([[bsr_mat]]), sparse.coo_matrix[ScalarType])
assert_type(sparse.bmat([[csr_mat], [None]]), sparse.csr_matrix[ScalarType])
assert_type(sparse.bmat([[dia_mat]], dtype=np.int_), sparse.coo_matrix[np.int_])
assert_type(sparse.bmat([[dok_mat], [None]], dtype=np.complex64), sparse.coo_matrix[np.complex64])
assert_type(sparse.bmat([[bsr_arr]]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.bmat([[csr_arr], [None]]), sparse.csr_array[ScalarType, tuple[int, int]])
assert_type(sparse.bmat([[dia_arr]], dtype=np.int_), sparse.coo_array[np.int_, tuple[int, int]])
assert_type(sparse.bmat([[dok_arr], [None]], dtype=np.complex64), sparse.coo_array[np.complex64, tuple[int, int]])

# block_diag
assert_type(sparse.block_diag([any_arr, any_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.block_diag([any_arr, any_mat]), sparse.coo_matrix[ScalarType] | sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.block_diag([any_arr, any_arr], dtype="bool"), sparse.coo_array[np.bool_, tuple[int, int]])
assert_type(sparse.block_diag([any_arr, any_arr], dtype=int), sparse.coo_array[np.int_, tuple[int, int]])
assert_type(
    sparse.block_diag([any_arr, any_mat], dtype=np.complex64),
    sparse.coo_matrix[np.complex64] | sparse.coo_array[np.complex64, tuple[int, int]],
)
assert_type(sparse.block_diag([any_arr, any_arr], format="bsr"), sparse.bsr_array[ScalarType])
assert_type(sparse.block_diag([any_arr, any_arr], format="csc", dtype=float), sparse.csc_array[np.float64])
assert_type(sparse.block_diag([any_arr, any_arr], format="csr", dtype=complex), sparse.csr_array[np.complex128, tuple[int, int]])
assert_type(sparse.block_diag([any_arr, any_arr], format="dia", dtype=np.int32), sparse.dia_array[np.int32])
assert_type(sparse.block_diag([any_arr, any_arr], format="dok", dtype="bool"), sparse.dok_array[np.bool_, tuple[int, int]])
assert_type(sparse.block_diag([any_arr, any_arr], format="lil", dtype=np.complex64), sparse.lil_array[np.complex64])

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
