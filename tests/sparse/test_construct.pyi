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

shape_2d: tuple[int, int]
dense_1d: np.ndarray[tuple[int], np.dtype[ScalarType]]
int_list: list[int]

# eye
assert_type(sparse.eye(5), sparse.dia_matrix[np.float64])

# random
assert_type(sparse.random(4, 2), sparse.coo_matrix[np.float64])

# diags
assert_type(sparse.diags([dense_1d, dense_1d, dense_1d], int_list, shape=shape_2d), sparse.dia_matrix[np.float64])

# stack (same as hstack)
assert_type(sparse.vstack([bsr_mat, bsr_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.vstack([coo_mat, coo_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.vstack([csc_mat, csc_mat]), sparse.csc_matrix[ScalarType])
assert_type(sparse.vstack([csr_mat, csr_mat]), sparse.csr_matrix[ScalarType])
assert_type(sparse.vstack([dia_mat, dia_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.vstack([dok_mat, dok_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.vstack([lil_mat, lil_mat]), sparse.coo_matrix[ScalarType])

assert_type(sparse.vstack([bsr_arr, bsr_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.vstack([coo_arr, coo_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.vstack([csc_arr, csc_arr]), sparse.csc_array[ScalarType])
assert_type(sparse.vstack([csr_arr, csr_arr]), sparse.csr_array[ScalarType])
assert_type(sparse.vstack([dia_arr, dia_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.vstack([dok_arr, dok_arr]), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.vstack([lil_arr, lil_arr]), sparse.coo_array[ScalarType, tuple[int, int]])

# block_diag
assert_type(sparse.block_diag([any_mat, any_mat]), sparse.coo_matrix[ScalarType])
assert_type(sparse.block_diag([any_arr, any_arr]), sparse.coo_array[ScalarType, tuple[int, int]])

# kron (Kronecker product)
assert_type(sparse.kron(any_mat, any_mat), sparse.bsr_matrix[ScalarType])
assert_type(sparse.kron(any_mat, any_arr), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, any_mat), sparse.bsr_array[ScalarType])
assert_type(sparse.kron(any_arr, any_arr), sparse.bsr_array[ScalarType])
