from typing import assert_type

import numpy as np

import scipy.sparse as sparse
from ._types import ScalarType, any_arr, any_mat

# find
assert_type(
    sparse.find(any_mat),
    tuple[
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[ScalarType]],
    ],
)
assert_type(
    sparse.find(any_arr),
    tuple[
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[ScalarType]],
    ],
)

# tril
assert_type(sparse.tril(any_mat), sparse.coo_matrix[ScalarType])
assert_type(sparse.tril(any_mat, 1), sparse.coo_matrix[ScalarType])
assert_type(sparse.tril(any_mat, k=1), sparse.coo_matrix[ScalarType])
assert_type(sparse.tril(any_mat, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.tril(any_mat, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.tril(any_mat, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.tril(any_mat, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.tril(any_mat, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.tril(any_mat, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.tril(any_mat, format="lil"), sparse.lil_matrix[ScalarType])
assert_type(sparse.tril(any_arr), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.tril(any_arr, 1), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.tril(any_arr, k=1), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.tril(any_arr, format="bsr"), sparse.bsr_array[ScalarType])
assert_type(sparse.tril(any_arr, format="coo"), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.tril(any_arr, format="csc"), sparse.csc_array[ScalarType])
assert_type(sparse.tril(any_arr, format="csr"), sparse.csr_array[ScalarType])
assert_type(sparse.tril(any_arr, format="dia"), sparse.dia_array[ScalarType])
assert_type(sparse.tril(any_arr, format="dok"), sparse.dok_array[ScalarType])
assert_type(sparse.tril(any_arr, format="lil"), sparse.lil_array[ScalarType])

# triu
assert_type(sparse.triu(any_mat), sparse.coo_matrix[ScalarType])
assert_type(sparse.triu(any_mat, 1), sparse.coo_matrix[ScalarType])
assert_type(sparse.triu(any_mat, k=1), sparse.coo_matrix[ScalarType])
assert_type(sparse.triu(any_mat, format="bsr"), sparse.bsr_matrix[ScalarType])
assert_type(sparse.triu(any_mat, format="coo"), sparse.coo_matrix[ScalarType])
assert_type(sparse.triu(any_mat, format="csc"), sparse.csc_matrix[ScalarType])
assert_type(sparse.triu(any_mat, format="csr"), sparse.csr_matrix[ScalarType])
assert_type(sparse.triu(any_mat, format="dia"), sparse.dia_matrix[ScalarType])
assert_type(sparse.triu(any_mat, format="dok"), sparse.dok_matrix[ScalarType])
assert_type(sparse.triu(any_mat, format="lil"), sparse.lil_matrix[ScalarType])
assert_type(sparse.triu(any_arr), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.triu(any_arr, 1), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.triu(any_arr, k=1), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.triu(any_arr, format="bsr"), sparse.bsr_array[ScalarType])
assert_type(sparse.triu(any_arr, format="coo"), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(sparse.triu(any_arr, format="csc"), sparse.csc_array[ScalarType])
assert_type(sparse.triu(any_arr, format="csr"), sparse.csr_array[ScalarType])
assert_type(sparse.triu(any_arr, format="dia"), sparse.dia_array[ScalarType])
assert_type(sparse.triu(any_arr, format="dok"), sparse.dok_array[ScalarType])
assert_type(sparse.triu(any_arr, format="lil"), sparse.lil_array[ScalarType])
