# ruff: noqa: ERA001
from typing import assert_type

import numpy as np

from ._types import ScalarType, csr_arr, csr_mat
from scipy.sparse import csc_array, csc_matrix

scalartype: ScalarType

shape2: tuple[int, int]

ind1: np.ndarray[tuple[int], np.dtype[np.intp]]
data1: np.ndarray[tuple[int], np.dtype[ScalarType]]
data2: np.ndarray[tuple[int, int], np.dtype[ScalarType]]

csc_spec2: tuple[
    np.ndarray[tuple[int], np.dtype[ScalarType]],
    tuple[np.ndarray[tuple[int], np.dtype[np.intp]], np.ndarray[tuple[int], np.dtype[np.intp]]],
]
csc_spec3: tuple[
    np.ndarray[tuple[int], np.dtype[ScalarType]],
    np.ndarray[tuple[int], np.dtype[np.intp]],
    np.ndarray[tuple[int], np.dtype[np.intp]],
]

###
# CSC matrix constructor

# csc_matrix(D)
assert_type(csc_matrix(data2), csc_matrix[ScalarType])
assert_type(csc_matrix(data2, dtype=scalartype), csc_matrix[ScalarType])
assert_type(csc_matrix(data2, dtype=bool), csc_matrix[np.bool_])
assert_type(csc_matrix(data2, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(data2, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(data2, dtype=complex), csc_matrix[np.complex128])

# csc_matrix(S)
assert_type(csc_matrix(csr_arr), csc_matrix[ScalarType])
assert_type(csc_matrix(csr_mat), csc_matrix[ScalarType])

# csc_matrix((M, N), [dtype])
assert_type(csc_matrix(shape2), csc_matrix[np.float64])
assert_type(csc_matrix(shape2, dtype=scalartype), csc_matrix[ScalarType])
assert_type(csc_matrix(shape2, dtype=bool), csc_matrix[np.bool_])
assert_type(csc_matrix(shape2, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(shape2, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(shape2, dtype=complex), csc_matrix[np.complex128])

# csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
assert_type(csc_matrix(csc_spec2), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec2, shape2), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec2, shape=shape2), csc_matrix[ScalarType])

assert_type(csc_matrix(csc_spec2, dtype=scalartype), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec2, dtype=bool), csc_matrix[np.bool_])
assert_type(csc_matrix(csc_spec2, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(csc_spec2, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(csc_spec2, dtype=complex), csc_matrix[np.complex128])

# csc_matrix((data, indices, indptr), [shape=(M, N)])
# NOTE: mypy incorrectly infers `csc_array[Any]` here, but it is correct in pyright.
assert_type(csc_matrix(csc_spec3), csc_matrix[ScalarType])  # type: ignore[assert-type]
assert_type(csc_matrix(csc_spec3, shape2), csc_matrix[ScalarType])  # type: ignore[assert-type]
assert_type(csc_matrix(csc_spec3, shape=shape2), csc_matrix[ScalarType])  # type: ignore[assert-type]

assert_type(csc_matrix(csc_spec3, dtype=scalartype), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec3, dtype=bool), csc_matrix[np.bool_])
assert_type(csc_matrix(csc_spec3, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(csc_spec3, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(csc_spec3, dtype=complex), csc_matrix[np.complex128])

###
# CSC array constructor

# csc_array(D)
assert_type(csc_array(data2), csc_array[ScalarType])
assert_type(csc_array(data2, dtype=scalartype), csc_array[ScalarType])
assert_type(csc_array(data2, dtype=bool), csc_array[np.bool_])
assert_type(csc_array(data2, dtype=int), csc_array[np.int_])
assert_type(csc_array(data2, dtype=float), csc_array[np.float64])
assert_type(csc_array(data2, dtype=complex), csc_array[np.complex128])

# csc_matrix(S)
assert_type(csc_array(csr_arr), csc_array[ScalarType])
assert_type(csc_array(csr_mat), csc_array[ScalarType])

# csc_array((M, N), [dtype])
assert_type(csc_array(shape2), csc_array[np.float64])
assert_type(csc_array(shape2, dtype=scalartype), csc_array[ScalarType])
assert_type(csc_array(shape2, dtype=bool), csc_array[np.bool_])
assert_type(csc_array(shape2, dtype=int), csc_array[np.int_])
assert_type(csc_array(shape2, dtype=float), csc_array[np.float64])
assert_type(csc_array(shape2, dtype=complex), csc_array[np.complex128])

# csc_array((data, (row_ind, col_ind)), [shape=(M, N)])
assert_type(csc_array(csc_spec2), csc_array[ScalarType])
assert_type(csc_array(csc_spec2, shape2), csc_array[ScalarType])
assert_type(csc_array(csc_spec2, shape=shape2), csc_array[ScalarType])

assert_type(csc_array(csc_spec2, dtype=scalartype), csc_array[ScalarType])
assert_type(csc_array(csc_spec2, dtype=bool), csc_array[np.bool_])
assert_type(csc_array(csc_spec2, dtype=int), csc_array[np.int_])
assert_type(csc_array(csc_spec2, dtype=float), csc_array[np.float64])
assert_type(csc_array(csc_spec2, dtype=complex), csc_array[np.complex128])

# csc_array((data, indices, indptr), [shape=(M, N)])
# NOTE: mypy incorrectly infers `csc_array[Any]` here, but it is correct in pyright.
assert_type(csc_array(csc_spec3), csc_array[ScalarType])  # type: ignore[assert-type]
assert_type(csc_array(csc_spec3, shape2), csc_array[ScalarType])  # type: ignore[assert-type]
assert_type(csc_array(csc_spec3, shape=shape2), csc_array[ScalarType])  # type: ignore[assert-type]

assert_type(csc_array(csc_spec3, dtype=scalartype), csc_array[ScalarType])
assert_type(csc_array(csc_spec3, dtype=bool), csc_array[np.bool_])
assert_type(csc_array(csc_spec3, dtype=int), csc_array[np.int_])
assert_type(csc_array(csc_spec3, dtype=float), csc_array[np.float64])
assert_type(csc_array(csc_spec3, dtype=complex), csc_array[np.complex128])
