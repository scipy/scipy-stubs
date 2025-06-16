from typing import assert_type

import numpy as np

from ._types import ScalarType, bsr_arr, bsr_mat, coo_arr, coo_mat, csc_arr, csc_mat, csr_arr, csr_mat, dia_arr, dia_mat
from scipy import sparse

###

bsr_arr_ui16: sparse.bsr_array[np.uint16 | np.int16]
bsr_arr_ui32_64: sparse.bsr_array[np.uint32 | np.int32 | np.uint64 | np.int64]
bsr_mat_ui16: sparse.bsr_matrix[np.uint16 | np.int16]
bsr_mat_ui32_64: sparse.bsr_matrix[np.uint32 | np.int32 | np.uint64 | np.int64]

coo_arr_ui16: sparse.coo_array[np.uint16 | np.int16, tuple[int, int]]
coo_arr_ui32_64: sparse.coo_array[np.uint32 | np.int32 | np.uint64 | np.int64, tuple[int, int]]
coo_mat_ui16: sparse.coo_matrix[np.uint16 | np.int16]
coo_mat_ui32_64: sparse.coo_matrix[np.uint32 | np.int32 | np.uint64 | np.int64]

csc_arr_ui16: sparse.csc_array[np.uint16 | np.int16]
csc_arr_ui32_64: sparse.csc_array[np.uint32 | np.int32 | np.uint64 | np.int64]
csc_mat_ui16: sparse.csc_matrix[np.uint16 | np.int16]
csc_mat_ui32_64: sparse.csc_matrix[np.uint32 | np.int32 | np.uint64 | np.int64]

csr_arr_ui16: sparse.csr_array[np.uint16 | np.int16]
csr_arr_ui32_64: sparse.csr_array[np.uint32 | np.int32 | np.uint64 | np.int64]
csr_mat_ui16: sparse.csr_matrix[np.uint16 | np.int16]
csr_mat_ui32_64: sparse.csr_matrix[np.uint32 | np.int32 | np.uint64 | np.int64]

dia_arr_ui16: sparse.dia_array[np.uint16 | np.int16]
dia_arr_ui32_64: sparse.dia_array[np.uint32 | np.int32 | np.uint64 | np.int64]
dia_mat_ui16: sparse.dia_matrix[np.uint16 | np.int16]
dia_mat_ui32_64: sparse.dia_matrix[np.uint32 | np.int32 | np.uint64 | np.int64]

###

# NOTE: the sign method shares the same signature with ceil, floor, rint, and trunc

# NOTE: the sqrt method shares the same signature with expm1, log1p, sin, arcsin,
#   sinh, arcsinh, tan, arctan, tanh, arctanh, deg2rad, and rad2deg

###
# BSR

assert_type(bsr_arr.sign(), sparse.bsr_array[ScalarType])
assert_type(bsr_arr.sqrt(), sparse.bsr_array[ScalarType])
assert_type(bsr_arr_ui16.sqrt(), sparse.bsr_array[np.float32])
assert_type(bsr_arr_ui32_64.sqrt(), sparse.bsr_array[np.float64])

assert_type(bsr_mat.sign(), sparse.bsr_matrix[ScalarType])
assert_type(bsr_mat.sqrt(), sparse.bsr_matrix[ScalarType])
assert_type(bsr_mat_ui16.sqrt(), sparse.bsr_matrix[np.float32])
assert_type(bsr_mat_ui32_64.sqrt(), sparse.bsr_matrix[np.float64])

###
# COO

assert_type(coo_arr.sign(), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(coo_arr.sqrt(), sparse.coo_array[ScalarType, tuple[int, int]])
assert_type(coo_arr_ui16.sqrt(), sparse.coo_array[np.float32, tuple[int, int]])
assert_type(coo_arr_ui32_64.sqrt(), sparse.coo_array[np.float64, tuple[int, int]])

assert_type(coo_mat.sign(), sparse.coo_matrix[ScalarType])
assert_type(coo_mat.sqrt(), sparse.coo_matrix[ScalarType])
assert_type(coo_mat_ui16.sqrt(), sparse.coo_matrix[np.float32])
assert_type(coo_mat_ui32_64.sqrt(), sparse.coo_matrix[np.float64])

###
# CSC

assert_type(csc_arr.sign(), sparse.csc_array[ScalarType])
assert_type(csc_arr.sqrt(), sparse.csc_array[ScalarType])
assert_type(csc_arr_ui16.sqrt(), sparse.csc_array[np.float32])
assert_type(csc_arr_ui32_64.sqrt(), sparse.csc_array[np.float64])

assert_type(csc_mat.sign(), sparse.csc_matrix[ScalarType])
assert_type(csc_mat.sqrt(), sparse.csc_matrix[ScalarType])
assert_type(csc_mat_ui16.sqrt(), sparse.csc_matrix[np.float32])
assert_type(csc_mat_ui32_64.sqrt(), sparse.csc_matrix[np.float64])

###
# CSR

assert_type(csr_arr.sign(), sparse.csr_array[ScalarType])
assert_type(csr_arr.sqrt(), sparse.csr_array[ScalarType])
assert_type(csr_arr_ui16.sqrt(), sparse.csr_array[np.float32])
assert_type(csr_arr_ui32_64.sqrt(), sparse.csr_array[np.float64])

assert_type(csr_mat.sign(), sparse.csr_matrix[ScalarType])
assert_type(csr_mat.sqrt(), sparse.csr_matrix[ScalarType])
assert_type(csr_mat_ui16.sqrt(), sparse.csr_matrix[np.float32])
assert_type(csr_mat_ui32_64.sqrt(), sparse.csr_matrix[np.float64])

###
# DIA

assert_type(dia_arr.sign(), sparse.dia_array[ScalarType])
assert_type(dia_arr.sqrt(), sparse.dia_array[ScalarType])
assert_type(dia_arr_ui16.sqrt(), sparse.dia_array[np.float32])
assert_type(dia_arr_ui32_64.sqrt(), sparse.dia_array[np.float64])

assert_type(dia_mat.sign(), sparse.dia_matrix[ScalarType])
assert_type(dia_mat.sqrt(), sparse.dia_matrix[ScalarType])
assert_type(dia_mat_ui16.sqrt(), sparse.dia_matrix[np.float32])
assert_type(dia_mat_ui32_64.sqrt(), sparse.dia_matrix[np.float64])
