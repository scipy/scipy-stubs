from typing import assert_type

import numpy as np
import optype.numpy as onp

from ._types import ScalarType, csr_arr, csr_mat, dia_arr, dia_mat
from scipy.sparse import bsr_array, bsr_matrix, csc_array, csc_matrix, csr_array, csr_matrix, dia_array, dia_matrix

###

_shape2: tuple[int, int]
_shape_nd: tuple[int, ...]
_py_i: int
_py_i_1d: list[int]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]
_dense_2d: onp.Array2D[ScalarType]

_dia_arr_bool: dia_array[np.bool]
_dia_arr_i64: dia_array[np.int64]
_dia_arr_f64: dia_array[np.float64]
_dia_arr_c128: dia_array[np.complex128]
_dia_mat_i64: dia_matrix[np.int64]

###
# (M, N) shape constructor

assert_type(dia_array(_shape2), dia_array[np.float64])
assert_type(dia_array(_shape2, dtype=np.bool), dia_array[np.bool])
assert_type(dia_array(_shape2, dtype=np.int64), dia_array[np.int64])
assert_type(dia_array(_shape2, dtype=np.float64), dia_array[np.float64])
assert_type(dia_array(_shape2, dtype=np.complex128), dia_array[np.complex128])
assert_type(dia_array(_shape2, dtype=np.int8), dia_array[np.int8])
assert_type(dia_array(_shape2, dtype=np.uint8), dia_array[np.uint8])
assert_type(dia_array(_shape2, dtype=np.float32), dia_array[np.float32])
assert_type(dia_array(_shape2, dtype=np.complex64), dia_array[np.complex64])
assert_type(dia_array(_shape2, shape=_shape_nd), dia_array[np.float64])

assert_type(dia_matrix(_shape2), dia_matrix[np.float64])
assert_type(dia_matrix(_shape2, dtype=np.bool), dia_matrix[np.bool])
assert_type(dia_matrix(_shape2, dtype=np.int64), dia_matrix[np.int64])
assert_type(dia_matrix(_shape2, dtype=np.float64), dia_matrix[np.float64])
assert_type(dia_matrix(_shape2, dtype=np.complex128), dia_matrix[np.complex128])
assert_type(dia_matrix(_shape2, dtype=np.int8), dia_matrix[np.int8])
assert_type(dia_matrix(_shape2, dtype=np.uint8), dia_matrix[np.uint8])
assert_type(dia_matrix(_shape2, dtype=np.float32), dia_matrix[np.float32])
assert_type(dia_matrix(_shape2, dtype=np.complex64), dia_matrix[np.complex64])

###
# (data, offsets) constructor

assert_type(dia_array((_f64_2d, _py_i_1d), shape=_shape2), dia_array[np.float64])
assert_type(dia_matrix((_f64_1d, _py_i), shape=_shape2), dia_matrix[np.float64])

###
# arithmetic with scalars

assert_type(dia_arr * 3, dia_array[ScalarType])
assert_type(dia_arr.multiply(3), dia_array[ScalarType])
assert_type(dia_arr.dot(3), dia_array[ScalarType])
assert_type(dia_mat * 3, dia_matrix[ScalarType])
assert_type(dia_mat.multiply(3), dia_matrix[ScalarType])
assert_type(dia_mat.dot(3), dia_matrix[ScalarType])

assert_type(_dia_arr_i64 * 0.5, dia_array[np.float64])
assert_type(_dia_mat_i64 * 0.5, dia_matrix[np.float64])
assert_type(_dia_arr_f64 * 1j, dia_array[np.complex128])

###
# arithmetic with dia

assert_type(dia_arr * dia_arr, dia_array[ScalarType])
assert_type(dia_arr.multiply(dia_arr), dia_array[ScalarType])
assert_type(dia_arr @ dia_arr, dia_array[ScalarType])
assert_type(dia_arr.dot(dia_arr), dia_array[ScalarType])
assert_type(dia_arr * dia_mat, dia_array[ScalarType])
assert_type(dia_arr @ dia_mat, dia_array[ScalarType])

assert_type(dia_mat * dia_mat, dia_matrix[ScalarType])
assert_type(dia_mat.multiply(dia_mat), dia_matrix[ScalarType])
assert_type(dia_mat @ dia_mat, dia_matrix[ScalarType])
assert_type(dia_mat.dot(dia_mat), dia_matrix[ScalarType])
assert_type(dia_mat * dia_arr, dia_matrix[ScalarType])
assert_type(dia_mat @ dia_arr, dia_matrix[ScalarType])

assert_type(_dia_arr_i64 * _dia_arr_bool, dia_array[np.int64])
assert_type(_dia_arr_f64 @ _dia_arr_i64, dia_array[np.float64])
assert_type(_dia_arr_f64 @ dia_arr, dia_array[np.float64])
assert_type(_dia_arr_c128 @ _dia_arr_f64, dia_array[np.complex128])

###
# arithmetic with other sparse formats and dense arrays

assert_type(dia_arr * csr_arr, bsr_array[ScalarType] | csc_array[ScalarType] | csr_array[ScalarType])
assert_type(dia_arr @ csr_arr, bsr_array[ScalarType] | csc_array[ScalarType] | csr_array[ScalarType])
assert_type(dia_mat * csr_mat, bsr_matrix[ScalarType] | csc_matrix[ScalarType] | csr_matrix[ScalarType])
assert_type(dia_mat @ csr_mat, bsr_matrix[ScalarType] | csc_matrix[ScalarType] | csr_matrix[ScalarType])
assert_type(csr_arr @ dia_arr, csr_array[ScalarType])

assert_type(dia_arr @ _dense_2d, onp.ArrayND[ScalarType])
assert_type(dia_mat @ _dense_2d, onp.ArrayND[ScalarType])
