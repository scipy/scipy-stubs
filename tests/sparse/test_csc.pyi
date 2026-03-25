# ruff: noqa: ERA001
from typing import assert_type

import numpy as np
import optype.numpy as onp

from ._types import ScalarType, csr_arr, csr_mat
from scipy.sparse import coo_array, csc_array, csc_matrix

dtype: np.dtype[ScalarType]

shape2: tuple[int, int]

ind1: onp.Array1D[np.intp]
data1: onp.Array1D[ScalarType]
data2: onp.Array2D[ScalarType]

csc_spec2: tuple[onp.Array1D[ScalarType], tuple[onp.Array1D[np.intp], onp.Array1D[np.intp]]]
csc_spec3: tuple[onp.Array1D[ScalarType], onp.Array1D[np.intp], onp.Array1D[np.intp]]

###
# CSC matrix constructor

# csc_matrix(D)
assert_type(csc_matrix(data2), csc_matrix[ScalarType])
assert_type(csc_matrix(data2, dtype=dtype), csc_matrix[ScalarType])
assert_type(csc_matrix(data2, dtype=bool), csc_matrix[np.bool_])
assert_type(csc_matrix(data2, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(data2, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(data2, dtype=complex), csc_matrix[np.complex128])

# csc_matrix(S)
assert_type(csc_matrix(csr_arr), csc_matrix[ScalarType])
assert_type(csc_matrix(csr_mat), csc_matrix[ScalarType])

# csc_matrix((M, N), [dtype])
assert_type(csc_matrix(shape2), csc_matrix[np.float64])
assert_type(csc_matrix(shape2, dtype=dtype), csc_matrix[ScalarType])
assert_type(csc_matrix(shape2, dtype=bool), csc_matrix[np.bool_])
assert_type(csc_matrix(shape2, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(shape2, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(shape2, dtype=complex), csc_matrix[np.complex128])

# csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
assert_type(csc_matrix(csc_spec2), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec2, shape2), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec2, shape=shape2), csc_matrix[ScalarType])

assert_type(csc_matrix(csc_spec2, dtype=dtype), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec2, dtype=bool), csc_matrix[np.bool_])
assert_type(csc_matrix(csc_spec2, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(csc_spec2, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(csc_spec2, dtype=complex), csc_matrix[np.complex128])

# csc_matrix((data, indices, indptr), [shape=(M, N)])
assert_type(csc_matrix(csc_spec3), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec3, shape2), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec3, shape=shape2), csc_matrix[ScalarType])

assert_type(csc_matrix(csc_spec3, dtype=dtype), csc_matrix[ScalarType])
assert_type(csc_matrix(csc_spec3, dtype=bool), csc_matrix[np.bool_])
assert_type(csc_matrix(csc_spec3, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(csc_spec3, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(csc_spec3, dtype=complex), csc_matrix[np.complex128])

###
# CSC array constructor

# csc_array(D)
assert_type(csc_array(data2), csc_array[ScalarType])
assert_type(csc_array(data2, dtype=dtype), csc_array[ScalarType])
assert_type(csc_array(data2, dtype=bool), csc_array[np.bool_])
assert_type(csc_array(data2, dtype=int), csc_array[np.int_])
assert_type(csc_array(data2, dtype=float), csc_array[np.float64])
assert_type(csc_array(data2, dtype=complex), csc_array[np.complex128])

# csc_array(S)
assert_type(csc_array(csr_arr), csc_array[ScalarType])
assert_type(csc_array(csr_mat), csc_array[ScalarType])

# csc_array((M, N), [dtype])
assert_type(csc_array(shape2), csc_array[np.float64])
assert_type(csc_array(shape2, dtype=dtype), csc_array[ScalarType])
assert_type(csc_array(shape2, dtype=bool), csc_array[np.bool_])
assert_type(csc_array(shape2, dtype=int), csc_array[np.int_])
assert_type(csc_array(shape2, dtype=float), csc_array[np.float64])
assert_type(csc_array(shape2, dtype=complex), csc_array[np.complex128])

# csc_array((data, (row_ind, col_ind)), [shape=(M, N)])
assert_type(csc_array(csc_spec2), csc_array[ScalarType])
assert_type(csc_array(csc_spec2, shape2), csc_array[ScalarType])
assert_type(csc_array(csc_spec2, shape=shape2), csc_array[ScalarType])

assert_type(csc_array(csc_spec2, dtype=dtype), csc_array[ScalarType])
assert_type(csc_array(csc_spec2, dtype=bool), csc_array[np.bool_])
assert_type(csc_array(csc_spec2, dtype=int), csc_array[np.int_])
assert_type(csc_array(csc_spec2, dtype=float), csc_array[np.float64])
assert_type(csc_array(csc_spec2, dtype=complex), csc_array[np.complex128])

# csc_array((data, indices, indptr), [shape=(M, N)])
assert_type(csc_array(csc_spec3), csc_array[ScalarType])
assert_type(csc_array(csc_spec3, shape2), csc_array[ScalarType])
assert_type(csc_array(csc_spec3, shape=shape2), csc_array[ScalarType])

assert_type(csc_array(csc_spec3, dtype=dtype), csc_array[ScalarType])
assert_type(csc_array(csc_spec3, dtype=bool), csc_array[np.bool_])
assert_type(csc_array(csc_spec3, dtype=int), csc_array[np.int_])
assert_type(csc_array(csc_spec3, dtype=float), csc_array[np.float64])
assert_type(csc_array(csc_spec3, dtype=complex), csc_array[np.complex128])

###
# regression tests for https://github.com/scipy/scipy-stubs/issues/1412

_csc_arr: csc_array[ScalarType]
_int_1d: onp.Array1D[np.int64]
_bool_1d: onp.Array1D[np.bool_]
_int_list: list[int]

assert_type(_csc_arr[0, 0], ScalarType)

assert_type(_csc_arr[_int_1d], csc_array[ScalarType])
assert_type(_csc_arr[:, _int_1d], csc_array[ScalarType])

assert_type(_csc_arr[_int_list], csc_array[ScalarType])
assert_type(_csc_arr[:, _int_list], csc_array[ScalarType])
assert_type(_csc_arr[_int_list, _int_list], onp.Array1D[ScalarType])

assert_type(_csc_arr[_bool_1d], csc_array[ScalarType])
assert_type(_csc_arr[:, _bool_1d], csc_array[ScalarType])
assert_type(_csc_arr[_bool_1d, _bool_1d], onp.Array1D[ScalarType])

assert_type(_csc_arr[_int_1d, 0], coo_array[ScalarType, tuple[int]])
assert_type(_csc_arr[0, _int_1d], coo_array[ScalarType, tuple[int]])

###
# regression tests for https://github.com/scipy/scipy-stubs/issues/1454

_arr_2d: onp.Array1D[ScalarType]

assert_type(_csc_arr[:, :], csc_array[ScalarType])
_csc_arr[0] = 1.0
_csc_arr[:] = 1.0
_csc_arr[:] = _arr_2d
_csc_arr[:, :] = 1.0
_csc_arr[:, :] = _arr_2d
_csc_arr[0, :] = 1.0
_csc_arr[0, :] = _arr_2d
_csc_arr[:, 0] = 1.0
_csc_arr[:, 0] = _arr_2d
_csc_arr[:, _int_list] = 1.0
_csc_arr[:, _int_list] = _arr_2d
_csc_arr[_int_list, :] = 1.0
_csc_arr[_int_list, :] = _arr_2d
_csc_arr[_int_list, _int_list] = 1.0
_csc_arr[_int_list, _int_list] = _arr_2d
