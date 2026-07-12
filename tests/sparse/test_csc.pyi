# ruff: noqa: ERA001
from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from ._types import ScalarType, csr_arr, csr_mat
from scipy.sparse import coo_array, csc_array, csc_matrix

_dtype: np.dtype[ScalarType]
_shape_2d: tuple[int, int]

_py_i_1d: list[int]

_bool_1d: onp.Array1D[np.bool]
_i64_1d: onp.Array1D[np.int64]

_sc_1d: onp.Array1D[ScalarType]
_sc_2d: onp.Array2D[ScalarType]

_csc_spec2: tuple[onp.Array1D[ScalarType], tuple[onp.Array1D[np.intp], onp.Array1D[np.intp]]]
_csc_spec3: tuple[onp.Array1D[ScalarType], onp.Array1D[np.intp], onp.Array1D[np.intp]]

_csc_sc: csc_array[ScalarType]
_csc_b: csc_array[np.bool]
_csc_i8: csc_array[np.int8]
_csc_i16: csc_array[np.int16]
_csc_i32: csc_array[np.int32]
_csc_i64: csc_array[np.int64]
_csc_f32: csc_array[np.float32]
_csc_f64: csc_array[np.float64]
_csc_c64: csc_array[np.complex64]
_csc_c128: csc_array[np.complex128]

_b: np.bool
_i8: np.int8
_i16: np.int16
_i32: np.int32
_i64: np.int64
_f32: np.float32
_f64: np.float64
_c64: np.complex64
_c128: np.complex128

_b_2d: onp.Array2D[np.bool]
_i8_2d: onp.Array2D[np.int8]
_i16_2d: onp.Array2D[np.int16]
_i32_2d: onp.Array2D[np.int32]
_i64_2d: onp.Array2D[np.int64]
_f32_2d: onp.Array2D[np.float32]
_f64_2d: onp.Array2D[np.float64]
_c64_2d: onp.Array2D[np.complex64]
_c128_2d: onp.Array2D[np.complex128]

###
# CSC matrix constructor

# csc_matrix(D)
assert_type(csc_matrix(_sc_2d), csc_matrix[ScalarType])
assert_type(csc_matrix(_sc_2d, dtype=_dtype), csc_matrix[ScalarType])
assert_type(csc_matrix(_sc_2d, dtype=bool), csc_matrix[np.bool])
assert_type(csc_matrix(_sc_2d, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(_sc_2d, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(_sc_2d, dtype=complex), csc_matrix[np.complex128])

# csc_matrix(S)
assert_type(csc_matrix(csr_arr), csc_matrix[ScalarType])
assert_type(csc_matrix(csr_mat), csc_matrix[ScalarType])

# csc_matrix((M, N), [dtype])
assert_type(csc_matrix(_shape_2d), csc_matrix[np.float64])
assert_type(csc_matrix(_shape_2d, dtype=_dtype), csc_matrix[ScalarType])
assert_type(csc_matrix(_shape_2d, dtype=bool), csc_matrix[np.bool])
assert_type(csc_matrix(_shape_2d, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(_shape_2d, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(_shape_2d, dtype=complex), csc_matrix[np.complex128])

# csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
assert_type(csc_matrix(_csc_spec2), csc_matrix[ScalarType])
assert_type(csc_matrix(_csc_spec2, _shape_2d), csc_matrix[ScalarType])
assert_type(csc_matrix(_csc_spec2, shape=_shape_2d), csc_matrix[ScalarType])

assert_type(csc_matrix(_csc_spec2, dtype=_dtype), csc_matrix[ScalarType])
assert_type(csc_matrix(_csc_spec2, dtype=bool), csc_matrix[np.bool])
assert_type(csc_matrix(_csc_spec2, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(_csc_spec2, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(_csc_spec2, dtype=complex), csc_matrix[np.complex128])

# csc_matrix((data, indices, indptr), [shape=(M, N)])
assert_type(csc_matrix(_csc_spec3), csc_matrix[ScalarType])
assert_type(csc_matrix(_csc_spec3, _shape_2d), csc_matrix[ScalarType])
assert_type(csc_matrix(_csc_spec3, shape=_shape_2d), csc_matrix[ScalarType])

assert_type(csc_matrix(_csc_spec3, dtype=_dtype), csc_matrix[ScalarType])
assert_type(csc_matrix(_csc_spec3, dtype=bool), csc_matrix[np.bool])
assert_type(csc_matrix(_csc_spec3, dtype=int), csc_matrix[np.int_])
assert_type(csc_matrix(_csc_spec3, dtype=float), csc_matrix[np.float64])
assert_type(csc_matrix(_csc_spec3, dtype=complex), csc_matrix[np.complex128])

###
# CSC array constructor

# csc_array(D)
assert_type(csc_array(_sc_2d), csc_array[ScalarType])
assert_type(csc_array(_sc_2d, dtype=_dtype), csc_array[ScalarType])
assert_type(csc_array(_sc_2d, dtype=bool), csc_array[np.bool])
assert_type(csc_array(_sc_2d, dtype=int), csc_array[np.int_])
assert_type(csc_array(_sc_2d, dtype=float), csc_array[np.float64])
assert_type(csc_array(_sc_2d, dtype=complex), csc_array[np.complex128])

# csc_array(S)
assert_type(csc_array(csr_arr), csc_array[ScalarType])
assert_type(csc_array(csr_mat), csc_array[ScalarType])

# csc_array((M, N), [dtype])
assert_type(csc_array(_shape_2d), csc_array[np.float64])
assert_type(csc_array(_shape_2d, dtype=_dtype), csc_array[ScalarType])
assert_type(csc_array(_shape_2d, dtype=bool), csc_array[np.bool])
assert_type(csc_array(_shape_2d, dtype=int), csc_array[np.int_])
assert_type(csc_array(_shape_2d, dtype=float), csc_array[np.float64])
assert_type(csc_array(_shape_2d, dtype=complex), csc_array[np.complex128])

# csc_array((data, (row_ind, col_ind)), [shape=(M, N)])
assert_type(csc_array(_csc_spec2), csc_array[ScalarType])
assert_type(csc_array(_csc_spec2, _shape_2d), csc_array[ScalarType])
assert_type(csc_array(_csc_spec2, shape=_shape_2d), csc_array[ScalarType])

assert_type(csc_array(_csc_spec2, dtype=_dtype), csc_array[ScalarType])
assert_type(csc_array(_csc_spec2, dtype=bool), csc_array[np.bool])
assert_type(csc_array(_csc_spec2, dtype=int), csc_array[np.int_])
assert_type(csc_array(_csc_spec2, dtype=float), csc_array[np.float64])
assert_type(csc_array(_csc_spec2, dtype=complex), csc_array[np.complex128])

# csc_array((data, indices, indptr), [shape=(M, N)])
assert_type(csc_array(_csc_spec3), csc_array[ScalarType])
assert_type(csc_array(_csc_spec3, _shape_2d), csc_array[ScalarType])
assert_type(csc_array(_csc_spec3, shape=_shape_2d), csc_array[ScalarType])

assert_type(csc_array(_csc_spec3, dtype=_dtype), csc_array[ScalarType])
assert_type(csc_array(_csc_spec3, dtype=bool), csc_array[np.bool])
assert_type(csc_array(_csc_spec3, dtype=int), csc_array[np.int_])
assert_type(csc_array(_csc_spec3, dtype=float), csc_array[np.float64])
assert_type(csc_array(_csc_spec3, dtype=complex), csc_array[np.complex128])

###
# +

# TODO

###
# -

# TODO

###
# *

# TODO

###
# @

# TODO

###
# /

assert_type(_csc_b / 1.0, csc_array[Any])
assert_type(_csc_i16 / 1.0, csc_array[Any])
assert_type(_csc_f32 / 1.0, csc_array[Any])
assert_type(_csc_f64 / 1.0, csc_array[np.float64])
assert_type(_csc_c64 / 1.0, csc_array[Any])
assert_type(_csc_c128 / 1.0, csc_array[np.complex128])
assert_type(_csc_b / 1j, csc_array[Any])
assert_type(_csc_i16 / 1j, csc_array[Any])
assert_type(_csc_f32 / 1j, csc_array[Any])
assert_type(_csc_f64 / 1j, csc_array[Any])
assert_type(_csc_c64 / 1j, csc_array[Any])
assert_type(_csc_c128 / 1j, csc_array[np.complex128])

assert_type(_csc_b / _b, csc_array[Any])
assert_type(_csc_i16 / _b, csc_array[Any])
assert_type(_csc_f32 / _b, csc_array[Any])
assert_type(_csc_f64 / _b, csc_array[np.float64])
assert_type(_csc_c64 / _b, csc_array[Any])
assert_type(_csc_c128 / _b, csc_array[np.complex128])
assert_type(_csc_b / _i16, csc_array[Any])
assert_type(_csc_i16 / _i16, csc_array[Any])
assert_type(_csc_f32 / _i16, csc_array[Any])
assert_type(_csc_f64 / _i16, csc_array[np.float64])
assert_type(_csc_c64 / _i16, csc_array[Any])
assert_type(_csc_c128 / _i16, csc_array[np.complex128])
assert_type(_csc_b / _f32, csc_array[Any])
assert_type(_csc_i16 / _f32, csc_array[Any])
assert_type(_csc_f32 / _f32, csc_array[Any])
assert_type(_csc_f64 / _f32, csc_array[np.float64])
assert_type(_csc_c64 / _f32, csc_array[Any])
assert_type(_csc_c128 / _f32, csc_array[np.complex128])
assert_type(_csc_b / _c64, csc_array[Any])
assert_type(_csc_i16 / _c64, csc_array[Any])
assert_type(_csc_f32 / _c64, csc_array[Any])
assert_type(_csc_f64 / _c64, csc_array[Any])
assert_type(_csc_c64 / _c64, csc_array[Any])
assert_type(_csc_c128 / _c64, csc_array[np.complex128])

assert_type(_csc_b / _csc_b, onp.Array2D[np.float64])
assert_type(_csc_i16 / _csc_b, onp.Array2D[np.float64])
assert_type(_csc_f32 / _csc_b, onp.Array2D[np.float64])
assert_type(_csc_f64 / _csc_b, onp.Array2D[np.float64])
assert_type(_csc_c64 / _csc_b, onp.Array2D[Any])
assert_type(_csc_c128 / _csc_b, onp.Array2D[Any])

assert_type(_csc_b / _csc_i16, onp.Array2D[np.float64])
assert_type(_csc_i16 / _csc_i16, onp.Array2D[np.float64])
assert_type(_csc_f32 / _csc_i16, onp.Array2D[np.float64])
assert_type(_csc_f64 / _csc_i16, onp.Array2D[np.float64])
assert_type(_csc_c64 / _csc_i16, onp.Array2D[Any])
assert_type(_csc_c128 / _csc_i16, onp.Array2D[Any])
assert_type(_csc_b / _csc_f32, onp.Array2D[np.float64])
assert_type(_csc_i16 / _csc_f32, onp.Array2D[np.float64])
assert_type(_csc_f32 / _csc_f32, onp.Array2D[np.float64])
assert_type(_csc_f64 / _csc_f32, onp.Array2D[np.float64])
assert_type(_csc_c64 / _csc_f32, onp.Array2D[Any])
assert_type(_csc_c128 / _csc_f32, onp.Array2D[Any])
assert_type(_csc_b / _csc_f64, onp.Array2D[np.float64])
assert_type(_csc_i16 / _csc_f64, onp.Array2D[np.float64])
assert_type(_csc_f32 / _csc_f64, onp.Array2D[np.float64])
assert_type(_csc_f64 / _csc_f64, onp.Array2D[np.float64])
assert_type(_csc_c64 / _csc_f64, onp.Array2D[Any])
assert_type(_csc_c128 / _csc_f64, onp.Array2D[Any])
assert_type(_csc_b / _csc_c64, onp.Array2D[Any])
assert_type(_csc_i16 / _csc_c64, onp.Array2D[Any])
assert_type(_csc_f32 / _csc_c64, onp.Array2D[Any])
assert_type(_csc_f64 / _csc_c64, onp.Array2D[Any])
assert_type(_csc_c64 / _csc_c64, onp.Array2D[Any])
assert_type(_csc_c128 / _csc_c64, onp.Array2D[Any])
assert_type(_csc_b / _csc_c128, onp.Array2D[Any])
assert_type(_csc_i16 / _csc_c128, onp.Array2D[Any])
assert_type(_csc_f32 / _csc_c128, onp.Array2D[Any])
assert_type(_csc_f64 / _csc_c128, onp.Array2D[Any])
assert_type(_csc_c64 / _csc_c128, onp.Array2D[Any])
assert_type(_csc_c128 / _csc_c128, onp.Array2D[Any])

assert_type(_csc_b / _i16_2d, coo_array[Any])
assert_type(_csc_i16 / _i16_2d, coo_array[Any])
assert_type(_csc_f32 / _i16_2d, coo_array[Any])
assert_type(_csc_f64 / _i16_2d, coo_array[Any])
assert_type(_csc_c64 / _i16_2d, coo_array[Any])
assert_type(_csc_c128 / _i16_2d, coo_array[Any])
assert_type(_csc_b / _f32_2d, coo_array[Any])
assert_type(_csc_i16 / _f32_2d, coo_array[Any])
assert_type(_csc_f32 / _f32_2d, coo_array[Any])
assert_type(_csc_f64 / _f32_2d, coo_array[Any])
assert_type(_csc_c64 / _f32_2d, coo_array[Any])
assert_type(_csc_c128 / _f32_2d, coo_array[Any])
assert_type(_csc_b / _f64_2d, coo_array[Any])
assert_type(_csc_i16 / _f64_2d, coo_array[Any])
assert_type(_csc_f32 / _f64_2d, coo_array[Any])
assert_type(_csc_f64 / _f64_2d, coo_array[Any])
assert_type(_csc_c64 / _f64_2d, coo_array[Any])
assert_type(_csc_c128 / _f64_2d, coo_array[Any])
assert_type(_csc_b / _c64_2d, coo_array[Any])
assert_type(_csc_i16 / _c64_2d, coo_array[Any])
assert_type(_csc_f32 / _c64_2d, coo_array[Any])
assert_type(_csc_f64 / _c64_2d, coo_array[Any])
assert_type(_csc_c64 / _c64_2d, coo_array[Any])
assert_type(_csc_c128 / _c64_2d, coo_array[Any])
assert_type(_csc_b / _c128_2d, coo_array[Any])
assert_type(_csc_i16 / _c128_2d, coo_array[Any])
assert_type(_csc_f32 / _c128_2d, coo_array[Any])
assert_type(_csc_f64 / _c128_2d, coo_array[Any])
assert_type(_csc_c64 / _c128_2d, coo_array[Any])
assert_type(_csc_c128 / _c128_2d, coo_array[Any])

###
# **

# TODO

###
# regression tests for https://github.com/scipy/scipy-stubs/issues/1412

assert_type(_csc_sc[0, 0], ScalarType)

assert_type(_csc_sc[_i64_1d], csc_array[ScalarType])
assert_type(_csc_sc[:, _i64_1d], csc_array[ScalarType])

assert_type(_csc_sc[_py_i_1d], csc_array[ScalarType])
assert_type(_csc_sc[:, _py_i_1d], csc_array[ScalarType])
assert_type(_csc_sc[_py_i_1d, _py_i_1d], onp.Array1D[ScalarType])

assert_type(_csc_sc[_bool_1d], csc_array[ScalarType])
assert_type(_csc_sc[:, _bool_1d], csc_array[ScalarType])
assert_type(_csc_sc[_bool_1d, _bool_1d], onp.Array1D[ScalarType])

assert_type(_csc_sc[_i64_1d, 0], coo_array[ScalarType, tuple[int]])
assert_type(_csc_sc[0, _i64_1d], coo_array[ScalarType, tuple[int]])

###
# regression tests for https://github.com/scipy/scipy-stubs/issues/1454

_arr_2d: onp.Array1D[ScalarType]

assert_type(_csc_sc[:, :], csc_array[ScalarType])
_csc_sc[0] = 1.0
_csc_sc[:] = 1.0
_csc_sc[:] = _arr_2d
_csc_sc[:, :] = 1.0
_csc_sc[:, :] = _arr_2d
_csc_sc[0, :] = 1.0
_csc_sc[0, :] = _arr_2d
_csc_sc[:, 0] = 1.0
_csc_sc[:, 0] = _arr_2d
_csc_sc[:, _py_i_1d] = 1.0
_csc_sc[:, _py_i_1d] = _arr_2d
_csc_sc[_py_i_1d, :] = 1.0
_csc_sc[_py_i_1d, :] = _arr_2d
_csc_sc[_py_i_1d, _py_i_1d] = 1.0
_csc_sc[_py_i_1d, _py_i_1d] = _arr_2d
