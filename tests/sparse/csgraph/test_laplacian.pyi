# type-tests for `sparse/csgraph/_laplacian.pyi`

from collections.abc import Callable
from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy import sparse
from scipy.sparse._base import _spbase
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import LinearOperator

###

type _Fn[ScalarT: np.generic] = Callable[[onp.ToComplex2D], onp.Array2D[ScalarT]]
type _AnyLaplacian = onp.Array2D[Any] | _spbase[Any, tuple[int, int]]

_f32_2d: onp.Array2D[np.float32]
_i64_2d: onp.Array2D[np.int64]
_f64_nd: onp.ArrayND[np.float64]
_seq_2d: list[list[float]]

_csr_arr: sparse.csr_array[np.float32, tuple[int, int]]
_csr_mat: sparse.csr_matrix[np.float32]
_dia_arr: sparse.dia_array[np.float32]
_dia_mat: sparse.dia_matrix[np.float32]
_csr_arr_i: sparse.csr_array[np.int64, tuple[int, int]]
_csr_mat_i: sparse.csr_matrix[np.int64]

###

assert_type(laplacian(_f32_2d), onp.Array2D[np.float32])
assert_type(laplacian(_f64_nd), onp.Array2D[np.float64])
assert_type(laplacian(_i64_2d), onp.Array2D[np.int64])
assert_type(laplacian(_f32_2d, symmetrized=True, copy=False), onp.Array2D[np.float32])

assert_type(laplacian(_i64_2d, True), onp.Array2D[np.float64])
assert_type(laplacian(_i64_2d, normed=True), onp.Array2D[np.float64])
assert_type(laplacian(_f32_2d, normed=True), onp.Array2D[np.float32])

assert_type(laplacian(_f32_2d, dtype=np.complex128), onp.Array2D[np.complex128])
assert_type(laplacian(_i64_2d, normed=True, dtype=np.float32), onp.Array2D[np.float32])
assert_type(laplacian(_f32_2d, dtype=float), _AnyLaplacian)

assert_type(laplacian(_csr_arr), sparse.coo_array[np.float32, tuple[int, int]])
assert_type(laplacian(_csr_mat), sparse.coo_matrix[np.float32])
assert_type(laplacian(_dia_arr), sparse.dia_array[np.float32])
assert_type(laplacian(_dia_mat), sparse.dia_matrix[np.float32])
assert_type(laplacian(_dia_arr, normed=True), sparse.coo_array[np.float32, tuple[int, int]])
assert_type(laplacian(_dia_mat, normed=True), sparse.coo_matrix[np.float32])
assert_type(laplacian(_dia_arr, dtype=np.int8), sparse.dia_array[np.int8])
assert_type(laplacian(_csr_arr, dtype=np.int8), sparse.coo_array[np.int8, tuple[int, int]])
assert_type(laplacian(_csr_arr_i, normed=True), sparse.coo_array[np.float64, tuple[int, int]])
assert_type(laplacian(_csr_mat_i, normed=True), sparse.coo_matrix[np.float64])

assert_type(laplacian(_f32_2d, return_diag=True), tuple[onp.Array2D[np.float32], onp.Array1D[np.float32]])
assert_type(laplacian(_i64_2d, normed=True, return_diag=True), tuple[onp.Array2D[np.float64], onp.Array1D[np.float64]])
assert_type(laplacian(_csr_arr, return_diag=True), tuple[sparse.coo_array[np.float32, tuple[int, int]], onp.Array1D[np.float32]])
assert_type(laplacian(_dia_mat, return_diag=True), tuple[sparse.dia_matrix[np.float32], onp.Array1D[np.float32]])
assert_type(laplacian(_f32_2d, return_diag=True, dtype=np.int16), tuple[onp.Array2D[np.int16], onp.Array1D[np.int16]])

assert_type(laplacian(_f32_2d, form="lo"), LinearOperator[np.float32, tuple[int, int]])
assert_type(laplacian(_csr_mat, form="lo"), LinearOperator[np.float32, tuple[int, int]])
assert_type(laplacian(_i64_2d, normed=True, form="lo"), LinearOperator[np.float64, tuple[int, int]])
assert_type(laplacian(_f32_2d, form="lo", dtype=np.complex64), LinearOperator[np.complex64, tuple[int, int]])
assert_type(
    laplacian(_csr_arr, form="lo", return_diag=True), tuple[LinearOperator[np.float32, tuple[int, int]], onp.Array1D[np.float32]]
)

assert_type(laplacian(_f32_2d, form="function"), _Fn[np.float32 | Any])
assert_type(laplacian(_f64_nd, form="function"), _Fn[np.float64 | Any])
assert_type(laplacian(_i64_2d, normed=True, form="function"), _Fn[np.float64 | Any])
assert_type(laplacian(_dia_arr, form="function", return_diag=True), tuple[_Fn[np.float32 | Any], onp.Array1D[np.float32]])
assert_type(laplacian(_f32_2d, form="function")(_f64_nd), onp.Array2D[np.float32 | Any])

assert_type(laplacian(_f32_2d, False, True), tuple[_AnyLaplacian, onp.Array1D[Any]])
assert_type(laplacian(_i64_2d, True, True), tuple[_AnyLaplacian, onp.Array1D[Any]])

assert_type(laplacian(_seq_2d), _AnyLaplacian)
