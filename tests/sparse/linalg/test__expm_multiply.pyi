from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from scipy.sparse import coo_array
from scipy.sparse.linalg import LinearOperator, expm_multiply

_dense_i8_1d: np.ndarray[tuple[int], np.dtype[np.int8]]
_dense_i8_2d: np.ndarray[tuple[int, int], np.dtype[np.int8]]
_dense_i8_nd: npt.NDArray[np.int8]

_dense_f32_1d: np.ndarray[tuple[int], np.dtype[np.float32]]
_dense_f32_2d: np.ndarray[tuple[int, int], np.dtype[np.float32]]
_dense_f32_nd: npt.NDArray[np.float32]

_sparse_i8_1d: coo_array[np.int8, tuple[int]]
_sparse_i8_2d: coo_array[np.int8, tuple[int, int]]
_sparse_i8_nd: coo_array[np.int8]

_sparse_f32_1d: coo_array[np.float32, tuple[int]]
_sparse_f32_2d: coo_array[np.float32, tuple[int, int]]
_sparse_f32_nd: coo_array[np.float32]

_linop_i8: LinearOperator[np.int8]
_linop_f32: LinearOperator[np.float32]

#

assert_type(expm_multiply(_dense_i8_2d, _dense_f32_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(expm_multiply(_dense_i8_2d, _dense_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(expm_multiply(_dense_i8_2d, _dense_f32_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])

assert_type(expm_multiply(_dense_i8_nd, _dense_f32_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(expm_multiply(_dense_i8_nd, _dense_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(expm_multiply(_dense_i8_nd, _dense_f32_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])

assert_type(expm_multiply(_sparse_i8_2d, _dense_f32_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(expm_multiply(_sparse_i8_2d, _dense_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(expm_multiply(_sparse_i8_2d, _dense_f32_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])

assert_type(expm_multiply(_linop_i8, _dense_f32_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(expm_multiply(_linop_i8, _dense_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(expm_multiply(_linop_i8, _dense_f32_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])

#

assert_type(expm_multiply(_dense_i8_2d, _dense_i8_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(expm_multiply(_dense_i8_2d, _dense_i8_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(expm_multiply(_dense_i8_2d, _dense_i8_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])

assert_type(expm_multiply(_dense_i8_nd, _dense_i8_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(expm_multiply(_dense_i8_nd, _dense_i8_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(expm_multiply(_dense_i8_nd, _dense_i8_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])

assert_type(expm_multiply(_sparse_i8_2d, _dense_i8_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(expm_multiply(_sparse_i8_2d, _dense_i8_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(expm_multiply(_sparse_i8_2d, _dense_i8_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])

assert_type(expm_multiply(_linop_i8, _dense_i8_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(expm_multiply(_linop_i8, _dense_i8_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(expm_multiply(_linop_i8, _dense_i8_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])

#

assert_type(expm_multiply(_dense_f32_2d, _dense_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(expm_multiply(_dense_f32_2d, _dense_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(expm_multiply(_dense_f32_2d, _dense_f32_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float32]])

assert_type(expm_multiply(_dense_f32_nd, _dense_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(expm_multiply(_dense_f32_nd, _dense_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(expm_multiply(_dense_f32_nd, _dense_f32_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float32]])

assert_type(expm_multiply(_sparse_f32_2d, _dense_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(expm_multiply(_sparse_f32_2d, _dense_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(expm_multiply(_sparse_f32_2d, _dense_f32_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float32]])

assert_type(expm_multiply(_linop_f32, _dense_f32_1d), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(expm_multiply(_linop_f32, _dense_f32_2d), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(expm_multiply(_linop_f32, _dense_f32_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float32]])
