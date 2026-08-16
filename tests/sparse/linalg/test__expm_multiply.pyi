from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import coo_array
from scipy.sparse._base import _spbase
from scipy.sparse.linalg import LinearOperator, expm_multiply

###

_dense_i8_1d: onp.Array1D[np.int8]
_dense_i8_2d: onp.Array2D[np.int8]
_dense_i8_nd: onp.ArrayND[np.int8]

_dense_f32_1d: onp.Array1D[np.float32]
_dense_f32_2d: onp.Array2D[np.float32]
_dense_f32_nd: onp.ArrayND[np.float32]

_sparse_i8_1d: coo_array[np.int8, tuple[int]]
_sparse_i8_2d: coo_array[np.int8, tuple[int, int]]
_sparse_i8_nd: coo_array[np.int8]

_sparse_f32_1d: coo_array[np.float32, tuple[int]]
_sparse_f32_2d: coo_array[np.float32, tuple[int, int]]
_sparse_f32_nd: coo_array[np.float32]

_linop_i8: LinearOperator[np.int8]
_linop_f32: LinearOperator[np.float32]

###

assert_type(expm_multiply(_dense_i8_2d, _dense_f32_1d), onp.Array1D[np.float64])
assert_type(expm_multiply(_dense_i8_2d, _dense_f32_2d), onp.Array2D[np.float64])
assert_type(expm_multiply(_dense_i8_2d, _dense_f32_nd), onp.ArrayND[np.float64])

assert_type(expm_multiply(_dense_i8_nd, _dense_f32_1d), onp.Array1D[np.float64])
assert_type(expm_multiply(_dense_i8_nd, _dense_f32_2d), onp.Array2D[np.float64])
assert_type(expm_multiply(_dense_i8_nd, _dense_f32_nd), onp.ArrayND[np.float64])

assert_type(expm_multiply(_sparse_i8_2d, _dense_f32_1d), onp.Array1D[np.float64])
assert_type(expm_multiply(_sparse_i8_2d, _dense_f32_2d), onp.Array2D[np.float64])
assert_type(expm_multiply(_sparse_i8_2d, _dense_f32_nd), onp.ArrayND[np.float64])

assert_type(expm_multiply(_linop_i8, _dense_f32_1d), onp.Array1D[np.float64])
assert_type(expm_multiply(_linop_i8, _dense_f32_2d), onp.Array2D[np.float64])
assert_type(expm_multiply(_linop_i8, _dense_f32_nd), onp.ArrayND[np.float64])

assert_type(expm_multiply(_dense_i8_2d, _dense_i8_1d), onp.Array1D[np.float64])
assert_type(expm_multiply(_dense_i8_2d, _dense_i8_2d), onp.Array2D[np.float64])
assert_type(expm_multiply(_dense_i8_2d, _dense_i8_nd), onp.ArrayND[np.float64])

assert_type(expm_multiply(_dense_i8_nd, _dense_i8_1d), onp.Array1D[np.float64])
assert_type(expm_multiply(_dense_i8_nd, _dense_i8_2d), onp.Array2D[np.float64])
assert_type(expm_multiply(_dense_i8_nd, _dense_i8_nd), onp.ArrayND[np.float64])

assert_type(expm_multiply(_sparse_i8_2d, _dense_i8_1d), onp.Array1D[np.float64])
assert_type(expm_multiply(_sparse_i8_2d, _dense_i8_2d), onp.Array2D[np.float64])
assert_type(expm_multiply(_sparse_i8_2d, _dense_i8_nd), onp.ArrayND[np.float64])

assert_type(expm_multiply(_linop_i8, _dense_i8_1d), onp.Array1D[np.float64])
assert_type(expm_multiply(_linop_i8, _dense_i8_2d), onp.Array2D[np.float64])
assert_type(expm_multiply(_linop_i8, _dense_i8_nd), onp.ArrayND[np.float64])

assert_type(expm_multiply(_dense_f32_2d, _dense_f32_1d), onp.Array1D[np.float32])
assert_type(expm_multiply(_dense_f32_2d, _dense_f32_2d), onp.Array2D[np.float32])
assert_type(expm_multiply(_dense_f32_2d, _dense_f32_nd), onp.ArrayND[np.float32])

assert_type(expm_multiply(_dense_f32_nd, _dense_f32_1d), onp.Array1D[np.float32])
assert_type(expm_multiply(_dense_f32_nd, _dense_f32_2d), onp.Array2D[np.float32])
assert_type(expm_multiply(_dense_f32_nd, _dense_f32_nd), onp.ArrayND[np.float32])

assert_type(expm_multiply(_sparse_f32_2d, _dense_f32_1d), onp.Array1D[np.float32])
assert_type(expm_multiply(_sparse_f32_2d, _dense_f32_2d), onp.Array2D[np.float32])
assert_type(expm_multiply(_sparse_f32_2d, _dense_f32_nd), onp.ArrayND[np.float32])

assert_type(expm_multiply(_linop_f32, _dense_f32_1d), onp.Array1D[np.float32])
assert_type(expm_multiply(_linop_f32, _dense_f32_2d), onp.Array2D[np.float32])
assert_type(expm_multiply(_linop_f32, _dense_f32_nd), onp.ArrayND[np.float32])

assert_type(expm_multiply(_dense_i8_2d, _sparse_f32_2d), _spbase[np.float64, tuple[int, int]])
assert_type(expm_multiply(_dense_f32_2d, _sparse_f32_2d), _spbase[np.float32, tuple[int, int]])
assert_type(expm_multiply(_linop_i8, _sparse_i8_2d), _spbase[np.float64, tuple[int, int]])

assert_type(expm_multiply(_dense_i8_2d, _dense_f32_1d, 0.0, 1.0), onp.ArrayND[np.float64])
assert_type(expm_multiply(_dense_i8_2d, _dense_f32_2d, start=0.0, stop=1.0, num=5), onp.ArrayND[np.float64])
assert_type(expm_multiply(_dense_i8_2d, _sparse_f32_2d, start=0.0, stop=1.0), onp.ArrayND[np.float64])
assert_type(expm_multiply(_dense_f32_2d, _dense_f32_1d, start=0.0, stop=1.0), onp.ArrayND[np.float32])
