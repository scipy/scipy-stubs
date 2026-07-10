from typing import Literal, assert_type

import numpy as np
import optype.numpy as onp

from ._types import ScalarType, bsr_arr, bsr_mat, csr_arr, csr_mat
from scipy.sparse import bsr_array, bsr_matrix, isspmatrix_bsr

###

_seq_seq_bool: list[list[bool]]
_seq_seq_int: list[list[int]]
_seq_seq_float: list[list[float]]
_seq_seq_complex: list[list[complex]]

_dtype: np.dtype[ScalarType]

_shape2: tuple[int, int]
_data2: onp.Array2D[ScalarType]
_data2_concrete: onp.Array2D[np.float32]

_bsr_spec3: tuple[onp.ArrayND[ScalarType], onp.Array1D[np.intp], onp.Array1D[np.intp]]

###

# pyrefly: ignore [no-matching-overload]
bsr_array(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]
# pyrefly: ignore [no-matching-overload]
bsr_matrix(1)  # type: ignore[call-overload]  # pyright: ignore[reportArgumentType, reportCallIssue]

###
# (M, N) shape constructor

assert_type(bsr_array(_shape2), bsr_array[np.float64])
assert_type(bsr_array(_shape2, dtype=np.bool_), bsr_array[np.bool_])
assert_type(bsr_array(_shape2, dtype=bool), bsr_array[np.bool_])
assert_type(bsr_array(_shape2, dtype="bool"), bsr_array[np.bool_])
assert_type(bsr_array(_shape2, dtype=np.int64), bsr_array[np.int64])
assert_type(bsr_array(_shape2, dtype=int), bsr_array[np.int64])
assert_type(bsr_array(_shape2, dtype="int"), bsr_array[np.int64])
assert_type(bsr_array(_shape2, dtype=np.float64), bsr_array[np.float64])
assert_type(bsr_array(_shape2, dtype=float), bsr_array[np.float64])
assert_type(bsr_array(_shape2, dtype="float"), bsr_array[np.float64])
assert_type(bsr_array(_shape2, dtype=np.complex128), bsr_array[np.complex128])
assert_type(bsr_array(_shape2, dtype=complex), bsr_array[np.complex128])
assert_type(bsr_array(_shape2, dtype="complex"), bsr_array[np.complex128])
assert_type(bsr_array(_shape2, None, None), bsr_array[np.float64])
assert_type(bsr_array(_shape2, None, np.bool_), bsr_array[np.bool_])
assert_type(bsr_array(_shape2, None, np.int64), bsr_array[np.int64])
assert_type(bsr_array(_shape2, None, np.float64), bsr_array[np.float64])
assert_type(bsr_array(_shape2, None, np.complex128), bsr_array[np.complex128])

# `bsr_matrix` lacks the dedicated shape+dtype overloads that `bsr_array` has
assert_type(bsr_matrix(_shape2), bsr_matrix[np.float64])

###
# matrix-like (sequences) # noqa: ERA001

assert_type(bsr_array(_seq_seq_bool), bsr_array[np.bool_])
assert_type(bsr_array(_seq_seq_int), bsr_array[np.int_])
assert_type(bsr_array(_seq_seq_float), bsr_array[np.float64])
assert_type(bsr_array(_seq_seq_complex), bsr_array[np.complex128])

assert_type(bsr_matrix(_seq_seq_bool), bsr_matrix[np.bool_])
assert_type(bsr_matrix(_seq_seq_int), bsr_matrix[np.int_])
assert_type(bsr_matrix(_seq_seq_float), bsr_matrix[np.float64])
assert_type(bsr_matrix(_seq_seq_complex), bsr_matrix[np.complex128])

###
# matrix-like (dense ndarray)

assert_type(bsr_array(_data2), bsr_array[ScalarType])
assert_type(bsr_array(_data2, dtype=_dtype), bsr_array[ScalarType])
assert_type(bsr_array(_data2_concrete, dtype=np.bool_), bsr_array[np.bool_])
assert_type(bsr_array(_data2_concrete, dtype=np.int_), bsr_array[np.int_])
assert_type(bsr_array(_data2_concrete, dtype=np.float64), bsr_array[np.float64])
assert_type(bsr_array(_data2_concrete, dtype=np.complex128), bsr_array[np.complex128])

assert_type(bsr_matrix(_data2), bsr_matrix[ScalarType])
assert_type(bsr_matrix(_data2, dtype=_dtype), bsr_matrix[ScalarType])
assert_type(bsr_matrix(_data2_concrete, dtype=np.bool_), bsr_matrix[np.bool_])
assert_type(bsr_matrix(_data2_concrete, dtype=np.int_), bsr_matrix[np.int_])
assert_type(bsr_matrix(_data2_concrete, dtype=np.float64), bsr_matrix[np.float64])
assert_type(bsr_matrix(_data2_concrete, dtype=np.complex128), bsr_matrix[np.complex128])

###
# sparse-to-sparse conversion

assert_type(bsr_array(csr_arr), bsr_array[ScalarType])
assert_type(bsr_array(csr_mat), bsr_array[ScalarType])
assert_type(bsr_matrix(csr_arr), bsr_matrix[ScalarType])
assert_type(bsr_matrix(csr_mat), bsr_matrix[ScalarType])

###
# (data, indices, indptr), [shape=(M, N)], [blocksize]

# pyrefly: ignore [assert-type]
assert_type(bsr_array(_bsr_spec3), bsr_array[ScalarType])
# pyrefly: ignore [assert-type]
assert_type(bsr_array(_bsr_spec3, _shape2), bsr_array[ScalarType])
# pyrefly: ignore [assert-type]
assert_type(bsr_array(_bsr_spec3, shape=_shape2), bsr_array[ScalarType])
# pyrefly: ignore [assert-type]
assert_type(bsr_array(_bsr_spec3, blocksize=(2, 2)), bsr_array[ScalarType])

# pyrefly: ignore [assert-type]
assert_type(bsr_matrix(_bsr_spec3), bsr_matrix[ScalarType])
# pyrefly: ignore [assert-type]
assert_type(bsr_matrix(_bsr_spec3, _shape2), bsr_matrix[ScalarType])
# pyrefly: ignore [assert-type]
assert_type(bsr_matrix(_bsr_spec3, shape=_shape2), bsr_matrix[ScalarType])
# pyrefly: ignore [assert-type]
assert_type(bsr_matrix(_bsr_spec3, blocksize=(2, 2)), bsr_matrix[ScalarType])

###
# BSR-specific tests

# .format
assert_type(bsr_arr.format, Literal["bsr"])
assert_type(bsr_mat.format, Literal["bsr"])

# .ndim (BSR is always 2-D)
assert_type(bsr_arr.ndim, Literal[2])
assert_type(bsr_mat.ndim, Literal[2])

# .shape
assert_type(bsr_arr.shape, tuple[int, int])
assert_type(bsr_mat.shape, tuple[int, int])

# .blocksize
assert_type(bsr_arr.blocksize, tuple[int, int])
assert_type(bsr_mat.blocksize, tuple[int, int])

# .data (3-D block data)
assert_type(bsr_arr.data, onp.Array3D[ScalarType])
assert_type(bsr_mat.data, onp.Array3D[ScalarType])

# .count_nonzero()
assert_type(bsr_arr.count_nonzero(), np.intp)
assert_type(bsr_mat.count_nonzero(), np.intp)

# T / transpose (BSR stays BSR)
assert_type(bsr_arr.T, bsr_array[ScalarType])
assert_type(bsr_arr.transpose(), bsr_array[ScalarType])
assert_type(bsr_mat.T, bsr_matrix[ScalarType])
assert_type(bsr_mat.transpose(), bsr_matrix[ScalarType])

# isspmatrix_bsr
assert_type(isspmatrix_bsr(bsr_arr), bool)
assert_type(isspmatrix_bsr(bsr_mat), bool)
assert_type(isspmatrix_bsr(object()), bool)

###
# blocksize with matrix-like data (not just _bsr_spec3)

assert_type(bsr_array(_seq_seq_int, blocksize=(2, 2)), bsr_array[np.int_])
assert_type(bsr_array(_data2, blocksize=(2, 2)), bsr_array[ScalarType])
assert_type(bsr_matrix(_seq_seq_int, blocksize=(2, 2)), bsr_matrix[np.int_])
assert_type(bsr_matrix(_data2, blocksize=(2, 2)), bsr_matrix[ScalarType])

###
# (M, N) shape constructor, dtype positional (requires shape positional too)

assert_type(bsr_array(_shape2, _shape2, np.float64), bsr_array[np.float64])
assert_type(bsr_array(_shape2, _shape2, np.bool_), bsr_array[np.bool_])
assert_type(bsr_array(_shape2, _shape2, np.int64), bsr_array[np.int64])
assert_type(bsr_array(_shape2, _shape2, np.complex128), bsr_array[np.complex128])

###
# bsr_matrix (M, N) shape constructor — only supports dtype=None per stub

assert_type(bsr_matrix(_shape2, None, None), bsr_matrix[np.float64])

###
# __getitem__ / __setitem__ are `Never` (BSR does not support indexing)

# pyrefly: ignore [bad-index]
bsr_arr[0]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
# pyrefly: ignore [bad-index]
bsr_mat[0]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
