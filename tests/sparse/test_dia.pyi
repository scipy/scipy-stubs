from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse import dia_array, dia_matrix

###

_shape2: tuple[int, int]
_shape_nd: tuple[int, ...]
_py_i: int
_py_i_1d: list[int]
_f64_1d: onp.Array1D[np.float64]
_f64_2d: onp.Array2D[np.float64]

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
