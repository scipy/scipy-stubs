from typing import assert_type

import numpy as np

from scipy.sparse import dia_array, dia_matrix

###

_shape2: tuple[int, int]

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

assert_type(dia_matrix(_shape2), dia_matrix[np.float64])
assert_type(dia_matrix(_shape2, dtype=np.bool), dia_matrix[np.bool])
assert_type(dia_matrix(_shape2, dtype=np.int64), dia_matrix[np.int64])
assert_type(dia_matrix(_shape2, dtype=np.float64), dia_matrix[np.float64])
assert_type(dia_matrix(_shape2, dtype=np.complex128), dia_matrix[np.complex128])
assert_type(dia_matrix(_shape2, dtype=np.int8), dia_matrix[np.int8])
assert_type(dia_matrix(_shape2, dtype=np.uint8), dia_matrix[np.uint8])
assert_type(dia_matrix(_shape2, dtype=np.float32), dia_matrix[np.float32])
assert_type(dia_matrix(_shape2, dtype=np.complex64), dia_matrix[np.complex64])
