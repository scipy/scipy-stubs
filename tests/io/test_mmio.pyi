from typing import Any, Literal, assert_type

import numpy as np
import optype.numpy as onp

from scipy.io import mminfo, mmread, mmwrite
from scipy.sparse import coo_array, coo_matrix

###

_arr_f64_2d: onp.Array2D[np.float64]
_coo_f64_2d: coo_array[np.float64] | coo_matrix[np.float64]

###

# mminfo
assert_type(
    mminfo("file.mtx"),
    tuple[
        int,
        int,
        int,
        Literal["coordinate", "array"],
        Literal["real", "complex", "pattern", "integer"],
        Literal["general", "symmetric", "skew-symmetric", "hermitian"],
    ],
)

# mmread
assert_type(mmread("file.mtx"), onp.Array2D | coo_matrix)
assert_type(mmread("file.mtx", spmatrix=True), onp.Array2D | coo_matrix)
assert_type(mmread("file.mtx", spmatrix=False), onp.Array2D | coo_array[Any, tuple[int, int]])

# mmwrite
assert_type(mmwrite("file_out.mtx", _arr_f64_2d), None)
assert_type(mmwrite("file_out.mtx", _coo_f64_2d), None)
