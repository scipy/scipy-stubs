from typing import Any, Literal, assert_type

import optype.numpy as onp

from scipy.io import mminfo, mmread
from scipy.sparse import coo_array, coo_matrix

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
