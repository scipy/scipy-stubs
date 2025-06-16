from typing import assert_type

import numpy as np

import scipy.sparse as sparse
from ._types import ScalarType, any_arr, any_mat

# find
assert_type(
    sparse.find(any_mat),
    tuple[
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[ScalarType]],
    ],
)
assert_type(
    sparse.find(any_arr),
    tuple[
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[ScalarType]],
    ],
)

# TODO(jorenham): test the other functions in sparse._extract
