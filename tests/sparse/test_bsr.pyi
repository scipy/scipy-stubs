from typing import assert_type

import numpy as np

import scipy.sparse as sparse

shape_2d: tuple[int, int]

assert_type(sparse.bsr_matrix(shape_2d), sparse.bsr_matrix[np.float64])
assert_type(sparse.isspmatrix_bsr(sparse.bsr_matrix(shape_2d)), bool)
