from typing import assert_type

import scipy.sparse.bsr as bsr

shape_2d: tuple[int, int]

assert_type(bsr.bsr_matrix(shape_2d), bsr.bsr_matrix)  # pyright: ignore[reportDeprecated]
assert_type(bsr.isspmatrix_bsr(bsr.bsr_matrix(shape_2d)), bool)  # pyright: ignore[reportDeprecated]
