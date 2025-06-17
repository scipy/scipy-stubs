from typing import assert_type

import numpy as np
import optype.numpy as onp

from ._types import bsr_arr, bsr_mat, coo_arr, coo_mat, csc_arr, csc_mat, csr_arr, csr_mat, dia_arr, dia_mat
from scipy import sparse

i16_1d: onp.Array1D[np.int16]
i32_1d: onp.Array1D[np.int32]
u32_1d: onp.Array1D[np.uint32]
i64_1d: onp.Array1D[np.int64]

# get_index_dtype

assert_type(sparse.get_index_dtype(), type[np.int32])
assert_type(sparse.get_index_dtype((i16_1d, i16_1d)), type[np.int32])
assert_type(sparse.get_index_dtype((i32_1d, i32_1d)), type[np.int32])
assert_type(sparse.get_index_dtype((u32_1d, u32_1d)), type[np.int64])
assert_type(sparse.get_index_dtype((i64_1d, i64_1d)), type[np.int64])
assert_type(sparse.get_index_dtype((i32_1d, i64_1d)), type[np.int32 | np.int64])

# safely_cast_index_arrays

assert_type(sparse.safely_cast_index_arrays(bsr_arr), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(sparse.safely_cast_index_arrays(bsr_mat), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(sparse.safely_cast_index_arrays(csc_arr), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(sparse.safely_cast_index_arrays(csc_mat), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(sparse.safely_cast_index_arrays(csr_arr), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(sparse.safely_cast_index_arrays(csr_mat), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(sparse.safely_cast_index_arrays(coo_arr), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(sparse.safely_cast_index_arrays(coo_mat), tuple[onp.Array1D[np.int32], onp.Array1D[np.int32]])
assert_type(sparse.safely_cast_index_arrays(dia_arr), onp.Array1D[np.int32])
assert_type(sparse.safely_cast_index_arrays(dia_mat), onp.Array1D[np.int32])
