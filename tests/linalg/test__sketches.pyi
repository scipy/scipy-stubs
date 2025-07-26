from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
import optype.numpy as onp

from scipy.linalg import clarkson_woodruff_transform
from scipy.sparse import csc_matrix, sparray, spmatrix

like_bool_2d: list[list[bool]]
like_i64_2d: list[list[int]]
like_f64_2d: list[list[float]]
like_c128_2d: list[list[complex]]

arr_any: npt.NDArray[Any]
arr_i8: npt.NDArray[np.int8]
arr_f32: npt.NDArray[np.float32]
arr_c64: npt.NDArray[np.complex64]

sparse_any: spmatrix[Any] | sparray[Any]
sparse_i8: spmatrix[np.int8] | sparray[np.int8]
sparse_f32: spmatrix[np.float32] | sparray[np.float32]
sparse_c64: spmatrix[np.complex64] | sparray[np.complex64]

# NOTE: These extended precision types may not exist at runtime, but are used
# here to work around `[c]longdouble` issues on `numpy<2.2`
arr_f80: npt.NDArray[np.float128]
sparse_f80: spmatrix[np.float128] | sparray[np.float128]
arr_c160: npt.NDArray[np.complex256]
sparse_c160: spmatrix[np.complex256] | sparray[np.complex256]

###

assert_type(clarkson_woodruff_transform(like_bool_2d, 2), onp.ArrayND[np.int_])
assert_type(clarkson_woodruff_transform(like_i64_2d, 2), onp.ArrayND[np.int_])
assert_type(clarkson_woodruff_transform(like_f64_2d, 2), onp.ArrayND[np.float64])
assert_type(clarkson_woodruff_transform(like_c128_2d, 2), onp.ArrayND[np.complex128])

assert_type(clarkson_woodruff_transform(arr_any, 2), onp.ArrayND[Any])  # type: ignore[assert-type]
assert_type(clarkson_woodruff_transform(arr_i8, 2), onp.ArrayND[np.int_])
assert_type(clarkson_woodruff_transform(arr_f32, 2), onp.ArrayND[np.float64])
assert_type(clarkson_woodruff_transform(arr_c64, 2), onp.ArrayND[np.complex128])
assert_type(clarkson_woodruff_transform(arr_f80, 2), onp.ArrayND[np.longdouble])
assert_type(clarkson_woodruff_transform(arr_c160, 2), onp.ArrayND[np.clongdouble])

assert_type(clarkson_woodruff_transform(sparse_any, 2), csc_matrix[Any])
assert_type(clarkson_woodruff_transform(sparse_i8, 2), csc_matrix[np.int_])
assert_type(clarkson_woodruff_transform(sparse_f32, 2), csc_matrix[np.float64])
assert_type(clarkson_woodruff_transform(sparse_c64, 2), csc_matrix[np.complex128])
assert_type(clarkson_woodruff_transform(sparse_f80, 2), csc_matrix[np.longdouble])
assert_type(clarkson_woodruff_transform(sparse_c160, 2), csc_matrix[np.clongdouble])
