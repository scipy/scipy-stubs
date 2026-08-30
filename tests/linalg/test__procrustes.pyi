# type-tests for `linalg/_procrustes.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import orthogonal_procrustes

###

_py_i_2d: list[list[int]]
_f32_nd: onp.ArrayND[np.float32]
_f64_nd: onp.ArrayND[np.float64]
_c64_nd: onp.ArrayND[np.complex64]
_c128_nd: onp.ArrayND[np.complex128]

###
# orthogonal_procrustes

assert_type(orthogonal_procrustes(_py_i_2d, _f64_nd), tuple[onp.ArrayND[np.float64], np.float64])
assert_type(orthogonal_procrustes(_f32_nd, _f32_nd), tuple[onp.ArrayND[np.float32], np.float32])
assert_type(orthogonal_procrustes(_f32_nd, _f64_nd), tuple[onp.ArrayND[np.float64], np.float64])
assert_type(orthogonal_procrustes(_f32_nd, _c64_nd), tuple[onp.ArrayND[np.complex64], np.float32])
assert_type(orthogonal_procrustes(_f64_nd, _py_i_2d), tuple[onp.ArrayND[np.float64], np.float64])
assert_type(orthogonal_procrustes(_f64_nd, _f64_nd), tuple[onp.ArrayND[np.float64], np.float64])
assert_type(orthogonal_procrustes(_f64_nd, _c64_nd), tuple[onp.ArrayND[np.complex128], np.float64])
assert_type(orthogonal_procrustes(_c64_nd, _f64_nd), tuple[onp.ArrayND[np.complex128], np.float64])
assert_type(orthogonal_procrustes(_c64_nd, _c64_nd), tuple[onp.ArrayND[np.complex64], np.float32])
assert_type(orthogonal_procrustes(_c64_nd, _c128_nd), tuple[onp.ArrayND[np.complex128], np.float64])
assert_type(orthogonal_procrustes(_c128_nd, _f64_nd), tuple[onp.ArrayND[np.complex128], np.float64])
assert_type(orthogonal_procrustes(_c128_nd, _c128_nd), tuple[onp.ArrayND[np.complex128], np.float64])
