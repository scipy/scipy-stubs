# type-tests for `linalg/_procrustes.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.linalg import orthogonal_procrustes

###

f64_nd: onp.ArrayND[np.float64]
f32_nd: onp.ArrayND[np.float32]
c128_nd: onp.ArrayND[np.complex128]
c64_nd: onp.ArrayND[np.complex64]
py_i_2d: list[list[int]]

###
# orthogonal_procrustes

assert_type(orthogonal_procrustes(f64_nd, f64_nd), tuple[onp.ArrayND[np.float32 | np.float64], np.float32 | np.float64])
assert_type(orthogonal_procrustes(py_i_2d, f64_nd), tuple[onp.ArrayND[np.float32 | np.float64], np.float32 | np.float64])
assert_type(orthogonal_procrustes(f64_nd, py_i_2d), tuple[onp.ArrayND[np.float32 | np.float64], np.float32 | np.float64])
assert_type(orthogonal_procrustes(f32_nd, f32_nd), tuple[onp.ArrayND[np.float32 | np.float64], np.float32 | np.float64])
assert_type(orthogonal_procrustes(c128_nd, c128_nd), tuple[onp.ArrayND[np.complex128], np.float64])
assert_type(orthogonal_procrustes(c128_nd, f64_nd), tuple[onp.ArrayND[np.complex128], np.float64])
assert_type(orthogonal_procrustes(c64_nd, c64_nd), tuple[onp.ArrayND[np.complex64 | np.complex128], np.float32 | np.float64])
