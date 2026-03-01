from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.sparse.linalg import svds

a_f32: onp.Array2D[np.float32]
a_f64: onp.Array2D[np.float64]
a_c64: onp.Array2D[np.complex64]
a_c128: onp.Array2D[np.complex128]

# svds
assert_type(svds(a_f32), tuple[onp.Array2D[np.float32], onp.ArrayND[np.float32 | np.float64], onp.ArrayND[np.float32]])
assert_type(svds(a_f64), tuple[onp.Array2D[np.float64], onp.ArrayND[np.float32 | np.float64], onp.ArrayND[np.float64]])
assert_type(svds(a_c64), tuple[onp.Array2D[np.complex64], onp.ArrayND[np.float32 | np.float64], onp.ArrayND[np.complex64]])
assert_type(svds(a_c128), tuple[onp.Array2D[np.complex128], onp.ArrayND[np.float32 | np.float64], onp.ArrayND[np.complex128]])
