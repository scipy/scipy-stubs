from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.interpolate import RBFInterpolator

y2d: onp.Array2D[np.float64]
x2d: onp.Array2D[np.float64]
d1d_f: onp.Array1D[np.float64]
d1d_c: onp.Array1D[np.complex128]
d2d_f: onp.Array2D[np.float64]
d2d_c: onp.Array2D[np.complex128]

rbf_f1 = RBFInterpolator(y2d, d1d_f)
assert_type(rbf_f1, RBFInterpolator[np.float64, tuple[int]])
assert_type(rbf_f1(x2d), onp.ArrayND[np.float64])

rbf_c1 = RBFInterpolator(y2d, d1d_c)
assert_type(rbf_c1, RBFInterpolator[np.complex128, tuple[int]])
assert_type(rbf_c1(x2d), onp.ArrayND[np.complex128])

rbf_f2 = RBFInterpolator(y2d, d2d_f)
assert_type(rbf_f2, RBFInterpolator[np.float64, tuple[int, int]])
assert_type(rbf_f2(x2d), onp.ArrayND[np.float64])

rbf_c2 = RBFInterpolator(y2d, d2d_c)
assert_type(rbf_c2, RBFInterpolator[np.complex128, tuple[int, int]])
assert_type(rbf_c2(x2d), onp.ArrayND[np.complex128])
