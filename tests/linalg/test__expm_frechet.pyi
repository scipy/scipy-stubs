# type-tests for `linalg/_expm_frechet.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.linalg import expm_cond, expm_frechet

###

_f64_2d: onp.Array2D[np.float64]
_f64_3d: onp.Array3D[np.float64]
_f64_nd: onp.ArrayND[np.float64]
_c128_2d: onp.Array2D[np.complex128]

###
# expm_frechet

assert_type(expm_frechet(_f64_2d, _f64_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(expm_frechet(_f64_2d, _f64_2d, compute_expm=True), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]])
assert_type(expm_frechet(_c128_2d, _c128_2d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(expm_frechet(_c128_2d, _f64_2d), tuple[onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]])
assert_type(expm_frechet(_f64_2d, _c128_2d), tuple[onp.ArrayND[np.float64], onp.ArrayND[np.complex128]])

assert_type(expm_frechet(_f64_2d, _f64_2d, "SPS", False), onp.ArrayND[np.float64])
assert_type(expm_frechet(_f64_2d, _f64_2d, compute_expm=False), onp.ArrayND[np.float64])
assert_type(expm_frechet(_c128_2d, _f64_2d, compute_expm=False), onp.ArrayND[np.complex128])
assert_type(expm_frechet(_f64_2d, _c128_2d, compute_expm=False), onp.ArrayND[np.complex128])
assert_type(expm_frechet(_c128_2d, _c128_2d, None, False), onp.ArrayND[np.complex128])

###
# expm_cond

assert_type(expm_cond(_f64_2d), np.float64)
assert_type(expm_cond(_f64_3d), onp.Array1D[np.float64])
assert_subtype[np.float64 | onp.ArrayND[np.float64]](expm_cond(_f64_nd))
assert_type(expm_cond(_c128_2d), np.float64)
