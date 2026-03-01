# type-tests for `linalg/_expm_frechet.pyi`

from typing import TypeAlias, assert_type

import numpy as np
import optype.numpy as onp
from optype.test import assert_subtype

from scipy.linalg import expm_cond, expm_frechet

###

f64_2d: onp.Array2D[np.float64]
f64_3d: onp.Array3D[np.float64]
f64_nd: onp.ArrayND[np.float64]
c128_2d: onp.Array2D[np.complex128]

_FreoResult: TypeAlias = (
    tuple[onp.ArrayND[np.float64], onp.ArrayND[np.float64]]
    | tuple[onp.ArrayND[np.float64] | onp.ArrayND[np.complex128], onp.ArrayND[np.complex128]]
)

###
# expm_frechet

assert_type(expm_frechet(f64_2d, f64_2d), _FreoResult)
assert_type(expm_frechet(c128_2d, c128_2d), _FreoResult)
assert_type(expm_frechet(f64_2d, f64_2d, compute_expm=True), _FreoResult)
assert_type(expm_frechet(f64_2d, f64_2d, "SPS", False), _FreoResult)
assert_type(expm_frechet(f64_2d, f64_2d, compute_expm=False), _FreoResult)

###
# expm_cond

assert_type(expm_cond(f64_2d), np.float64)
assert_type(expm_cond(f64_3d), onp.Array1D[np.float64])
assert_subtype[np.float64 | onp.ArrayND[np.float64]](expm_cond(f64_nd))
assert_type(expm_cond(c128_2d), np.float64)
