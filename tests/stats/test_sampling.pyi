# type-tests for `stats/_sampling.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats.sampling import RatioUniforms

###

_1d: tuple[int]
_2d: tuple[int, int]
_3d: tuple[int, int, int]

def _f(x: onp.ArrayND[np.float64]) -> list[float]: ...

###
# RatioUniforms

_ru = RatioUniforms(_f, umax=1.0, vmin=-1.0, vmax=1.0)
assert_type(_ru, RatioUniforms)
assert_type(_ru.rvs(), onp.Array1D[np.float64])
assert_type(_ru.rvs(3), onp.Array1D[np.float64])
assert_type(_ru.rvs(3), onp.Array1D[np.float64])
assert_type(_ru.rvs(_1d), onp.Array1D[np.float64])
assert_type(_ru.rvs(_2d), onp.Array2D[np.float64])
assert_type(_ru.rvs(_3d), onp.Array3D[np.float64])
