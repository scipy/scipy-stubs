# type-tests for `stats/mstats.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats.mstats import tmax, tmean, tmin

###

_py_i_1d: list[int]
_f32_1d: onp.Array1D[np.float32]

###

assert_type(tmax(_py_i_1d), np.int_ | onp.MArray[np.int_])
assert_type(tmax(_f32_1d), np.float32 | onp.MArray[np.float32])

assert_type(tmean(_f32_1d), np.float32 | onp.MArray[np.float32])

assert_type(tmin(_py_i_1d), np.int_ | onp.MArray[np.int_])
assert_type(tmin(_f32_1d), np.float32 | onp.MArray[np.float32])
