# type-tests for `boxcox*` from `stats/_morestats.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import boxcox, boxcox_normmax

###

_i8_1d: onp.Array1D[np.int8]
_f16_1d: onp.Array1D[np.float16]
_f32_1d: onp.Array1D[np.float32]
_f64_1d: onp.Array1D[np.float64]

###

# boxcox
assert_type(boxcox(_i8_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(boxcox(_f16_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(boxcox(_f32_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(boxcox(_f64_1d), tuple[onp.Array1D[np.float64], np.float64])
assert_type(boxcox(_i8_1d, alpha=0.1), tuple[onp.Array1D[np.float64], np.float64, tuple[float, float]])
assert_type(boxcox(_f16_1d, alpha=0.1), tuple[onp.Array1D[np.float64], np.float64, tuple[float, float]])
assert_type(boxcox(_f32_1d, alpha=0.1), tuple[onp.Array1D[np.float64], np.float64, tuple[float, float]])
assert_type(boxcox(_f64_1d, alpha=0.1), tuple[onp.Array1D[np.float64], np.float64, tuple[float, float]])

# boxcox_normmax
assert_type(boxcox_normmax(_i8_1d), np.float64)
assert_type(boxcox_normmax(_f16_1d), np.float64)
assert_type(boxcox_normmax(_f32_1d), np.float64)
assert_type(boxcox_normmax(_f64_1d), np.float64)
assert_type(boxcox_normmax(_i8_1d, method="all"), onp.Array1D[np.float64])
assert_type(boxcox_normmax(_f16_1d, method="all"), onp.Array1D[np.float64])
assert_type(boxcox_normmax(_f32_1d, method="all"), onp.Array1D[np.float64])
assert_type(boxcox_normmax(_f64_1d, method="all"), onp.Array1D[np.float64])
