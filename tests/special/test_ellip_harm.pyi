from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.special import ellip_harm, ellip_harm_2, ellip_normal

###

_i64_1d: onp.Array1D[np.int64]
_f64_1d: onp.Array1D[np.float64]

###

assert_type(ellip_harm(2.0, 3.0, 3, 4.0, 6), np.float64)
assert_type(ellip_harm(_f64_1d, 3.0, 3, 4.0, 6), onp.ArrayND[np.float64])
assert_type(ellip_harm(2.0, _f64_1d, 3, 4.0, 6), onp.ArrayND[np.float64])
assert_type(ellip_harm(2.0, 3.0, _i64_1d, 4.0, 6), onp.ArrayND[np.float64])
assert_type(ellip_harm(2.0, 3.0, 3, _f64_1d, 6), onp.ArrayND[np.float64])
assert_type(ellip_harm(2.0, 3.0, 3, 4.0, _f64_1d), onp.ArrayND[np.float64])

#
assert_type(ellip_harm_2(2.0, 3.0, 3, 4, 6.0), onp.Array0D[np.float64])
assert_type(ellip_harm_2(2.0, 3.0, 3, 4, _f64_1d), onp.ArrayND[np.float64])

#
assert_type(ellip_normal(2.0, 3.0, 3, 4), onp.Array0D[np.float64])
assert_type(ellip_normal(2.0, 3.0, _f64_1d, 4), onp.ArrayND[np.float64])
