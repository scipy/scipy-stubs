# type-tests for `moment` from `stats/_mstats_basic.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats.mstats import argstoarray, count_tied_groups

###
_x: onp.ToFloatND
assert_type(count_tied_groups(_x), dict[np.intp, np.intp | int])

###
_a: onp.ToFloatND
assert_type(argstoarray(_a), onp.MArray[np.float64])

_b: onp.ToFloatND
assert_type(argstoarray(_a, _b), onp.MArray[np.float64])

_c: onp.ToFloatND
assert_type(argstoarray(_a, _b, _c), onp.MArray[np.float64])
