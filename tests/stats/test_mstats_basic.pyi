# type-tests for `moment` from `stats/_mstats_basic.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats.mstats import count_tied_groups

###
_x: onp.ToFloatND
assert_type(count_tied_groups(_x), dict[np.intp, np.intp | int])
