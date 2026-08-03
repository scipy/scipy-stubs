# type-tests for `moment` from `stats/_mstats_basic.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats._mstats_basic import KruskalResult
from scipy.stats.mstats import argstoarray, count_tied_groups, kruskal

_nd: onp.ToFloatND

# count_tied_groups
assert_type(count_tied_groups(_nd), dict[np.intp, np.intp | int])

# argstoarray
assert_type(argstoarray(_nd), onp.MArray[np.float64])
assert_type(argstoarray(_nd, _nd), onp.MArray[np.float64])
assert_type(argstoarray(_nd, _nd, _nd), onp.MArray[np.float64])

# kruskal
assert_type(kruskal(_nd, _nd), KruskalResult)
assert_type(kruskal(_nd, _nd, _nd), KruskalResult)
