from typing import Literal, LiteralString, assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize import milp

###
# milp

_res = milp([1.0, 2.0])
assert_type(_res.x, onp.Array1D[np.float64] | None)
assert_type(_res.fun, float | np.float64 | None)
assert_type(_res.status, Literal[0, 1, 2, 3, 4])
assert_type(_res.message, LiteralString)
assert_type(_res.success, bool)
assert_type(_res.mip_node_count, int | None)
assert_type(_res.mip_dual_bound, float | np.float64 | None)
assert_type(_res.mip_gap, float | np.float64 | None)
