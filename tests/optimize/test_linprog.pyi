from collections.abc import Sequence
from typing import assert_type

import numpy as np

from scipy.optimize import OptimizeResult, linprog
from scipy.optimize._typing import Bound

c: list[float]

bound: Bound
bounds: Sequence[Bound]

###

res = linprog(c, bounds=bounds)
assert_type(res.fun, float | None)
assert_type(res.x, np.ndarray[tuple[int], np.dtype[np.float64]] | None)
assert_type(res.success, bool)
assert_type(res.message, str)

_1: OptimizeResult[np.float64] = linprog(c, bounds=bound)
_2: OptimizeResult[np.float64] = linprog(c, bounds=bounds)
