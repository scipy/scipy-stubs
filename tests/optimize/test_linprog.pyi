from collections.abc import Sequence
from typing import assert_type

from scipy.optimize import OptimizeResult, linprog
from scipy.optimize._typing import Bound

c: list[float]

bound: Bound
uniformly_bounded = linprog(c, bounds=bound)
assert_type(uniformly_bounded, OptimizeResult)

bounds: Sequence[Bound]
variably_bounded = linprog(c, bounds=bounds)
assert_type(variably_bounded, OptimizeResult)
