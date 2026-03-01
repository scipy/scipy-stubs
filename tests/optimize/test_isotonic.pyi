from typing import assert_type

from scipy.optimize import isotonic_regression
from scipy.optimize._isotonic import OptimizeResult

y1d: list[float]

assert_type(isotonic_regression(y1d), OptimizeResult)
assert_type(isotonic_regression(y1d, weights=y1d, increasing=False), OptimizeResult)
