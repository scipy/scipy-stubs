from typing import assert_type

import numpy as np
import numpy.typing as npt

from scipy.optimize import least_squares

# Regression test for a mypy bug, see:
# https://github.com/scipy/scipy-stubs/issues/939
# https://github.com/python/mypy/issues/20079

def example_residual(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

x0: npt.NDArray[np.float64]

result = least_squares(example_residual, x0=x0)
assert_type(result.x, np.ndarray[tuple[int], np.dtype[np.float64]])
