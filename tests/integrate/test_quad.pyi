from collections.abc import Callable
from typing import Literal
from typing_extensions import assert_type

import numpy as np
from scipy.integrate import quad
from scipy.integrate._quadpack_py import _QuadExplain
from scipy.integrate._typing import QuadInfoDict

TRUE: Literal[True] = True

# ufunc
assert_type(quad(np.exp, 0, 1), tuple[float | int | bool, float | int | bool])

# (float | int | bool) -> float | int | bool
f0_float_float: Callable[[float | int | bool], float | int | bool]
assert_type(quad(f0_float_float, 0, 1), tuple[float | int | bool, float | int | bool])

# (float | int | bool) -> np.float64
f0_float_f8: Callable[[float | int | bool], np.float64]
assert_type(quad(f0_float_f8, 0, 1), tuple[float | int | bool, float | int | bool])

# (np.float64) -> float | int | bool
f0_f8_float: Callable[[np.float64], float | int | bool]
assert_type(quad(f0_f8_float, 0, 1), tuple[float | int | bool, float | int | bool])

# (float | int | bool, str) -> float | int | bool
f1_float_float: Callable[[float | int | bool, str], float | int | bool]
assert_type(quad(f1_float_float, 0, 1, args=("",)), tuple[float | int | bool, float | int | bool])

# (float | int | bool, str, str) -> float | int | bool
f2_float_float: Callable[[float | int | bool, str, str], float | int | bool]
assert_type(quad(f2_float_float, 0, 1, args=("", "")), tuple[float | int | bool, float | int | bool])

# (float | int | bool) -> float | int | bool, full output
# NOTE: this test fails (only) in mypy due to some mypy bug
assert_type(
    quad(f0_float_float, 0, 1, full_output=TRUE),
    tuple[float | int | bool, float | int | bool, QuadInfoDict]
    | tuple[float | int | bool, float | int | bool, QuadInfoDict, str]
    | tuple[float | int | bool, float | int | bool, QuadInfoDict, str, _QuadExplain],
)

# (float | int | bool) -> complex | float | int | bool
# NOTE: this test fails (only) in mypy due to some mypy bug
z0_float_complex: Callable[[float | int | bool], complex | float | int | bool]
assert_type(quad(z0_float_complex, 0, 1, complex_func=TRUE), tuple[complex | float | int | bool, complex | float | int | bool])
