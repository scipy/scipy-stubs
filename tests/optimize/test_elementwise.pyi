from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.optimize._elementwise import _BracketResult, _FindResult
from scipy.optimize.elementwise import bracket_minimum, bracket_root, find_minimum, find_root

def f_1d(x: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...
def f_2d(x: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]: ...
def g_1d(x: onp.Array1D[np.float64], a: float) -> onp.Array1D[np.float64]: ...

# find_root
assert_type(find_root(f_1d, (-1.0, 1.0)), _FindResult[tuple[int]])
assert_type(find_root(f_2d, (-1.0, 1.0)), _FindResult[tuple[int, int]])
assert_type(find_root(g_1d, (-1.0, 1.0), args=(0.5,)), _FindResult[tuple[int]])

# find_minimum
assert_type(find_minimum(f_1d, (-1.0, 0.0, 1.0)), _FindResult[tuple[int]])
assert_type(find_minimum(f_2d, (-1.0, 0.0, 1.0)), _FindResult[tuple[int, int]])
assert_type(find_minimum(g_1d, (-1.0, 0.0, 1.0), args=(0.5,)), _FindResult[tuple[int]])

# bracket_root
assert_type(bracket_root(f_1d, -1.0), _BracketResult[tuple[int]])
assert_type(bracket_root(f_2d, -1.0), _BracketResult[tuple[int, int]])
assert_type(bracket_root(g_1d, -1.0, args=(0.5,)), _BracketResult[tuple[int]])

# bracket_minimum
assert_type(bracket_minimum(f_1d, 0.0), _BracketResult[tuple[int]])
assert_type(bracket_minimum(f_2d, 0.0), _BracketResult[tuple[int, int]])
assert_type(bracket_minimum(g_1d, 0.0, args=(0.5,)), _BracketResult[tuple[int]])
