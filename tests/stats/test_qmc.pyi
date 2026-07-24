# type-tests for `stats/_qmc.pyi`

from typing import Any, assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats import qmc
from scipy.stats.qmc import Halton

###
_f8_xd: np.ndarray[Any, np.dtype[np.float64]]
qmc.scale(_f8_xd, 0, 1)

_f8_nd: onp.ArrayND[np.float64]
qmc.scale(_f8_nd, 0, 1)

_f8_2d: onp.Array2D[np.float64]
qmc.scale(_f8_2d, 0, 1)

assert_type(qmc.update_discrepancy(_f8_2d[0], _f8_2d, 0.5), float)

###
# Halton
_d: onp.ToJustInt

_h1 = Halton(_d)
assert_type(_h1, Halton)

assert_type(Halton(d=_d), Halton)
assert_type(Halton(_d, scramble=False), Halton)
assert_type(Halton(_d, rng=0), Halton)
assert_type(Halton(_d, seed=0), Halton)

assert_type(_h1.base, list[int])
assert_type(_h1.scramble, bool)
assert_type(_h1._permutations, list[onp.Array2D[np.int_]])  # ruff: ignore[private-member-access]
