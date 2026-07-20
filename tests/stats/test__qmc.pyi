# type-tests for `stats/_qmc.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats.qmc import Halton

###
_d: onp.ToJustInt

_h1 = Halton(_d)
assert_type(_h1, Halton)

_h2 = Halton(d=_d)
assert_type(_h2, Halton)

_h3 = Halton(_d, scramble=False)
assert_type(_h3, Halton)

_h4 = Halton(_d, rng=0)
assert_type(_h4, Halton)

_h5 = Halton(_d, seed=0)
assert_type(_h5, Halton)

assert_type(_h1.base, list[int])
assert_type(_h1.scramble, bool)
assert_type(_h1._permutations, list[onp.Array2D[np.int_]])  # noqa: SLF001
