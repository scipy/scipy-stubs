# type-tests for `stats/_sampling.pyi`

from typing import assert_type

import numpy as np
import optype.numpy as onp

from scipy.stats.sampling import RatioUniforms

def _pdf(x: onp.ToFloat) -> onp.ToFloat: ...

_ru = RatioUniforms(_pdf, umax=1.0, vmin=-1.0, vmax=1.0)

assert_type(_ru, RatioUniforms)
assert_type(_ru.rvs(size=3), np.float64 | onp.ArrayND[np.float64])
